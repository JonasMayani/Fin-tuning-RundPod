"""
src/serving/api.py
─────────────────────────────────────────────────────────────────────────────
Phase 5.2 — Inference API
Multilingual African Health Assistant | Zindi ITU Challenge

Endpoints:
  POST /answer     — Submit a health question → receive an answer
  GET  /health     — Service health check
  POST /feedback   — Submit user feedback on an answer

Features:
  - Automatic language detection (lingua-py)
  - Per-language response disclaimers
  - Rate limiting: 100 req/min per IP
  - Anonymised request logging (no PII stored)
  - Localised disclaimers in 5 languages
"""

from __future__ import annotations

import hashlib
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ─── App Initialisation ───────────────────────────────────────────────────────

CONFIG_PATH = Path("src/training/config.yaml")
with open(config_path, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SERVING_CFG  = CFG["serving"]
MODEL_CFG    = CFG["model"]
DECODE_CFG   = CFG["decoding"]
NLLB_CODES   = CFG["nllb_codes"]
DISCLAIMERS  = SERVING_CFG["disclaimer"]

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="African Health Assistant API",
    description="Multilingual health Q&A for sub-Saharan communities",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Singleton ─────────────────────────────────────────────────────────

_model_cache: dict = {}

LANGUAGE_NAMES = {
    "Aka_Gha": "Akan",
    "Amh_Eth": "Amharic",
    "Lug_Uga": "Luganda",
    "Swa_Eas": "Swahili",
    "Eng":     "English",
}

# Reverse map: lingua/langdetect codes → subset tags
_DETECT_TO_SUBSET = {
    "am": "Amh_Eth",
    "sw": "Swa_Eas",
    "en": "Eng",
    # Akan and Luganda have limited detection support; default to Eng on fail
}


def get_model():
    """Load and cache the model (lazy init at first request)."""
    if "model" not in _model_cache:
        paths     = CFG["paths"]
        base_name = MODEL_CFG["base_model"].replace("/", "_")
        model_dir = Path(paths["models"]) / base_name / "best"

        if not model_dir.exists():
            raise RuntimeError(
                f"Model not found at {model_dir}. Run training pipeline first."
            )

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        logger.info(f"Loading model from {model_dir} …")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        _model_cache["model"]     = model
        _model_cache["tokenizer"] = tokenizer
        logger.success("Model loaded and ready.")
    return _model_cache["model"], _model_cache["tokenizer"]


# ─── Language Detection ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language subset tag from text. Returns 'Eng' as fallback."""
    try:
        from lingua import LanguageDetectorBuilder, Language
        detector = (
            LanguageDetectorBuilder
            .from_languages(Language.ENGLISH, Language.AMHARIC, Language.SWAHILI)
            .build()
        )
        result = detector.detect_language_of(text)
        if result:
            code = result.iso_code_639_1.name.lower()
            return _DETECT_TO_SUBSET.get(code, "Eng")
    except Exception:
        pass

    # Ethiopic script heuristic → Amharic
    import re
    if re.search(r"[\u1200-\u137F]", text):
        return "Amh_Eth"

    try:
        from langdetect import detect
        code = detect(text)
        return _DETECT_TO_SUBSET.get(code, "Eng")
    except Exception:
        return "Eng"


# ─── Inference ───────────────────────────────────────────────────────────────

def generate_answer(question: str, subset: str) -> tuple[str, float]:
    """
    Generate a health answer for the given question and language.
    Returns (answer_text, confidence_score).
    """
    model, tokenizer = get_model()

    lang_name = LANGUAGE_NAMES.get(subset, "English")
    prompt    = f"Answer in {lang_name}: {question}"

    dc  = DECODE_CFG["per_language"].get(subset, DECODE_CFG["default"])
    max_in  = MODEL_CFG["max_input_length"]
    max_out = MODEL_CFG["max_output_length"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_in,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            num_beams=dc.get("num_beams", 8),
            length_penalty=dc.get("length_penalty", 0.8),
            no_repeat_ngram_size=dc.get("no_repeat_ngram", 3),
            max_new_tokens=max_out,
            return_dict_in_generate=True,
            output_scores=True,
        )

    answer = tokenizer.decode(output.sequences[0], skip_special_tokens=True).strip()

    # Approximate confidence: mean of top-1 softmax scores per step
    try:
        import numpy as np
        step_probs = [
            torch.softmax(s, dim=-1).max().item()
            for s in output.scores
        ]
        confidence = float(np.mean(step_probs))
    except Exception:
        confidence = 0.85  # fallback

    return answer, confidence


# ─── Request / Response Schemas ───────────────────────────────────────────────

class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          description="Health question in any supported language")
    language: Optional[str] = Field(
        None,
        description="Optional language override: Aka_Gha | Amh_Eth | Lug_Uga | Swa_Eas | Eng"
    )


class AnswerResponse(BaseModel):
    answer:            str
    language_detected: str
    confidence:        float
    disclaimer:        str
    question_id:       str


class FeedbackRequest(BaseModel):
    question_id: str
    answer:      str
    rating:      int = Field(..., ge=1, le=5)
    comment:     Optional[str] = None


# ─── Anonymisation ───────────────────────────────────────────────────────────

def anonymise_ip(ip: str) -> str:
    """One-way hash IP address for logging (no PII stored)."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


# ─── Request Log ─────────────────────────────────────────────────────────────

_request_log: list[dict] = []   # in-memory log; flush to disk periodically in prod


def log_request(
    question_id: str,
    language:    str,
    latency_ms:  float,
    ip_hash:     str,
) -> None:
    _request_log.append({
        "question_id": question_id,
        "language":    language,
        "latency_ms":  round(latency_ms, 1),
        "ip_hash":     ip_hash,
        "timestamp":   datetime.utcnow().isoformat(),
    })


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/answer", response_model=AnswerResponse)
@limiter.limit(SERVING_CFG.get("rate_limit", "100/minute"))
async def answer_question(request: Request, body: AnswerRequest):
    """
    Submit a health question and receive an answer in the same language.
    Language is auto-detected if not provided.
    """
    t0 = time.perf_counter()
    question_id = str(uuid.uuid4())

    # Language detection / validation
    if body.language and body.language in LANGUAGE_NAMES:
        subset = body.language
    else:
        subset = detect_language(body.question)

    # Validate language
    if subset not in LANGUAGE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {subset}. Supported: {list(LANGUAGE_NAMES.keys())}"
        )

    try:
        answer, confidence = generate_answer(body.question, subset)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    disclaimer = DISCLAIMERS.get(subset, DISCLAIMERS["Eng"])

    latency_ms = (time.perf_counter() - t0) * 1000
    ip_hash    = anonymise_ip(request.client.host if request.client else "unknown")
    log_request(question_id, subset, latency_ms, ip_hash)

    return AnswerResponse(
        answer=answer,
        language_detected=subset,
        confidence=round(confidence, 4),
        disclaimer=disclaimer,
        question_id=question_id,
    )


@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "ok",
        "model_version": MODEL_CFG["base_model"],
        "languages_supported": list(LANGUAGE_NAMES.keys()),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/feedback")
@limiter.limit("30/minute")
async def submit_feedback(request: Request, body: FeedbackRequest):
    """
    Submit user feedback on a generated answer.
    Ratings ≤ 2 are flagged for clinical review.
    """
    is_flagged = body.rating <= 2
    record = {
        "question_id": body.question_id,
        "rating":      body.rating,
        "comment":     body.comment,
        "flagged":     is_flagged,
        "timestamp":   datetime.utcnow().isoformat(),
    }
    _request_log.append({"type": "feedback", **record})

    if is_flagged:
        logger.warning(f"Low-rated answer flagged for review: {body.question_id}")

    return {"logged": True, "flagged_for_review": is_flagged}


@app.get("/metrics")
async def metrics():
    """Internal metrics endpoint for monitoring."""
    if not _request_log:
        return {"message": "No requests logged yet."}

    answer_logs  = [r for r in _request_log if "language" in r]
    feedback_logs = [r for r in _request_log if r.get("type") == "feedback"]

    lang_counts  = {}
    lang_latency = {}
    for r in answer_logs:
        lang = r.get("language", "unknown")
        lang_counts[lang]  = lang_counts.get(lang, 0) + 1
        lang_latency.setdefault(lang, []).append(r.get("latency_ms", 0))

    return {
        "total_requests":       len(answer_logs),
        "total_feedback":       len(feedback_logs),
        "requests_by_language": lang_counts,
        "avg_latency_ms":       {
            lang: round(sum(v) / len(v), 1)
            for lang, v in lang_latency.items()
        },
        "flagged_answers":      sum(1 for r in feedback_logs if r.get("flagged")),
    }


# ─── Server Entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.api:app",
        host=SERVING_CFG.get("host", "0.0.0.0"),
        port=SERVING_CFG.get("port", 8000),
        reload=False,
        log_level="info",
    )
