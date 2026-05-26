"""
pipeline/utils.py — shared utilities used across the augmentation pipeline.

  • Script detection (which Unicode block is the text written in?)
  • Language ID via fastText lid.176 (free, fast, 176 languages)
  • LaBSE embeddings (multilingual, 109 languages, GOAT for QA similarity)
  • chrF scorer for paraphrase distance
  • MinHash near-duplicate detection

These are loaded lazily — only when first called — so importing this module
costs nothing.
"""
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Script detection
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from our language codes to the Unicode script we expect the text in.
# This catches the #1 noise source: text in the WRONG SCRIPT.
EXPECTED_SCRIPT: dict[str, str] = {
    "Eng_Uga": "LATIN",
    "Eng_Gha": "LATIN",
    "Eng_Eth": "LATIN",
    "Eng_Ken": "LATIN",
    "Aka_Gha": "LATIN",   # Akan uses Latin script
    "Lug_Uga": "LATIN",   # Luganda uses Latin script
    "Swa_Ken": "LATIN",   # Swahili uses Latin script
    "Amh_Eth": "ETHIOPIC",  # Amharic uses Ge'ez (Ethiopic) script
}


def detect_script(text: str) -> str:
    """Detect the dominant Unicode script in a string. Returns the script name
    of the script accounting for >60% of letter chars, or 'MIXED' otherwise."""
    if not text:
        return "EMPTY"
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue
        # The first token of the Unicode name is usually the script.
        # "ETHIOPIC SYLLABLE GA" → "ETHIOPIC"
        # "LATIN SMALL LETTER A" → "LATIN"
        script = name.split(" ", 1)[0] if name else "UNKNOWN"
        counts[script] = counts.get(script, 0) + 1
    if total == 0:
        return "EMPTY"
    top, top_count = max(counts.items(), key=lambda kv: kv[1])
    return top if top_count / total >= 0.6 else "MIXED"


def script_ok(text: str, language: str) -> bool:
    """True if `text` is in the expected script for `language`."""
    expected = EXPECTED_SCRIPT.get(language)
    if not expected:
        return True
    return detect_script(text) == expected


# ──────────────────────────────────────────────────────────────────────────────
# fastText language ID (free, no API needed)
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from our language codes to the ISO-639 codes that fastText uses.
# Notes:
#   - fastText lid.176 doesn't have Akan (`aka`) — closest is `tw` (Twi) which
#     is the main Akan variant. We allow either.
#   - Luganda is `lg`. Amharic is `am`. Swahili is `sw`.
FASTTEXT_ACCEPT: dict[str, set[str]] = {
    "Eng_Uga": {"en"}, "Eng_Gha": {"en"}, "Eng_Eth": {"en"}, "Eng_Ken": {"en"},
    "Aka_Gha": {"tw", "ak"},      # Twi (Asante/Akuapem) or Akan
    "Lug_Uga": {"lg"},
    "Swa_Ken": {"sw"},
    "Amh_Eth": {"am"},
}

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = Path(os.environ.get(
    "FASTTEXT_LID_PATH", "/workspace/.hf_cache/lid.176.bin"))


@lru_cache(maxsize=1)
def _load_fasttext():
    """Load fastText lid.176, downloading if needed (~125 MB)."""
    try:
        import fasttext
    except ImportError:
        raise ImportError(
            "fasttext not installed. Add to requirements: fasttext-wheel==0.9.2"
        )

    if not FASTTEXT_MODEL_PATH.exists():
        import urllib.request
        FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading fastText lid.176 → {FASTTEXT_MODEL_PATH}")
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)

    # fastText prints a deprecation banner to stderr; suppress noise.
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        return fasttext.load_model(str(FASTTEXT_MODEL_PATH))


def detect_language(text: str) -> tuple[str, float]:
    """Return (lang_code, confidence). Returns ('und', 0.0) for empty/very short."""
    text = (text or "").strip().replace("\n", " ")
    if len(text) < 5:
        return "und", 0.0
    try:
        model = _load_fasttext()
        labels, probs = model.predict(text, k=1)
        # fastText returns labels like "__label__en"
        code = labels[0].replace("__label__", "")
        return code, float(probs[0])
    except Exception as exc:
        logger.warning(f"langid failed on '{text[:50]}...': {exc}")
        return "und", 0.0


def language_ok(text: str, language: str, min_confidence: float = 0.5) -> bool:
    """True if fastText thinks `text` is in the expected language family."""
    code, conf = detect_language(text)
    accept = FASTTEXT_ACCEPT.get(language, set())
    return code in accept and conf >= min_confidence


# ──────────────────────────────────────────────────────────────────────────────
# LaBSE embeddings — multilingual sentence similarity
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_labse():
    """Load LaBSE on CUDA, bf16 for speed."""
    import torch
    from sentence_transformers import SentenceTransformer
    logger.info("Loading LaBSE (sentence-transformers/LaBSE)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    # half-precision for ~2× throughput on A100, no quality loss in practice
    if device == "cuda":
        model = model.half()
    return model


def embed_texts(texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Return L2-normalized embeddings as a (N, 768) float32 array."""
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    model = _load_labse()
    embs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return embs.astype(np.float32)


def cosine_sim(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two L2-normalized matrices."""
    return (emb_a * emb_b).sum(axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# chrF (character F-score) — surface similarity that works cross-lingually
# ──────────────────────────────────────────────────────────────────────────────

def chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF score in [0, 1]. Works across scripts (char-level, no tokenizer)."""
    if not reference or not hypothesis:
        return 0.0
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    if not ref or not hyp:
        return 0.0
    f_total = 0.0; valid_n = 0
    for k in range(1, n + 1):
        ref_ngrams = [ref[i:i+k] for i in range(len(ref) - k + 1)]
        hyp_ngrams = [hyp[i:i+k] for i in range(len(hyp) - k + 1)]
        if not ref_ngrams or not hyp_ngrams:
            continue
        from collections import Counter
        ref_c, hyp_c = Counter(ref_ngrams), Counter(hyp_ngrams)
        overlap = sum((ref_c & hyp_c).values())
        p = overlap / sum(hyp_c.values()) if hyp_c else 0.0
        r = overlap / sum(ref_c.values()) if ref_c else 0.0
        if p + r == 0:
            continue
        f = (1 + beta**2) * p * r / (beta**2 * p + r)
        f_total += f; valid_n += 1
    return f_total / valid_n if valid_n else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# MinHash near-duplicate detection
# ──────────────────────────────────────────────────────────────────────────────

def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-shingles. Works regardless of language/script."""
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicate_indices(
    texts: list[str],
    threshold: float = 0.85,
    k: int = 5,
) -> list[int]:
    """Return indices that are near-duplicates of an EARLIER row.
    Greedy: keeps the first occurrence, marks later ones as dupes.
    O(N²) — only use on sets up to ~50k rows. For larger, use datasketch.
    """
    if len(texts) > 50_000:
        logger.warning(
            f"near_duplicate_indices on {len(texts)} rows is O(N²); "
            "consider MinHashLSH from datasketch for larger sets."
        )
    shingle_sets = [shingles(t, k) for t in texts]
    dupes: list[int] = []
    seen: list[set[str]] = []
    for i, s in enumerate(shingle_sets):
        is_dup = False
        for kept in seen:
            if jaccard(s, kept) >= threshold:
                is_dup = True
                break
        if is_dup:
            dupes.append(i)
        else:
            seen.append(s)
    return dupes


# ──────────────────────────────────────────────────────────────────────────────
# Text quality heuristics
# ──────────────────────────────────────────────────────────────────────────────

def repetition_ratio(text: str, n: int = 4) -> float:
    """Ratio of repeated n-grams. >0.4 indicates degenerate output."""
    tokens = (text or "").split()
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def token_count(text: str) -> int:
    """Cheap whitespace token count — language-agnostic."""
    return len((text or "").split())


def clean_whitespace(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
