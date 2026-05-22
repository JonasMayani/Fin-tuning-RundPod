"""
src/data/quality_gate.py
─────────────────────────────────────────────────────────────────────────────
Phase 2.3 — Data Quality Gate
Multilingual African Health Assistant | Zindi ITU Challenge

Filters applied (in order, each is a pass/fail per row):
  1. Length ratio:     answer_len / question_len ∈ [0.5, 20]
  2. Language ID:      detected language must match `subset` tag
  3. Toxicity filter:  multilingual-toxic-xlm-roberta score < threshold
  4. Semantic relevance: multilingual-e5-base cosine similarity ≥ 0.3

Produces:
  - final_train.csv              (all rows passing all gates)
  - quality_gate_report.csv      (counts per filter stage, per language)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import torch
import yaml
from loguru import logger
from tqdm.auto import tqdm

# ─── Lazy Model Cache ─────────────────────────────────────────────────────────

_models: dict[str, object] = {}


def _get_toxicity_classifier():
    """Load multilingual toxicity classifier (cached)."""
    key = "toxicity"
    if key not in _models:
        from transformers import pipeline
        logger.info("Loading toxicity classifier …")
        _models[key] = pipeline(
            "text-classification",
            model="unitary/multilingual-toxic-xlm-roberta",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )
    return _models[key]


def _get_embedding_model():
    """Load multilingual-e5-base for semantic similarity (cached)."""
    key = "embedder"
    if key not in _models:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading multilingual-e5-base embedder …")
        _models[key] = SentenceTransformer("intfloat/multilingual-e5-base")
    return _models[key]


# ─── Language Detection ───────────────────────────────────────────────────────

# Mapping from subset tag → expected lingua/langdetect language codes
_LANG_CODE_MAP = {
    "Aka_Gha": {"tw", "ak"},           # Twi / Akan (limited support)
    "Amh_Eth": {"am"},                  # Amharic
    "Lug_Uga": {"lg"},                  # Luganda (limited support)
    "Swa_Eas": {"sw"},                  # Swahili
    "Eng":     {"en"},                  # English
}

# Fallback: use script heuristic for under-supported languages
_ETHIOSCRIPT_RE  = __import__("re").compile(r"[\u1200-\u137F]")   # Ethiopic
_LATIN_RE        = __import__("re").compile(r"[a-zA-Z]")


def _script_lang_id(text: str, subset: str) -> bool:
    """
    Heuristic language ID based on Unicode script.
    Returns True if the text is consistent with the expected language.
    """
    if subset == "Amh_Eth":
        return bool(_ETHIOSCRIPT_RE.search(text))
    if subset in ("Aka_Gha", "Lug_Uga", "Swa_Eas", "Eng"):
        return bool(_LATIN_RE.search(text))
    return True


def detect_language(text: str) -> Optional[str]:
    """Try lingua-language-detector, fall back to langdetect."""
    try:
        from lingua import LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_all_languages().build()
        result = detector.detect_language_of(text)
        return result.iso_code_639_1.name.lower() if result else None
    except Exception:
        pass
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return None


def language_id_check(row: pd.Series) -> bool:
    """Return True if text language is consistent with subset tag."""
    subset = row["subset"]
    text   = str(row["input"])

    # Always trust script heuristic first (fast, covers Amharic reliably)
    if not _script_lang_id(text, subset):
        return False

    # For Latin-script languages try LLM-free detection
    detected = detect_language(text[:200])  # use first 200 chars for speed
    if detected is None:
        return True  # can't tell → pass (conservative)

    expected = _LANG_CODE_MAP.get(subset, set())
    # Broad pass: accept if detected is in expected OR if expected is empty
    return not expected or detected in expected


# ─── Gate Functions ───────────────────────────────────────────────────────────

def gate_length_ratio(df: pd.DataFrame, min_r: float = 0.5, max_r: float = 20.0) -> pd.Series:
    """Pass if answer_len / question_len ∈ [min_r, max_r]."""
    q_len = df["input"].str.split().apply(len).clip(lower=1)
    a_len = df["output"].str.split().apply(len).clip(lower=1)
    ratio = a_len / q_len
    return (ratio >= min_r) & (ratio <= max_r)


def gate_language_id(df: pd.DataFrame) -> pd.Series:
    """Pass if text script/language matches the subset tag."""
    logger.info("Running language ID gate …")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LangID"):
        results.append(language_id_check(row))
    return pd.Series(results, index=df.index)


def gate_toxicity(
    df: pd.DataFrame,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> pd.Series:
    """Pass if neither input nor output is toxic (score < threshold)."""
    logger.info("Running toxicity gate …")
    classifier = _get_toxicity_classifier()

    def is_toxic(texts: list[str]) -> list[bool]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            preds = classifier(batch)
            for pred in preds:
                score = pred["score"] if pred["label"] == "toxic" else 1.0 - pred["score"]
                results.append(score >= threshold)
        return results

    input_toxic  = is_toxic(df["input"].tolist())
    output_toxic = is_toxic(df["output"].tolist())
    return ~pd.Series(input_toxic, index=df.index) & ~pd.Series(output_toxic, index=df.index)


def gate_semantic_relevance(
    df: pd.DataFrame,
    threshold: float = 0.3,
    batch_size: int = 128,
) -> pd.Series:
    """Pass if cosine similarity between input and output embeddings ≥ threshold."""
    logger.info("Running semantic relevance gate …")
    model = _get_embedding_model()

    # multilingual-e5 convention: prepend "query: " for questions, "passage: " for answers
    inputs  = ["query: "   + str(t) for t in df["input"].tolist()]
    outputs = ["passage: " + str(t) for t in df["output"].tolist()]

    emb_q = model.encode(inputs,  batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
    emb_a = model.encode(outputs, batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)

    cos_sims = (emb_q * emb_a).sum(axis=1)
    return pd.Series(cos_sims >= threshold, index=df.index)


# ─── Main Quality Gate Runner ─────────────────────────────────────────────────

def run_quality_gate(config_path: str = "src/training/config.yaml") -> None:
    """Apply all quality gates to the augmented training set."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths   = cfg["paths"]
    aug_cfg = cfg["augmentation"]

    augmented_dir = Path(paths["data_augmented"])
    reports_dir   = Path(paths["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_path  = augmented_dir / "augmented_train.csv"
    output_path = augmented_dir / "final_train.csv"
    report_path = reports_dir   / "quality_gate_report.csv"

    if not input_path.exists():
        logger.error(f"Augmented training data not found at {input_path}. Run augment.py first.")
        return

    df = pd.read_csv(input_path)
    logger.info(f"Quality gate input: {len(df)} rows.")

    report_rows: list[dict] = []

    def record(gate_name: str, mask: pd.Series) -> None:
        total    = len(mask)
        passing  = mask.sum()
        failing  = total - passing
        for lang, grp in df.groupby("subset"):
            lang_mask    = mask[grp.index]
            lang_passing = lang_mask.sum()
            lang_failing = len(lang_mask) - lang_passing
            report_rows.append({
                "gate":           gate_name,
                "language":       lang,
                "total":          len(grp),
                "passing":        int(lang_passing),
                "failing":        int(lang_failing),
                "pass_rate":      round(lang_passing / max(len(grp), 1), 4),
            })
        logger.info(f"Gate [{gate_name}]: {passing}/{total} passed "
                    f"({100*passing/max(total,1):.1f}%)")

    # Gate 1: Length ratio
    g1 = gate_length_ratio(
        df,
        min_r=aug_cfg["length_ratio_min"],
        max_r=aug_cfg["length_ratio_max"],
    )
    record("length_ratio", g1)
    df = df[g1].copy()

    # Gate 2: Language ID
    g2 = gate_language_id(df)
    record("language_id", g2)
    df = df[g2].copy()

    # Gate 3: Toxicity
    g3 = gate_toxicity(df)
    record("toxicity", g3)
    df = df[g3].copy()

    # Gate 4: Semantic relevance
    g4 = gate_semantic_relevance(df, threshold=aug_cfg["semantic_cos_threshold"])
    record("semantic_relevance", g4)
    df = df[g4].copy()

    # Save outputs
    df.to_csv(output_path, index=False)
    logger.success(f"Final training set saved → {output_path}  ({len(df)} rows)")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)
    logger.success(f"Quality gate report saved → {report_path}")

    logger.info("\n── Final language distribution ──")
    for lang, cnt in df["subset"].value_counts().items():
        logger.info(f"  {lang:<12} {cnt:>6} rows")


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/training/config.yaml"
    run_quality_gate(cfg_path)
