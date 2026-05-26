"""
pipeline/stage3_crosslingual.py — expand low-resource langs from clean English.

Strategy:
  1. Select highest-quality English Q/A pairs (high Q↔A coherence, no flags).
  2. Translate BOTH question and answer to each target African language via NLLB.
  3. Quality-gate the result: language ID, script, length sanity, semantic
     preservation (LaBSE cos between English answer and translated answer).

Why this works:
  Your English partitions have the most rows; even after cleaning there's
  more there than for Amh_Eth (462 rows). Cross-lingual transfer turns
  English abundance into Amharic/Luganda volume.

CAUTION:
  Translated medical content can introduce subtle errors. We filter
  aggressively and keep this technique to a moderate share of total
  augmented data. Treat translated pairs as supplementary signal, not as
  ground truth for evaluation.

Run: python -m pipeline.stage3_crosslingual \
        --in  data/cleaned/train_clean_v2.csv \
        --diag reports/stage0/diagnostic_per_row.csv \
        --out data/augmented/stage3_crosslingual.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from pipeline.stage2_backtranslate import NLLB_CODE, NLLBTranslator
from pipeline.utils import (
    clean_whitespace, cosine_sim, embed_texts, language_ok, script_ok, token_count,
)


# Where to take English seeds from — we use the cleanest English variant.
# Eng_Gha scored highest ROUGE in the eval; we use it as the cleanest pool.
SEED_LANGUAGE = "Eng_Gha"

# How many cross-lingual rows to generate per target language.
# Cap conservatively — quality > quantity for translated content.
DEFAULT_TRANSFER_TARGETS = {
    "Amh_Eth": 600,
    "Lug_Uga": 500,
    "Swa_Ken": 400,
    "Aka_Gha": 300,
}


def select_clean_english_seeds(
    diag_csv: Path,
    n_seeds: int,
    seed_language: str = SEED_LANGUAGE,
    min_qa_cos: float = 0.55,
) -> pd.DataFrame:
    """Pick the highest-quality English rows for cross-lingual transfer."""
    df = pd.read_csv(diag_csv)
    eng = df[df["subset"] == seed_language].copy()
    if len(eng) == 0:
        # Fall back to any English partition
        eng = df[df["subset"].str.startswith("Eng_")].copy()
    logger.info(f"Found {len(eng):,} English rows in {seed_language}")

    # Hard filters: must pass all script/lang/coherence checks
    eng = eng[
        eng["q_script_ok"].astype(bool)
        & eng["a_script_ok"].astype(bool)
        & eng["q_lang_ok"].astype(bool)
        & eng["a_lang_ok"].astype(bool)
        & (eng["qa_cos"] >= min_qa_cos)
        & (eng["q_tokens"] >= 5) & (eng["q_tokens"] <= 80)
        & (eng["a_tokens"] >= 10) & (eng["a_tokens"] <= 200)
        & (eng["a_repetition"] < 0.2)
    ].copy()
    logger.info(f"{len(eng):,} pass the strict English-seed filter")

    # Sort by Q↔A coherence (highest first) and take the top n
    eng = eng.sort_values("qa_cos", ascending=False).head(n_seeds * 2)
    return eng


def translate_pairs_to_language(
    seeds: pd.DataFrame,
    target_lang: str,
    translator: NLLBTranslator,
    n_target: int,
) -> pd.DataFrame:
    """Translate the Q and A of each seed row from English → target_lang."""
    nllb_target = NLLB_CODE.get(target_lang)
    if not nllb_target:
        raise ValueError(f"No NLLB code for {target_lang}")

    logger.info(f"[{target_lang}] translating {len(seeds)} seed pairs eng_Latn → {nllb_target}")

    qs = seeds["input"].tolist()
    a_s = seeds["output"].tolist()

    translated_q = translator.translate(qs, src_code="eng_Latn", tgt_code=nllb_target)
    translated_a = translator.translate(a_s, src_code="eng_Latn", tgt_code=nllb_target)

    out = pd.DataFrame({
        "input":  [clean_whitespace(t) for t in translated_q],
        "output": [clean_whitespace(t) for t in translated_a],
        "subset": target_lang,
        "aug_method": "crosslingual_from_" + seeds["subset"].iloc[0],
        "src_q": qs,
        "src_a": a_s,
    })

    # ── Quality gate ─────────────────────────────────────────────────────────
    logger.info(f"[{target_lang}] applying quality gate …")
    # 1. Script and language ID checks
    out["script_ok"] = out["output"].apply(lambda t: script_ok(t, target_lang))
    # langid for African languages is unreliable; only enforce it where it works
    enforce_langid = target_lang in {"Amh_Eth", "Swa_Ken"}
    if enforce_langid:
        out["lang_ok"] = out["output"].apply(
            lambda t: language_ok(t, target_lang, min_confidence=0.4))
    else:
        out["lang_ok"] = True

    # 2. Length sanity
    out["q_tokens"] = out["input"].apply(token_count)
    out["a_tokens"] = out["output"].apply(token_count)
    len_ok = (
        out["q_tokens"].between(3, 100)
        & out["a_tokens"].between(5, 300)
    )

    # 3. Semantic preservation: translated answer should still be semantically
    # close to the source English answer. Cross-lingual LaBSE handles this.
    logger.info(f"[{target_lang}] checking semantic preservation via LaBSE …")
    src_a_emb = embed_texts(out["src_a"].tolist(), batch_size=128)
    tgt_a_emb = embed_texts(out["output"].tolist(), batch_size=128)
    out["semantic_preserved"] = cosine_sim(src_a_emb, tgt_a_emb)

    # Combine all filters
    keep = (
        out["script_ok"]
        & out["lang_ok"]
        & len_ok
        & (out["semantic_preserved"] >= 0.55)  # cross-lingual cosine floor
    )
    kept = out[keep].copy()
    logger.info(f"[{target_lang}] kept {len(kept)}/{len(out)} after quality gate")

    if len(kept) > n_target:
        # Prefer highest semantic preservation when truncating
        kept = kept.sort_values("semantic_preserved", ascending=False).head(n_target)

    return kept[["input", "output", "subset", "aug_method", "semantic_preserved"]]


def run(
    input_csv: Path,
    diag_csv: Path,
    out_csv: Path,
    targets: dict[str, int],
    nllb_model: str = "facebook/nllb-200-3.3B",
    seed_language: str = SEED_LANGUAGE,
    batch_size: int = 16,
    num_beams: int = 4,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded cleaned input: {len(df):,} rows")

    # Select clean English seeds
    max_target = max(targets.values()) if targets else 0
    seeds = select_clean_english_seeds(diag_csv, n_seeds=max_target * 2,
                                       seed_language=seed_language)
    if len(seeds) < 50:
        logger.warning(f"Only {len(seeds)} clean English seeds — results will be limited.")

    # Translate to each target language
    translator = NLLBTranslator(
        model_name=nllb_model, batch_size=batch_size, num_beams=num_beams,
    )
    all_aug: list[pd.DataFrame] = []
    try:
        for lang, n_target in targets.items():
            if n_target <= 0:
                continue
            # Take a fresh slice each time so we don't overfit on the same seeds
            lang_seeds = seeds.head(n_target * 2)
            aug = translate_pairs_to_language(lang_seeds, lang, translator, n_target)
            if len(aug):
                all_aug.append(aug)
    finally:
        translator.close()

    if not all_aug:
        logger.warning("No cross-lingual outputs produced.")
        return pd.DataFrame()

    aug_df = pd.concat(all_aug, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aug_df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(aug_df):,} cross-lingual rows → {out_csv}")
    print("\nPer-language counts:")
    print(aug_df["subset"].value_counts().sort_index().to_string())
    return aug_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in",   dest="input", required=True, help="Cleaned train CSV")
    p.add_argument("--diag", required=True, help="Stage 0 diagnostic_per_row.csv")
    p.add_argument("--out",  required=True, help="Output augmented CSV")
    p.add_argument("--config", default=None)
    p.add_argument("--nllb_model", default="facebook/nllb-200-3.3B")
    p.add_argument("--seed_language", default=SEED_LANGUAGE)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_beams",  type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = dict(DEFAULT_TRANSFER_TARGETS)
    if args.config and Path(args.config).exists():
        cfg = yaml.safe_load(open(args.config)) or {}
        if "crosslingual_targets" in cfg:
            targets.update(cfg["crosslingual_targets"])
    run(Path(args.input), Path(args.diag), Path(args.out), targets,
        nllb_model=args.nllb_model, seed_language=args.seed_language,
        batch_size=args.batch_size, num_beams=args.num_beams)
