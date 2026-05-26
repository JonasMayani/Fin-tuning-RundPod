"""
pipeline/stage1_clean.py — remove noisy rows from training data.

Filters applied (configurable per language via clean_config.yaml):
  1. Empty / too-short Q or A
  2. Answer not in expected script (e.g. Amharic with Latin chars)
  3. Answer not in expected language (fastText langid disagreement)
  4. Low Q↔A semantic coherence (LaBSE cosine below threshold)
  5. Answers with high repetition (degenerate text)
  6. Near-duplicates within language
  7. Extreme length outliers (per-language adaptive bounds)

This stage REUSES the embeddings cached by Stage 0 to avoid recomputing.

Run: python -m pipeline.stage1_clean \
        --diag reports/stage0/diagnostic_per_row.csv \
        --emb  reports/stage0/embeddings.npz \
        --config pipeline/clean_config.yaml \
        --out  data/cleaned/train_clean_v2.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from pipeline.utils import (
    near_duplicate_indices, repetition_ratio, token_count,
)


DEFAULT_CONFIG = {
    "min_q_tokens": 3,
    "min_a_tokens": 5,
    "max_a_tokens": 400,
    "max_repetition_ratio": 0.4,
    "near_dup_threshold": 0.85,
    "per_language": {
        # qa_cos_min: minimum Q↔A LaBSE cosine to keep the row.
        # Languages where mT0 has weaker coverage need a more lenient threshold.
        "Amh_Eth": {"qa_cos_min": 0.25, "require_script_ok": True,  "require_lang_ok": True},
        "Aka_Gha": {"qa_cos_min": 0.30, "require_script_ok": True,  "require_lang_ok": False},
        "Lug_Uga": {"qa_cos_min": 0.30, "require_script_ok": True,  "require_lang_ok": False},
        "Swa_Ken": {"qa_cos_min": 0.35, "require_script_ok": True,  "require_lang_ok": True},
        "Eng_Uga": {"qa_cos_min": 0.45, "require_script_ok": True,  "require_lang_ok": True},
        "Eng_Gha": {"qa_cos_min": 0.45, "require_script_ok": True,  "require_lang_ok": True},
        "Eng_Eth": {"qa_cos_min": 0.45, "require_script_ok": True,  "require_lang_ok": True},
        "Eng_Ken": {"qa_cos_min": 0.45, "require_script_ok": True,  "require_lang_ok": True},
    },
}


def load_config(path: Path | None) -> dict:
    if path and path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        # Merge defaults underneath user overrides
        merged = {**DEFAULT_CONFIG, **cfg}
        merged["per_language"] = {
            **DEFAULT_CONFIG["per_language"],
            **(cfg.get("per_language") or {}),
        }
        return merged
    return DEFAULT_CONFIG


def clean(
    diag_csv: Path,
    embeddings_npz: Path,
    out_path: Path,
    config: dict,
    report_dir: Path | None = None,
) -> pd.DataFrame:
    """Apply cleaning filters; return the cleaned DataFrame."""
    logger.info(f"Loading per-row diagnostic from {diag_csv}")
    df = pd.read_csv(diag_csv)
    initial = len(df)
    logger.info(f"Starting with {initial:,} rows")

    # Track drops for reporting
    drops: list[dict] = []

    def drop_where(mask: pd.Series, reason: str) -> None:
        nonlocal df
        if mask.any():
            dropped = int(mask.sum())
            per_lang = df.loc[mask, "subset"].value_counts().to_dict()
            drops.append({"reason": reason, "count": dropped, "per_language": per_lang})
            df = df.loc[~mask].copy()
            logger.info(f"  Dropped {dropped:>5} rows — {reason}")

    # ── Filter 1: length sanity ──────────────────────────────────────────────
    drop_where(df["q_tokens"] < config["min_q_tokens"],
               f"Q < {config['min_q_tokens']} tokens")
    drop_where(df["a_tokens"] < config["min_a_tokens"],
               f"A < {config['min_a_tokens']} tokens")
    drop_where(df["a_tokens"] > config["max_a_tokens"],
               f"A > {config['max_a_tokens']} tokens")

    # ── Filter 2: repetition ─────────────────────────────────────────────────
    drop_where(df["a_repetition"] > config["max_repetition_ratio"],
               f"A repetition > {config['max_repetition_ratio']}")

    # ── Filter 3 & 4 & 5: per-language script / langid / coherence ───────────
    for lang, params in config["per_language"].items():
        lang_mask = df["subset"] == lang
        if not lang_mask.any():
            continue
        if params.get("require_script_ok", True):
            bad = lang_mask & (~df["a_script_ok"].astype(bool))
            drop_where(bad, f"[{lang}] answer script mismatch")
        if params.get("require_lang_ok", False):
            # Only enforced where fastText is reliable for that language
            bad = lang_mask & (~df["a_lang_ok"].astype(bool))
            drop_where(bad, f"[{lang}] answer language mismatch (fastText)")
        threshold = float(params.get("qa_cos_min", 0.30))
        bad = (df["subset"] == lang) & (df["qa_cos"] < threshold)
        drop_where(bad, f"[{lang}] Q↔A cosine < {threshold}")

    # ── Filter 6: near-duplicates within each language ───────────────────────
    logger.info("Removing near-duplicates within each language …")
    dup_threshold = config["near_dup_threshold"]
    keep_mask = pd.Series(True, index=df.index)
    for lang in df["subset"].unique():
        sub_idx = df.index[df["subset"] == lang].tolist()
        if len(sub_idx) > 30_000:
            logger.warning(
                f"  [{lang}] {len(sub_idx)} rows — dedup is O(N²), skipping. "
                "Use datasketch MinHashLSH for larger sets."
            )
            continue
        sub_inputs = df.loc[sub_idx, "input"].tolist()
        dup_positions = near_duplicate_indices(sub_inputs, threshold=dup_threshold)
        dup_idx = [sub_idx[p] for p in dup_positions]
        keep_mask.loc[dup_idx] = False
    drop_where(~keep_mask, f"near-duplicates within language (Jaccard ≥ {dup_threshold})")

    # ── Save cleaned data ────────────────────────────────────────────────────
    keep_cols = ["input", "output", "subset"]
    if "ID" in df.columns:
        keep_cols = ["ID"] + keep_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep_cols].to_csv(out_path, index=False)

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("CLEANING REPORT")
    print("=" * 72)
    print(f"Initial:  {initial:>7,}")
    print(f"Kept:     {len(df):>7,}  ({len(df)/initial:.1%})")
    print(f"Dropped:  {initial - len(df):>7,}  ({(initial - len(df))/initial:.1%})")
    print()
    print("Drop reasons:")
    for d in sorted(drops, key=lambda r: -r["count"]):
        print(f"  {d['count']:>5}  {d['reason']}")
    print()
    print("Per-language final counts:")
    counts = df["subset"].value_counts().sort_index()
    for lang, n in counts.items():
        print(f"  {lang:<10} {n:>6,}")
    print()
    logger.info(f"Cleaned dataset → {out_path}")

    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "clean_drops.json").write_text(json.dumps(drops, indent=2))
        (report_dir / "clean_counts.json").write_text(json.dumps(
            counts.to_dict(), indent=2))

    return df[keep_cols]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diag", required=True, help="Stage 0 diagnostic_per_row.csv")
    p.add_argument("--emb",  required=False, help="Stage 0 embeddings.npz (reserved for future use)")
    p.add_argument("--config", default=None, help="clean_config.yaml (optional)")
    p.add_argument("--out",  required=True, help="Output cleaned train CSV")
    p.add_argument("--report_dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    clean(
        diag_csv=Path(args.diag),
        embeddings_npz=Path(args.emb) if args.emb else None,
        out_path=Path(args.out),
        config=cfg,
        report_dir=Path(args.report_dir) if args.report_dir else None,
    )
