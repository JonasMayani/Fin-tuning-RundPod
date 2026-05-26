"""
pipeline/stage0_diagnose.py — what is actually wrong with this dataset?

Produces a report per language with:
  • Row count
  • Script mismatch rate (#1 noise source for Amharic)
  • Language ID disagreement rate
  • Q↔A semantic similarity (LaBSE) — distribution
  • Q and A length distributions
  • Repetition ratio in answers
  • Near-duplicate rate within language
  • Sample of bottom-quartile rows (saved as CSV for hand inspection)

Run: python -m pipeline.stage0_diagnose --train data/cleaned/train_clean.csv --out reports/stage0/

This stage NEVER deletes anything — it only inspects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pipeline.utils import (
    EXPECTED_SCRIPT, FASTTEXT_ACCEPT,
    clean_whitespace, detect_script, detect_language,
    embed_texts, cosine_sim, repetition_ratio, token_count,
    near_duplicate_indices,
)


def diagnose(train_csv: Path, out_dir: Path, dup_check_rows: int = 5000) -> dict:
    """Run all diagnostics and write reports to out_dir. Returns summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reading {train_csv}")
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=["input", "output", "subset"]).copy()
    for col in ["input", "output", "subset"]:
        df[col] = df[col].astype(str).map(clean_whitespace)
    df = df[(df["input"] != "") & (df["output"] != "")].reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows")

    # ── Script + langid checks ────────────────────────────────────────────────
    logger.info("Detecting script for each Q and A …")
    df["q_script"] = df["input"].map(detect_script)
    df["a_script"] = df["output"].map(detect_script)
    df["expected_script"] = df["subset"].map(EXPECTED_SCRIPT).fillna("UNKNOWN")
    df["q_script_ok"] = df["q_script"] == df["expected_script"]
    df["a_script_ok"] = df["a_script"] == df["expected_script"]

    logger.info("Running fastText language ID on Q and A (may take a few minutes) …")
    df["q_lang"], df["q_lang_conf"] = zip(*[detect_language(t) for t in tqdm(df["input"], desc="Q langid")])
    df["a_lang"], df["a_lang_conf"] = zip(*[detect_language(t) for t in tqdm(df["output"], desc="A langid")])
    df["q_lang_ok"] = [
        (lang in FASTTEXT_ACCEPT.get(subset, set())) and conf >= 0.5
        for lang, conf, subset in zip(df["q_lang"], df["q_lang_conf"], df["subset"])
    ]
    df["a_lang_ok"] = [
        (lang in FASTTEXT_ACCEPT.get(subset, set())) and conf >= 0.5
        for lang, conf, subset in zip(df["a_lang"], df["a_lang_conf"], df["subset"])
    ]

    # ── Length and repetition ────────────────────────────────────────────────
    df["q_tokens"] = df["input"].map(token_count)
    df["a_tokens"] = df["output"].map(token_count)
    df["a_repetition"] = df["output"].map(repetition_ratio)

    # ── Q↔A semantic similarity (LaBSE) ──────────────────────────────────────
    logger.info("Computing LaBSE embeddings for Q and A (this is the slow part) …")
    q_embs = embed_texts(df["input"].tolist(), batch_size=128)
    a_embs = embed_texts(df["output"].tolist(), batch_size=128)
    df["qa_cos"] = cosine_sim(q_embs, a_embs)

    # Save the embeddings; downstream stages will reuse them
    np.savez_compressed(out_dir / "embeddings.npz",
                        q=q_embs, a=a_embs, indices=df.index.to_numpy())

    # ── Near-dup rate per language (sampled, since O(N²)) ────────────────────
    logger.info("Estimating near-duplicate rate per language (sampled) …")
    nd_rates: dict[str, float] = {}
    for lang in df["subset"].unique():
        sub = df[df["subset"] == lang]
        sample = sub["input"].sample(
            n=min(dup_check_rows, len(sub)), random_state=42
        ).tolist()
        dupes = near_duplicate_indices(sample, threshold=0.85)
        nd_rates[lang] = len(dupes) / max(len(sample), 1)

    # ── Per-language summary ─────────────────────────────────────────────────
    summary_rows = []
    for lang in sorted(df["subset"].unique()):
        sub = df[df["subset"] == lang]
        summary_rows.append({
            "language": lang,
            "n": len(sub),
            "q_script_ok_rate": float(sub["q_script_ok"].mean()),
            "a_script_ok_rate": float(sub["a_script_ok"].mean()),
            "q_lang_ok_rate":   float(sub["q_lang_ok"].mean()),
            "a_lang_ok_rate":   float(sub["a_lang_ok"].mean()),
            "qa_cos_mean":      float(sub["qa_cos"].mean()),
            "qa_cos_median":    float(sub["qa_cos"].median()),
            "qa_cos_p10":       float(sub["qa_cos"].quantile(0.10)),
            "q_tokens_median":  float(sub["q_tokens"].median()),
            "a_tokens_median":  float(sub["a_tokens"].median()),
            "a_repetition_mean": float(sub["a_repetition"].mean()),
            "near_dup_rate_sampled": nd_rates.get(lang, 0.0),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "diagnostic_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Full per-row diagnostic with all flags (for downstream stages) ───────
    full_path = out_dir / "diagnostic_per_row.csv"
    df.to_csv(full_path, index=False)

    # ── Bottom-quartile sample for hand inspection ───────────────────────────
    suspicious_rows = []
    for lang in df["subset"].unique():
        sub = df[df["subset"] == lang].sort_values("qa_cos")
        suspicious_rows.append(sub.head(20))  # 20 worst per language
    suspicious_df = pd.concat(suspicious_rows, ignore_index=True)[
        ["subset", "qa_cos", "q_script", "a_script", "q_lang", "a_lang",
         "q_tokens", "a_tokens", "a_repetition", "input", "output"]
    ]
    suspicious_df.to_csv(out_dir / "suspicious_rows_for_review.csv", index=False)

    # ── Pretty print summary ─────────────────────────────────────────────────
    print()
    print("=" * 95)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 95)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print("Flag legend:")
    print("  q_script_ok / a_script_ok : fraction of rows where text is in the expected Unicode script")
    print("  q_lang_ok / a_lang_ok     : fraction where fastText agrees with the language label (conf ≥ 0.5)")
    print("  qa_cos_*                  : LaBSE cosine similarity between Q and A (higher = more related)")
    print("  near_dup_rate_sampled     : fraction of input questions that are ~duplicates of an earlier one")
    print()

    # ── Diagnose probable root cause per language ────────────────────────────
    print("PROBABLE ISSUES PER LANGUAGE:")
    for _, row in summary_df.iterrows():
        issues = []
        if row["a_script_ok_rate"] < 0.8:
            issues.append(f"⚠ answer-script mismatch ({row['a_script_ok_rate']:.0%} OK)")
        if row["a_lang_ok_rate"] < 0.6:
            issues.append(f"⚠ answer-language mismatch ({row['a_lang_ok_rate']:.0%} OK)")
        if row["qa_cos_median"] < 0.4:
            issues.append(f"⚠ low Q↔A coherence (median cos={row['qa_cos_median']:.2f})")
        if row["near_dup_rate_sampled"] > 0.15:
            issues.append(f"⚠ high duplication ({row['near_dup_rate_sampled']:.0%})")
        if row["a_repetition_mean"] > 0.2:
            issues.append(f"⚠ repetitive answers (mean rep={row['a_repetition_mean']:.2f})")
        if not issues:
            issues.append("✓ looks healthy")
        print(f"  {row['language']:<10}  {'; '.join(issues)}")
    print()
    print(f"Full per-row diagnostic   → {full_path}")
    print(f"Per-language summary      → {summary_path}")
    print(f"Top-suspicious sample     → {out_dir / 'suspicious_rows_for_review.csv'}")
    print(f"Cached LaBSE embeddings   → {out_dir / 'embeddings.npz'}")

    return {
        "n_total": len(df),
        "per_language": summary_df.to_dict(orient="records"),
        "near_dup_rates": nd_rates,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Input train CSV")
    p.add_argument("--out", required=True, help="Output report dir")
    p.add_argument("--dup_sample", type=int, default=5000,
                   help="Rows per language to sample for dedup check (O(N²))")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = diagnose(Path(args.train), Path(args.out), args.dup_sample)
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2))
