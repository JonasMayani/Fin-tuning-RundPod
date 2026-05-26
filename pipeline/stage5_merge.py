"""
pipeline/stage5_merge.py — merge original + all augmentations into final train set.

Final steps:
  1. Combine cleaned originals + back-translation + cross-lingual + LLM paraphrases
  2. Mark provenance of each row (source column)
  3. Near-dedup the augmentations against the originals (catch translation
     loops that produced something already in the source)
  4. Balance per-language to a target distribution (avoid over-augmenting
     one language)
  5. Write final_train.csv ready for the trainer

Run: python -m pipeline.stage5_merge \
        --clean    data/cleaned/train_clean_v2.csv \
        --aug      data/augmented/stage2_backtrans.csv \
                   data/augmented/stage3_crosslingual.csv \
                   data/augmented/stage4_paraphrase.csv \
        --out      data/augmented/final_train.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from pipeline.utils import shingles, jaccard


CORE_COLUMNS = ["input", "output", "subset"]


def dedup_augs_against_originals(
    df_orig: pd.DataFrame,
    df_aug: pd.DataFrame,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """Drop augmented rows whose question is near-duplicate of any original
    question in the same language."""
    if len(df_aug) == 0:
        return df_aug
    keep = []
    for lang in df_aug["subset"].unique():
        orig_sub = df_orig[df_orig["subset"] == lang]["input"].tolist()
        aug_sub = df_aug[df_aug["subset"] == lang]
        orig_shingles = [shingles(t) for t in orig_sub]
        kept_for_lang = []
        for _, row in aug_sub.iterrows():
            row_sh = shingles(row["input"])
            is_dup = any(jaccard(row_sh, os) >= threshold for os in orig_shingles)
            if not is_dup:
                kept_for_lang.append(row)
        if kept_for_lang:
            keep.append(pd.DataFrame(kept_for_lang))
            logger.info(f"  [{lang}] kept {len(kept_for_lang)}/{len(aug_sub)} aug rows (deduped vs originals)")
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=df_aug.columns)


def balance_per_language(
    df: pd.DataFrame,
    caps: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Optionally cap each language to a maximum count. None = no cap."""
    if not caps:
        return df
    pieces = []
    for lang in df["subset"].unique():
        sub = df[df["subset"] == lang]
        cap = caps.get(lang)
        if cap is not None and len(sub) > cap:
            # Prefer original over augmented when capping
            sub = sub.sort_values("source", key=lambda s: s != "original").head(cap)
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def run(
    clean_csv: Path,
    aug_csvs: list[Path],
    out_csv: Path,
    caps: dict[str, int] | None = None,
    dedup_threshold: float = 0.85,
) -> pd.DataFrame:
    logger.info(f"Loading cleaned originals: {clean_csv}")
    df_orig = pd.read_csv(clean_csv)
    df_orig = df_orig[CORE_COLUMNS].copy()
    df_orig["source"] = "original"
    logger.info(f"  {len(df_orig):,} original rows")

    aug_pieces = []
    for path in aug_csvs:
        if not path.exists():
            logger.warning(f"Aug file missing, skipping: {path}")
            continue
        df_aug = pd.read_csv(path)
        if "aug_method" in df_aug.columns:
            df_aug["source"] = df_aug["aug_method"]
        else:
            df_aug["source"] = path.stem
        df_aug = df_aug[CORE_COLUMNS + ["source"]].copy()
        logger.info(f"  {len(df_aug):,} rows from {path.name}")
        aug_pieces.append(df_aug)

    if not aug_pieces:
        logger.warning("No augmentation files — writing originals only.")
        df_final = df_orig
    else:
        df_aug_all = pd.concat(aug_pieces, ignore_index=True)
        logger.info(f"Total augmentation rows before dedup: {len(df_aug_all):,}")

        logger.info("Deduplicating augmentations against originals …")
        df_aug_clean = dedup_augs_against_originals(
            df_orig, df_aug_all, threshold=dedup_threshold)
        logger.info(f"After dedup: {len(df_aug_clean):,} aug rows kept")

        df_final = pd.concat([df_orig, df_aug_clean], ignore_index=True)

    df_final = df_final.dropna(subset=CORE_COLUMNS).copy()
    df_final = balance_per_language(df_final, caps)

    # ── Final report ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("FINAL MERGED TRAINING SET")
    print("=" * 72)
    print(f"Total rows: {len(df_final):,}")
    print()
    print("Per language × source:")
    pivot = df_final.pivot_table(
        index="subset", columns="source", aggfunc="size", fill_value=0)
    print(pivot.to_string())
    print()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Keep only the columns the trainer needs (drop `source` from the output
    # for compatibility, but write a separate provenance file).
    df_final[CORE_COLUMNS].to_csv(out_csv, index=False)
    df_final.to_csv(out_csv.parent / (out_csv.stem + "_with_provenance.csv"), index=False)

    logger.info(f"Final train set      → {out_csv}")
    logger.info(f"With provenance      → {out_csv.parent / (out_csv.stem + '_with_provenance.csv')}")

    return df_final


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True, help="Cleaned originals from stage 1")
    p.add_argument("--aug", nargs="*", default=[],
                   help="Augmentation CSVs from stages 2/3/4")
    p.add_argument("--out", required=True, help="Final merged train CSV")
    p.add_argument("--dedup_threshold", type=float, default=0.85)
    p.add_argument("--cap_per_lang", default=None,
                   help="JSON dict, e.g. '{\"Eng_Uga\": 6000}'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    caps = json.loads(args.cap_per_lang) if args.cap_per_lang else None
    run(Path(args.clean),
        [Path(p) for p in args.aug],
        Path(args.out),
        caps=caps,
        dedup_threshold=args.dedup_threshold)
