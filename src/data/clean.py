"""
src/data/clean.py
─────────────────────────────────────────────────────────────────────────────
Phase 1.2 — Data Cleaning Pipeline
Multilingual African Health Assistant | Zindi ITU Challenge

Steps applied (in order):
  1. Remove exact duplicate (input, output) pairs
  2. NFC Unicode normalisation across all scripts
  3. Strip HTML tags, URLs, citation brackets e.g. [[1]]
  4. Standardise whitespace and punctuation
  5. Flag / quarantine rows containing PII
  6. Flag answers that may contradict WHO MSRH guidelines (manual review queue)
  7. Enforce consistent `subset` language tags

All changes are logged to cleaning_log.csv.
"""

from __future__ import annotations

import re
import unicodedata
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import yaml
from loguru import logger

# ─── Constants ────────────────────────────────────────────────────────────────

VALID_SUBSETS = {
    "Eng_Uga", "Aka_Gha", "Eng_Gha", "Eng_Eth",
    "Lug_Uga", "Eng_Ken", "Swa_Ken", "Amh_Eth"
}

# Regex patterns
_HTML_TAG       = re.compile(r"<[^>]+>")
_URL            = re.compile(r"https?://\S+|www\.\S+")
_CITATION       = re.compile(r"\[\[?\d+\]?\]")
_EXTRA_WS       = re.compile(r"\s{2,}")
_PHONE_NUMBER   = re.compile(r"\b[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,}\b")
_EMAIL          = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PERSON_NAME    = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b")

# WHO MSRH safety keywords (simplified; expand with clinical guidance)
_UNSAFE_PATTERNS = [
    r"take\s+\d+\s+(tablets?|pills?|mg)\s+at\s+once",  # dangerous dosage
    r"do\s+not\s+go\s+to\s+(a\s+)?(doctor|hospital|clinic)",  # avoid care
    r"abort\s+using\s+[a-z\s]+at\s+home",  # unsafe abortion advice
    r"self[-\s]medicate\s+with",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)

# Subset tag normalisation map (common mislabellings → canonical)
_TAG_FIX_MAP = {
    # Akan variants
    "twi": "Aka_Gha", "akan": "Aka_Gha", "aka_gha": "Aka_Gha",
    # Amharic variants
    "amharic": "Amh_Eth", "amh": "Amh_Eth", "amh_eth": "Amh_Eth",
    # Luganda variants
    "luganda": "Lug_Uga", "lug": "Lug_Uga", "lug_uga": "Lug_Uga",
    # Swahili variants
    "swahili": "Swa_Ken", "swa": "Swa_Ken", "swa_eas": "Swa_Ken",
    "kiswahili": "Swa_Ken", "swh_ken": "Swa_Ken",
    # English regional variants
    "eng": "Eng_Uga",       # fallback English → Uganda (largest group)
    "english": "Eng_Uga",
    "eng_uganda": "Eng_Uga",
    "eng_ghana": "Eng_Gha",
    "eng_ethiopia": "Eng_Eth",
    "eng_kenya": "Eng_Ken",
}


# ─── Text Helpers ─────────────────────────────────────────────────────────────

def nfc_normalise(text: str) -> str:
    """Apply NFC Unicode normalisation."""
    return unicodedata.normalize("NFC", text)


def strip_noise(text: str) -> str:
    """Remove HTML, URLs, citation brackets; normalise whitespace."""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _CITATION.sub(" ", text)
    text = _EXTRA_WS.sub(" ", text)
    return text.strip()


def normalise_text(text: str) -> str:
    """Full text normalisation pipeline."""
    text = nfc_normalise(text)
    text = strip_noise(text)
    return text


def row_hash(row: pd.Series, cols: list[str]) -> str:
    """Stable hash of selected columns for deduplication."""
    combined = "".join(str(row[c]) for c in cols).encode("utf-8")
    return hashlib.md5(combined).hexdigest()


# ─── PII Detection ────────────────────────────────────────────────────────────

def contains_pii(text: str) -> bool:
    """Heuristic PII check — phone numbers, emails, named person references."""
    if _PHONE_NUMBER.search(text):
        return True
    if _EMAIL.search(text):
        return True
    if _PERSON_NAME.search(text):
        return True
    return False


# ─── Safety Flag ──────────────────────────────────────────────────────────────

def flag_unsafe(text: str) -> bool:
    """Flag potential WHO MSRH guideline violations."""
    return bool(_UNSAFE_RE.search(text))


# ─── Subset Tag Normalisation ─────────────────────────────────────────────────

def normalise_subset(tag: str) -> Optional[str]:
    """Return canonical tag or None if unresolvable."""
    if tag in VALID_SUBSETS:
        return tag
    normalised = _TAG_FIX_MAP.get(tag.strip().lower())
    return normalised  # may be None


# ─── Main Cleaning Function ───────────────────────────────────────────────────

def clean_dataframe(
    df: pd.DataFrame,
    split_name: str,
    log_rows: list[dict],
) -> pd.DataFrame:
    """
    Apply all cleaning steps to a dataframe in-place.
    Appends change records to log_rows.

    Parameters
    ----------
    df         : Raw dataframe with columns ID, input, output, subset
    split_name : Human-readable label used in log (e.g. 'train', 'val')
    log_rows   : Accumulated log list (mutated in-place)

    Returns
    -------
    Cleaned dataframe (rows that must be quarantined are excluded).
    """
    original_len = len(df)
    quarantine_mask = pd.Series(False, index=df.index)

    # ── 1. Exact deduplication ──────────────────────────────────────────────
    df["_hash"] = df.apply(lambda r: row_hash(r, ["input", "output"]), axis=1)
    dupes = df.duplicated(subset="_hash", keep="first")
    for idx in df[dupes].index:
        log_rows.append({
            "split": split_name, "row_id": df.at[idx, "ID"],
            "change_type": "REMOVED_DUPLICATE",
            "original_value": df.at[idx, "input"][:120],
            "new_value": None,
        })
    df = df[~dupes].copy()
    logger.info(f"[{split_name}] Removed {dupes.sum()} exact duplicates.")

    # ── 2 & 3 & 4. Unicode NFC + strip noise + whitespace ───────────────────
    for col in ["input", "output"]:
        original = df[col].copy()
        df[col] = df[col].astype(str).apply(normalise_text)
        changed = df[col] != original
        for idx in df[changed].index:
            log_rows.append({
                "split": split_name, "row_id": df.at[idx, "ID"],
                "change_type": f"TEXT_NORMALISED_{col.upper()}",
                "original_value": str(original[idx])[:120],
                "new_value": str(df.at[idx, col])[:120],
            })
    logger.info(f"[{split_name}] Text normalisation applied.")

    # ── 5. PII flagging → quarantine ────────────────────────────────────────
    pii_mask = df["input"].apply(contains_pii) | df["output"].apply(contains_pii)
    for idx in df[pii_mask].index:
        log_rows.append({
            "split": split_name, "row_id": df.at[idx, "ID"],
            "change_type": "QUARANTINE_PII",
            "original_value": df.at[idx, "input"][:120],
            "new_value": "[QUARANTINED]",
        })
    quarantine_mask |= pii_mask
    logger.info(f"[{split_name}] Flagged {pii_mask.sum()} rows containing PII.")

    # ── 6. WHO safety flag → mark for manual review ─────────────────────────
    safety_mask = df["output"].apply(flag_unsafe)
    df["_safety_flag"] = safety_mask
    for idx in df[safety_mask].index:
        log_rows.append({
            "split": split_name, "row_id": df.at[idx, "ID"],
            "change_type": "FLAGGED_UNSAFE_ANSWER",
            "original_value": df.at[idx, "output"][:200],
            "new_value": "[MANUAL_REVIEW]",
        })
    logger.info(f"[{split_name}] Flagged {safety_mask.sum()} potentially unsafe answers for review.")

    # ── 7. Subset tag normalisation ─────────────────────────────────────────
    df["_subset_original"] = df["subset"]
    df["subset"] = df["subset"].astype(str).apply(normalise_subset)
    invalid_mask = df["subset"].isna()
    tag_fixed_mask = df["subset"] != df["_subset_original"]
    for idx in df[tag_fixed_mask & ~invalid_mask].index:
        log_rows.append({
            "split": split_name, "row_id": df.at[idx, "ID"],
            "change_type": "SUBSET_TAG_FIXED",
            "original_value": df.at[idx, "_subset_original"],
            "new_value": df.at[idx, "subset"],
        })
    for idx in df[invalid_mask].index:
        log_rows.append({
            "split": split_name, "row_id": df.at[idx, "ID"],
            "change_type": "QUARANTINE_INVALID_SUBSET",
            "original_value": df.at[idx, "_subset_original"],
            "new_value": "[QUARANTINED]",
        })
    quarantine_mask |= invalid_mask
    logger.info(f"[{split_name}] Fixed {tag_fixed_mask.sum()} subset tags; "
                f"quarantined {invalid_mask.sum()} unresolvable tags.")

    # ── Finalise ────────────────────────────────────────────────────────────
    df = df[~quarantine_mask].copy()
    df.drop(columns=["_hash", "_subset_original"], inplace=True, errors="ignore")
    logger.info(f"[{split_name}] Cleaning complete: {original_len} → {len(df)} rows "
                f"({original_len - len(df)} removed/quarantined).")
    return df


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def run_cleaning(config_path: str = "src/training/config.yaml") -> None:
    """End-to-end cleaning pipeline driven by config.yaml."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    raw_dir     = Path(paths["data_raw"])
    cleaned_dir = Path(paths["data_cleaned"])
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict] = []

    for split in ("Train", "Val"):
        src = raw_dir / f"{split}.csv"
        if not src.exists():
            logger.warning(f"{src} not found — skipping.")
            continue

        logger.info(f"Loading {src} …")
        df = pd.read_csv(src)
        df_clean = clean_dataframe(df, split.lower(), log_rows)

        out_path = cleaned_dir / f"{split.lower()}_clean.csv"
        df_clean.to_csv(out_path, index=False)
        logger.success(f"Saved cleaned {split} → {out_path}")

    # Save cleaning log
    log_df = pd.DataFrame(log_rows)
    log_path = cleaned_dir / "cleaning_log.csv"
    log_df.to_csv(log_path, index=False)
    logger.success(f"Cleaning log saved → {log_path}  ({len(log_rows)} entries)")


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/training/config.yaml"
    run_cleaning(cfg_path)
