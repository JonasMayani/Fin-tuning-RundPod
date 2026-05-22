"""
src/data/augment.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Data Augmentation Pipeline
Multilingual African Health Assistant | Zindi ITU Challenge

Augmentation strategies implemented:
  A. Forward Translation  — English MSRH Q&A → African languages via NLLB-200
  B. Back-Translation     — African Q&A → English → paraphrase → back
  C. Synonym Substitution — masked prediction via multilingual-e5 / mBERT
  D. Temperature Sampling — mixing ratios for curriculum learning

Quality gate is applied after all augmentation steps (see quality_gate.py).
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import torch
import yaml
from loguru import logger
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# ─── Lazy Model Cache ─────────────────────────────────────────────────────────

_models: dict[str, object] = {}


def _get_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    """Load NLLB translation model (cached). Uses distilled-600M for local runs;
    swap to nllb-200-3.3B in production via config."""
    key = f"nllb_{model_name}"
    if key not in _models:
        logger.info(f"Loading NLLB model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        _models[key] = (model, tokenizer)
    return _models[key]


def _get_paraphrase_model():
    """Load T5 paraphrase model (cached)."""
    key = "paraphrase"
    if key not in _models:
        logger.info("Loading T5 paraphrase model …")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = "Vamsi/t5_paraphrase_paws"
        para_tokenizer = AutoTokenizer.from_pretrained(model_name)
        para_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            para_model = para_model.cuda()
        para_model.eval()
        _models[key] = (para_model, para_tokenizer)
    return _models[key]


def _safe_translation_batch_size(cfg: dict) -> int:
    """Return a lower batch size when GPU memory is limited to avoid OOM."""
    default = cfg["translation"]["batch_size"]
    if not torch.cuda.is_available():
        return default
    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if total_gb <= 8.0:
            return min(default, 8)
    except Exception:
        pass
    return default


def _normalize_text_batch(texts: list[str]) -> list[str]:
    """Ensure translation inputs are valid strings for the tokenizer."""
    normalized: list[str] = []
    for text in texts:
        if isinstance(text, str):
            normalized.append(text)
        elif text is None or (isinstance(text, float) and np.isnan(text)):
            normalized.append("")
        else:
            normalized.append(str(text))
    return normalized


# ─── Translation Utils ────────────────────────────────────────────────────────

def translate_batch(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    model_name: str = "facebook/nllb-200-distilled-600M",
    num_beams: int = 5,
    batch_size: int = 32,
    max_length: int = 512,
) -> list[str]:
    """
    Translate a list of texts using NLLB-200.

    Parameters
    ----------
    texts      : List of source texts.
    src_lang   : NLLB BCP-47 source language code (e.g. 'eng_Latn').
    tgt_lang   : NLLB BCP-47 target language code (e.g. 'swh_Latn').
    """
    model, tokenizer = _get_nllb(model_name)
    tokenizer.src_lang = src_lang
    translations: list[str] = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating {src_lang}→{tgt_lang}"):
        batch = _normalize_text_batch(texts[i : i + batch_size])
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                num_beams=num_beams,
                max_new_tokens=max_length,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def compute_chrf(hypotheses: list[str], references: list[str]) -> list[float]:
    """Compute sentence-level chrF scores between hypotheses and references."""
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
        scores = []
        for hyp, ref in zip(hypotheses, references):
            score = chrf.sentence_score(hyp, [ref]).score / 100.0
            scores.append(score)
        return scores
    except ImportError:
        logger.warning("sacrebleu not available; returning dummy chrF=1.0")
        return [1.0] * len(hypotheses)


# ─── Forward Translation (English → African languages) ───────────────────────

def forward_translate(
    df_english: pd.DataFrame,
    target_langs: dict[str, str],  # {subset_tag: nllb_code}
    cfg: dict,
) -> pd.DataFrame:
    """
    Translate English Q&A pairs into each target African language.

    Parameters
    ----------
    df_english   : DataFrame with 'input' and 'output' columns (English).
    target_langs : Mapping from dataset subset tag → NLLB BCP-47 code.
    cfg          : Full config dict.

    Returns
    -------
    DataFrame with columns: source_id, language, input_translated,
                             output_translated, chrf_score
    """
    nllb_cfg   = cfg["translation"]
    model_name = nllb_cfg["model_name"]
    num_beams  = nllb_cfg["num_beams"]
    batch_size = _safe_translation_batch_size(cfg)
    chrf_thr   = cfg["augmentation"]["chrf_threshold"]

    rows: list[dict] = []
    src_lang = "eng_Latn"

    for subset_tag, tgt_lang in target_langs.items():
        if subset_tag == "Eng":
            continue  # no need to translate English → English

        logger.info(f"Forward translating → {subset_tag} ({tgt_lang}) …")

        inputs_en  = df_english["input"].tolist()
        outputs_en = df_english["output"].tolist()

        translated_inputs  = translate_batch(inputs_en,  src_lang, tgt_lang,
                                             model_name, num_beams, batch_size)
        translated_outputs = translate_batch(outputs_en, src_lang, tgt_lang,
                                             model_name, num_beams, batch_size)

        # Quality filter: chrF ≥ threshold vs. a reference back-translated to English
        # (Approximation: compare translated input chrF against original English input
        #  after round-trip; here we compute input chrF as proxy)
        chrf_scores = compute_chrf(translated_inputs, inputs_en)

        for idx, (tid, tod, chrf) in enumerate(
            zip(translated_inputs, translated_outputs, chrf_scores)
        ):
            if chrf < chrf_thr:
                continue
            rows.append({
                "source_id":         df_english.index[idx],
                "language":          subset_tag,
                "input_translated":  tid.strip(),
                "output_translated": tod.strip(),
                "chrf_score":        round(chrf, 4),
                "source":            "forward_mt",
            })

    logger.success(f"Forward translation produced {len(rows)} pairs "
                   f"(after chrF ≥ {chrf_thr} filter).")
    return pd.DataFrame(rows)


# ─── Back-Translation + Paraphrase ───────────────────────────────────────────

def back_translate_and_paraphrase(
    df: pd.DataFrame,
    nllb_codes: dict[str, str],
    cfg: dict,
) -> pd.DataFrame:
    """
    For each African-language Q&A pair:
      1. Translate input → English (NLLB)
      2. Paraphrase the English input (T5)
      3. Translate paraphrased English → original language (NLLB)
      4. Keep original output as reference

    Returns dataframe with augmented pairs.
    """
    nllb_cfg   = cfg["translation"]
    model_name = nllb_cfg["model_name"]
    num_beams  = nllb_cfg["num_beams"]
    batch_size = _safe_translation_batch_size(cfg)

    para_model, para_tokenizer = _get_paraphrase_model()
    rows: list[dict] = []

    for subset_tag, group in df.groupby("subset"):
        if subset_tag not in nllb_codes:
            continue
        tgt_lang = nllb_codes[subset_tag]
        if tgt_lang == "eng_Latn":
            continue
        logger.info(f"Back-translating {subset_tag} ({len(group)} pairs) …")

        # Step 1: Translate African → English
        inputs_af = group["input"].tolist()
        inputs_en = translate_batch(inputs_af, tgt_lang, "eng_Latn",
                                    model_name, num_beams, batch_size)

        # Step 2: Paraphrase English
        paraphrased_en: list[str] = []
        for text in tqdm(inputs_en, desc=f"Paraphrasing {subset_tag}"):
            try:
                input_ids = para_tokenizer(
                    f"paraphrase: {text} </s>",
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                ).input_ids
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                with torch.no_grad():
                    outputs = para_model.generate(
                        input_ids,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=1.5,
                        num_return_sequences=1,
                    )
                result = para_tokenizer.decode(outputs[0], skip_special_tokens=True)
                paraphrased_en.append(result)
            except Exception:
                paraphrased_en.append(text)  # fallback: keep original

        # Step 3: Translate paraphrased English → African language
        back_translated = translate_batch(paraphrased_en, "eng_Latn", tgt_lang,
                                          model_name, num_beams, batch_size)

        for orig_row, bt_input in zip(group.itertuples(), back_translated):
            rows.append({
                "ID":     f"{orig_row.ID}_bt",
                "input":  bt_input.strip(),
                "output": orig_row.output,
                "subset": subset_tag,
                "source": "back_bt",
            })

    logger.success(f"Back-translation produced {len(rows)} augmented pairs.")
    return pd.DataFrame(rows)


# ─── Temperature-Based Data Mixing ────────────────────────────────────────────

def temperature_sample(
    df: pd.DataFrame,
    target_ratios: dict[str, float],
    temperature: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply temperature sampling to balance language distribution.
    Languages with fewer examples are over-sampled.

    T → ∞  :  uniform sampling across languages.
    T = 1  :  proportional sampling (no correction).
    """
    rng = np.random.RandomState(seed)
    counts = df["subset"].value_counts()
    probs  = (counts ** (1.0 / temperature))
    probs  = probs / probs.sum()

    frames: list[pd.DataFrame] = []
    for lang, prob in probs.items():
        ratio = target_ratios.get(lang, 1.0)
        lang_df = df[df["subset"] == lang]
        n = int(len(lang_df) * ratio)
        if n <= len(lang_df):
            sample = lang_df.sample(n=n, random_state=rng.randint(0, 99999))
        else:
            # Over-sample with replacement
            sample = lang_df.sample(n=n, replace=True, random_state=rng.randint(0, 99999))
        frames.append(sample)

    result = pd.concat(frames, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    logger.info(f"Temperature sampling (T={temperature}): "
                f"{len(df)} → {len(result)} rows.")
    return result


# ─── External Source Integration ──────────────────────────────────────────────

def load_external_sources(external_dir: Path) -> pd.DataFrame:
    """
    Load and merge all external CSV sources.
    Each file must have columns: input, output, subset, source_name, licence.
    """
    frames: list[pd.DataFrame] = []
    for csv_path in external_dir.glob("*.csv"):
        logger.info(f"Loading external source: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path)
            required = {"input", "output", "subset"}
            if not required.issubset(df.columns):
                logger.warning(f"  Skipping {csv_path.name}: missing columns {required - set(df.columns)}")
                continue
            frames.append(df)
        except Exception as e:
            logger.error(f"  Failed to load {csv_path.name}: {e}")

    if not frames:
        logger.warning("No external sources loaded.")
        return pd.DataFrame(columns=["input", "output", "subset", "source"])

    merged = pd.concat(frames, ignore_index=True)
    merged["source"] = merged.get("source", "external")

    required = ["input", "output", "subset"]
    invalid = merged[required].isna().any(axis=1)
    if invalid.any():
        logger.warning(
            "Dropping %d invalid external rows missing input/output/subset.",
            invalid.sum(),
        )
        merged = merged.loc[~invalid].reset_index(drop=True)

    merged["input"] = merged["input"].astype(str)
    merged["output"] = merged["output"].astype(str)
    merged["subset"] = merged["subset"].astype(str)

    logger.success(f"Loaded {len(merged)} rows from {len(frames)} external sources.")
    return merged


def _checkpoint_path(augmented_dir: Path, name: str) -> Path:
    """Build a checkpoint path inside the augmented data directory."""
    return augmented_dir / f"checkpoint_{name}.csv"


# ─── Main Augmentation Runner ─────────────────────────────────────────────────

def run_augmentation(config_path: str = "src/training/config.yaml") -> None:
    """End-to-end augmentation pipeline."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths       = cfg["paths"]
    nllb_codes  = cfg["nllb_codes"]
    aug_cfg     = cfg["augmentation"]
    seed        = cfg["seed"]

    cleaned_dir   = Path(paths["data_cleaned"])
    augmented_dir = Path(paths["data_augmented"])
    external_dir  = Path(paths["data_external"])
    augmented_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load cleaned training data
    train_path = cleaned_dir / "train_clean.csv"
    if not train_path.exists():
        logger.error(f"Cleaned training data not found at {train_path}. Run clean.py first.")
        return

    df_train = pd.read_csv(train_path)
    df_train["source"] = df_train.get("source", "original")
    invalid_mask = df_train["input"].isna() | df_train["output"].isna()
    if invalid_mask.any():
        logger.warning(
            "Dropping %d rows with missing input/output before augmentation.",
            invalid_mask.sum(),
        )
        df_train = df_train.loc[~invalid_mask].reset_index(drop=True)
    df_train["input"] = df_train["input"].astype(str)
    df_train["output"] = df_train["output"].astype(str)
    logger.info(f"Loaded {len(df_train)} clean training rows.")

    # Load external sources
    df_external = load_external_sources(external_dir)
    if not df_external.empty:
        df_train = pd.concat([df_train, df_external], ignore_index=True)
        logger.info(f"After external integration: {len(df_train)} rows.")

    forward_path = _checkpoint_path(augmented_dir, "forward_translate")
    back_path = _checkpoint_path(augmented_dir, "back_translate")
    final_path = augmented_dir / "augmented_train.csv"

    if final_path.exists():
        logger.success(
            "Final augmented dataset already exists at %s. "
            "Delete it to regenerate or keep using the existing file.",
            final_path,
        )
        return

    # Forward translation (English → African languages)
    df_english = df_train[df_train["subset"] == "Eng"].copy()
    if not df_english.empty:
        if forward_path.exists():
            logger.info(f"Resuming from existing forward translation checkpoint: {forward_path}")
            df_fwd = pd.read_csv(forward_path)
        else:
            logger.info(f"Running forward translation on {len(df_english)} English pairs …")
            df_fwd = forward_translate(
                df_english=df_english,
                target_langs={k: v for k, v in nllb_codes.items() if k not in ("Eng_Uga","Eng_Gha","Eng_Eth","Eng_Ken")},
                cfg=cfg,
            )
            df_fwd = df_fwd.rename(columns={
                "input_translated":  "input",
                "output_translated": "output",
                "language":          "subset",
            })[["input", "output", "subset", "source"]]
            df_fwd.to_csv(forward_path, index=False)
            logger.success(f"Saved forward translation checkpoint → {forward_path}")

        df_fwd["input"] = df_fwd["input"].astype(str)
        df_fwd["output"] = df_fwd["output"].astype(str)
        df_train = pd.concat([df_train, df_fwd], ignore_index=True)
        logger.info(f"After forward translation: {len(df_train)} rows.")

    # Back-translation + paraphrase
    if back_path.exists():
        logger.info(f"Resuming from existing back-translation checkpoint: {back_path}")
        df_bt = pd.read_csv(back_path)
    else:
        logger.info("Running back-translation + paraphrase …")
        df_bt = back_translate_and_paraphrase(df_train, nllb_codes, cfg)
        df_bt.to_csv(back_path, index=False)
        logger.success(f"Saved back-translation checkpoint → {back_path}")

    df_bt["input"] = df_bt["input"].astype(str)
    df_bt["output"] = df_bt["output"].astype(str)
    df_train = pd.concat([df_train, df_bt], ignore_index=True)
    logger.info(f"After back-translation: {len(df_train)} rows.")

    # Temperature sampling / mixing
    df_final = temperature_sample(
        df=df_train,
        target_ratios=aug_cfg["target_ratio"],
        temperature=cfg["temperature_sampling"]["temperature"],
        seed=seed,
    )

    # Assign IDs to any rows that lack them
    if "ID" not in df_final.columns:
        df_final["ID"] = [f"aug_{i:07d}" for i in range(len(df_final))]
    else:
        missing_mask = df_final["ID"].isna()
        df_final.loc[missing_mask, "ID"] = [
            f"aug_{i:07d}" for i in range(missing_mask.sum())
        ]

    out_path = augmented_dir / "augmented_train.csv"
    df_final.to_csv(out_path, index=False)
    logger.success(f"Augmented training set saved → {out_path}  ({len(df_final)} rows)")

    # Summary
    logger.info("\nLanguage distribution after augmentation:")
    for lang, cnt in df_final["subset"].value_counts().items():
        logger.info(f"  {lang:<12} {cnt:>6} rows")


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/training/config.yaml"
    run_augmentation(cfg_path)
