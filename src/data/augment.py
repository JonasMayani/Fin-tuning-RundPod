"""
src/data/augment.py
─────────────────────────────────────────────────────────────────────────────
Data Augmentation Pipeline — RunPod A40 (48 GB VRAM) version
Multilingual African Health Assistant | Zindi ITU Challenge

Key changes vs original:
  - Uses facebook/nllb-200-1.3B (higher quality than distilled-600M)
  - STRICT English→English skip: all subsets whose NLLB code is "eng_Latn"
    are excluded as translation targets AND sources in back-translation
  - Larger batch sizes tuned for A40 48 GB (batch=64 for translation)
  - Parallel tokenisation with fast tokenizer
  - Checkpointing at every language pair so a crash can resume mid-augmentation
  - Cleaner separation of augmentation strategies

Augmentation strategies:
  A. Forward Translation  — English Q&A → non-English African languages (NLLB)
  B. Back-Translation     — African Q&A → English → paraphrase → back (NLLB + T5)
  C. Temperature Sampling — language mixing with target ratios
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# All subset codes whose language is English.
# Used to prevent any English→English translation in both forward and back passes.
ENGLISH_SUBSETS = frozenset({"Eng_Uga", "Eng_Gha", "Eng_Eth", "Eng_Ken"})
ENGLISH_NLLB    = "eng_Latn"


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model cache — load each model once, reuse across calls
# ─────────────────────────────────────────────────────────────────────────────

_models: dict[str, object] = {}


def _get_nllb(model_name: str) -> tuple:
    """
    Load and cache the NLLB translation model.

    On A40 48 GB the 1.3B model loads in bf16 (~2.6 GB) and runs translation
    batches of 64 at ~3–4× the speed of the distilled-600M on a T4.
    The model is pinned to GPU 0 and stays loaded for the entire augmentation run.

    Args:
        model_name: HuggingFace model ID, e.g. "facebook/nllb-200-1.3B".

    Returns:
        (model, tokenizer) tuple.
    """
    key = f"nllb_{model_name}"
    if key not in _models:
        logger.info(f"Loading NLLB model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        # Compile the model for faster inference on A40 (Ampere supports this well)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("NLLB compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile skipped: {e}")
        torch.backends.cuda.enable_flash_sdp(True)   # faster attention on A40
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        _models[key] = (model, tokenizer)
        logger.info(f"NLLB loaded  dtype={dtype}  "
                    f"VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB")
    return _models[key]


def _get_paraphrase_model() -> tuple:
    """
    Load and cache the T5 paraphrase model.

    Uses bf16 on A40 for ~2× memory saving vs fp32.
    Runs alongside NLLB during back-translation — both fit on 48 GB.

    Returns:
        (model, tokenizer) tuple.
    """
    key = "paraphrase"
    if key not in _models:
        logger.info("Loading T5 paraphrase model …")
        model_name = "Vamsi/t5_paraphrase_paws"
        tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        dtype      = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        _models[key] = (model, tokenizer)
    return _models[key]


def _free_model(key_prefix: str) -> None:
    """
    Remove a model from the cache and free its GPU memory.
    Useful when switching between NLLB and paraphrase model on smaller GPUs.
    On A40 48 GB both fit simultaneously so this is optional.
    """
    to_remove = [k for k in _models if k.startswith(key_prefix)]
    for k in to_remove:
        model, _ = _models.pop(k)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Freed model cache: {k}")


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_texts(texts: list) -> list[str]:
    """
    Convert all items to non-empty strings.
    NaN, None, and empty strings are replaced with a space so the tokenizer
    doesn't crash on degenerate inputs.
    """
    result = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            result.append(t.strip())
        elif t is None or (isinstance(t, float) and np.isnan(t)):
            result.append(" ")
        else:
            s = str(t).strip()
            result.append(s if s else " ")
    return result


def compute_chrf(hypotheses: list[str], references: list[str]) -> list[float]:
    """
    Compute sentence-level chrF scores (0–1).
    Ensures all inputs are clean strings before scoring.
    """
    try:
        from sacrebleu.metrics import CHRF
        chrf_metric = CHRF()
        scores = []
        for h, r in zip(hypotheses, references):
            # Convert to string and strip — prevents TypeError from None/float/NaN
            h_clean = str(h).strip() if h is not None else ""
            r_clean = str(r).strip() if r is not None else ""
            if not h_clean or not r_clean:
                scores.append(0.0)
                continue
            try:
                score = chrf_metric.sentence_score(h_clean, [r_clean]).score / 100.0
            except Exception:
                score = 0.0
            scores.append(score)
        return scores
    except ImportError:
        logger.warning("sacrebleu not installed — skipping chrF filter")
        return [1.0] * len(hypotheses)


# ─────────────────────────────────────────────────────────────────────────────
# Core translation function
# ─────────────────────────────────────────────────────────────────────────────

def translate_batch(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    model_name: str,
    num_beams: int = 2,
    batch_size: int = 256,
    max_length: int = 384,
    desc: str = "",
) -> list[str]:
    """
    Translate with automatic batch size reduction on OOM.
    If a batch causes OOM, splits it in half and retries.
    Guarantees completion regardless of sequence length variance.
    """
    if src_lang == ENGLISH_NLLB and tgt_lang == ENGLISH_NLLB:
        logger.warning("eng→eng skipped")
        return _normalize_texts(texts)

    model, tokenizer = _get_nllb(model_name)
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    texts = _normalize_texts(texts)
    label = desc or f"{src_lang}→{tgt_lang}"
    translations: list[str] = []

    i = 0
    current_batch_size = batch_size
    pbar = tqdm(total=len(texts), desc=label)

    while i < len(texts):
        chunk = texts[i : i + current_batch_size]

        try:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                pad_to_multiple_of=8,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    do_sample=False,
                )

            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            translations.extend(decoded)
            del inputs, output_ids

            pbar.update(len(chunk))
            i += current_batch_size

            # Gradually restore batch size after successful batches
            if current_batch_size < batch_size:
                current_batch_size = min(current_batch_size * 2, batch_size)

        except torch.cuda.OutOfMemoryError:
            # OOM — halve the batch size and retry
            del inputs
            torch.cuda.empty_cache()
            new_size = max(1, current_batch_size // 2)
            logger.warning(
                f"OOM at batch={current_batch_size} — "
                f"reducing to {new_size} and retrying"
            )
            current_batch_size = new_size

    pbar.close()
    return translations


# ─────────────────────────────────────────────────────────────────────────────
# Paraphrase function
# ─────────────────────────────────────────────────────────────────────────────

def paraphrase_batch(
    texts: list[str],
    batch_size: int = 128,
    max_length: int = 384,
) -> list[str]:
    """
    Paraphrase a list of English texts using T5.

    Called during back-translation after the African text has been translated
    to English. Paraphrasing before translating back creates more diverse
    training examples than direct round-trip translation.

    Batched for efficiency — on A40 batch_size=64 keeps GPU busy without
    exceeding memory when NLLB is also loaded.

    Args:
        texts:      English strings to paraphrase.
        batch_size: Number of texts per GPU batch.
        max_length: Maximum output token length.

    Returns:
        List of paraphrased strings. Falls back to the original text on error.
    """
    model, tokenizer = _get_paraphrase_model()
    results: list[str] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Paraphrasing", leave=False):
        chunk = [f"paraphrase: {t} </s>" for t in texts[i : i + batch_size]]
        try:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=1.5,
                    num_return_sequences=1,
                )
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)
        except Exception as exc:
            logger.warning(f"Paraphrase batch failed: {exc} — using originals")
            # Fallback: keep original texts for this batch
            results.extend(texts[i : i + batch_size])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Forward translation — English → African languages
# ─────────────────────────────────────────────────────────────────────────────

def forward_translate(
    df_english: pd.DataFrame,
    nllb_codes: dict[str, str],
    cfg: dict,
    augmented_dir: Path,
) -> pd.DataFrame:
    """
    Translate English Q&A pairs into each non-English African language.

    English→English prevention:
      Any subset whose NLLB code is "eng_Latn" is skipped as a target.
      This covers all four English subsets: Eng_Uga, Eng_Gha, Eng_Eth, Eng_Ken.

    Per-language checkpointing:
      Each language pair is saved to a separate checkpoint CSV so a crash
      mid-translation doesn't lose completed languages.

    Quality filter:
      chrF score between the translated input and the original English input
      is used as a proxy for translation quality. Pairs below the threshold
      are dropped.

    Args:
        df_english:    DataFrame of English Q&A rows (any English subset).
        nllb_codes:    Dict mapping subset codes to NLLB BCP-47 codes.
        cfg:           Full config dict.
        augmented_dir: Directory for checkpoint files.

    Returns:
        Combined DataFrame of all translated pairs with columns:
        input, output, subset, source.
    """
    nllb_cfg   = cfg["translation"]
    model_name = nllb_cfg["model_name"]
    num_beams  = nllb_cfg["num_beams"]
    batch_size = nllb_cfg["batch_size"]
    max_length = nllb_cfg["max_length"]
    chrf_global = cfg["augmentation"]["chrf_threshold"]
    chrf_map    = cfg["augmentation"].get("chrf_threshold_per_lang", {})

    # Build target language list: only non-English subsets
    non_english_targets = {
        subset: code
        for subset, code in nllb_codes.items()
        if code != ENGLISH_NLLB          # skip all English subsets
    }

    if not non_english_targets:
        logger.warning("No non-English target languages found — skipping forward translation.")
        return pd.DataFrame(columns=["input", "output", "subset", "source"])

    logger.info(f"Forward translation targets: {list(non_english_targets.keys())}")
    logger.info(f"Translating {len(df_english)} English pairs → "
                f"{len(non_english_targets)} languages using {model_name}")

    inputs_en  = df_english["input"].tolist()
    outputs_en = df_english["output"].tolist()
    all_frames: list[pd.DataFrame] = []

    for subset_tag, tgt_lang in non_english_targets.items():
        # Per-language checkpoint: skip if already translated
        ckpt = augmented_dir / f"fwd_{subset_tag}.csv"
        if ckpt.exists():
            logger.info(f"  [{subset_tag}] resuming from checkpoint {ckpt.name}")
            all_frames.append(pd.read_csv(ckpt))
            continue

        logger.info(f"  [{subset_tag}] {ENGLISH_NLLB} → {tgt_lang} …")

        translated_inputs  = translate_batch(
            inputs_en, ENGLISH_NLLB, tgt_lang, model_name,
            num_beams, batch_size, max_length,
            desc=f"Fwd inputs  [{subset_tag}]",
        )
        output_batch_size = max(16, batch_size // 4)  # 64→16
        translated_outputs = translate_batch(
            outputs_en, ENGLISH_NLLB, tgt_lang, model_name,
            num_beams, output_batch_size, max_length,
            desc=f"Fwd outputs [{subset_tag}]"
        )

        # Quality filter: chrF of translated input vs original English input
        chrf_thr    = chrf_map.get(subset_tag, chrf_global)
        chrf_scores = compute_chrf(translated_inputs, inputs_en)

        rows = []
        for ti, to, chrf in zip(translated_inputs, translated_outputs, chrf_scores):
            if chrf < chrf_thr:
                continue
            ti_clean = ti.strip()
            to_clean = to.strip()
            if ti_clean and to_clean:
                rows.append({
                    "input":  ti_clean,
                    "output": to_clean,
                    "subset": subset_tag,
                    "source": "forward_mt",
                })

        df_lang = pd.DataFrame(rows)
        df_lang.to_csv(ckpt, index=False)
        logger.info(f"  [{subset_tag}] {len(df_lang)} pairs saved "
                    f"(kept {len(df_lang)}/{len(inputs_en)}, "
                    f"chrF≥{chrf_thr})")
        all_frames.append(df_lang)

    if not all_frames:
        return pd.DataFrame(columns=["input", "output", "subset", "source"])

    result = pd.concat(all_frames, ignore_index=True)
    logger.success(f"Forward translation complete: {len(result)} total pairs")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Back-translation + paraphrase — African → English → paraphrase → African
# ─────────────────────────────────────────────────────────────────────────────

def back_translate_and_paraphrase(
    df: pd.DataFrame,
    nllb_codes: dict[str, str],
    cfg: dict,
    augmented_dir: Path,
) -> pd.DataFrame:
    """
    Augment non-English African language pairs via back-translation.

    English→English prevention:
      Any subset whose NLLB code is "eng_Latn" is entirely skipped.
      Only truly non-English subsets (Aka_Gha, Lug_Uga, Swa_Ken, Amh_Eth)
      are processed.

    Pipeline per language:
      1. Translate African input → English (NLLB)
      2. Paraphrase the English input (T5) — creates diversity
      3. Translate paraphrased English → original African language (NLLB)
      4. Keep the original African output as the target label

    Per-language checkpointing:
      Each language saves to its own checkpoint CSV.

    Args:
        df:            Full training DataFrame (all languages).
        nllb_codes:    Dict mapping subset codes to NLLB codes.
        cfg:           Full config dict.
        augmented_dir: Directory for checkpoint files.

    Returns:
        DataFrame of back-translated pairs with columns: ID, input, output, subset, source.
    """
    nllb_cfg   = cfg["translation"]
    model_name = nllb_cfg["model_name"]
    num_beams  = nllb_cfg["num_beams"]
    batch_size = nllb_cfg["batch_size"]
    max_length = nllb_cfg["max_length"]

    # Only process non-English subsets that are actually in the data
    non_english_subsets = {
        subset: code
        for subset, code in nllb_codes.items()
        if code != ENGLISH_NLLB and subset in df["subset"].unique()
    }

    if not non_english_subsets:
        logger.warning("No non-English subsets found for back-translation.")
        return pd.DataFrame(columns=["ID", "input", "output", "subset", "source"])

    logger.info(f"Back-translation subsets: {list(non_english_subsets.keys())}")
    all_frames: list[pd.DataFrame] = []

    for subset_tag, af_lang in non_english_subsets.items():
        # Per-language checkpoint
        ckpt = augmented_dir / f"bt_{subset_tag}.csv"
        if ckpt.exists():
            logger.info(f"  [{subset_tag}] resuming from checkpoint {ckpt.name}")
            all_frames.append(pd.read_csv(ckpt))
            continue

        group = df[df["subset"] == subset_tag].copy()
        logger.info(f"  [{subset_tag}] {len(group)} pairs  {af_lang} → "
                    f"{ENGLISH_NLLB} → paraphrase → {af_lang}")

        inputs_af = group["input"].tolist()

        # Step 1: African → English
        inputs_en = translate_batch(
            inputs_af, af_lang, ENGLISH_NLLB, model_name,
            num_beams, batch_size, max_length,
            desc=f"BT step1 [{subset_tag}]",
        )

        # Step 2: Paraphrase English
        # Paraphrase batch_size is smaller (32) since T5 and NLLB are both loaded
        para_batch = min(32, batch_size // 2)
        paraphrased_en = paraphrase_batch(inputs_en, batch_size=para_batch,
                                          max_length=max_length)

        # Step 3: Paraphrased English → African language
        back_translated = translate_batch(
            paraphrased_en, ENGLISH_NLLB, af_lang, model_name,
            num_beams, batch_size, max_length,
            desc=f"BT step3 [{subset_tag}]",
        )

        rows = []
        for i, (orig_row, bt_input) in enumerate(
            zip(group.itertuples(index=False), back_translated)
        ):
            bt_clean = bt_input.strip()
            if not bt_clean:
                continue
            orig_id = getattr(orig_row, "ID", f"bt_{subset_tag}_{i}")
            rows.append({
                "ID":     f"{orig_id}_bt",
                "input":  bt_clean,
                "output": orig_row.output,
                "subset": subset_tag,
                "source": "back_bt",
            })

        df_bt = pd.DataFrame(rows)
        df_bt.to_csv(ckpt, index=False)
        logger.info(f"  [{subset_tag}] {len(df_bt)} back-translated pairs saved")
        all_frames.append(df_bt)

    if not all_frames:
        return pd.DataFrame(columns=["ID", "input", "output", "subset", "source"])

    result = pd.concat(all_frames, ignore_index=True)
    logger.success(f"Back-translation complete: {len(result)} total pairs")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Temperature-based data mixing
# ─────────────────────────────────────────────────────────────────────────────

def temperature_sample(
    df: pd.DataFrame,
    target_ratios: dict[str, float],
    temperature: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Balance the language distribution using temperature sampling.

    Temperature controls how much to correct for language imbalance:
      T → ∞ : uniform sampling (all languages equally represented)
      T = 1 : proportional sampling (no correction)
      T < 1 : amplifies majority languages (rarely useful)

    Each language is then over/under-sampled according to target_ratio
    from the config to further boost low-resource languages.

    Args:
        df:            Full augmented DataFrame with a 'subset' column.
        target_ratios: Dict mapping subset codes to sampling multipliers.
        temperature:   Sampling temperature (default 5.0 from config).
        seed:          Random seed for reproducibility.

    Returns:
        Shuffled DataFrame with the balanced language distribution.
    """
    rng    = np.random.RandomState(seed)
    counts = df["subset"].value_counts()
    probs  = counts ** (1.0 / temperature)
    probs  = probs / probs.sum()

    frames: list[pd.DataFrame] = []
    for lang in probs.index:
        ratio   = target_ratios.get(lang, 1.0)
        lang_df = df[df["subset"] == lang]
        n       = int(len(lang_df) * ratio)
        if n <= 0:
            continue
        rs = rng.randint(0, 99999)
        if n <= len(lang_df):
            sample = lang_df.sample(n=n, random_state=rs)
        else:
            sample = lang_df.sample(n=n, replace=True, random_state=rs)
        frames.append(sample)

    result = (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    logger.info(f"Temperature sampling (T={temperature}): {len(df)} → {len(result)} rows")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# External source loader
# ─────────────────────────────────────────────────────────────────────────────

def load_external_sources(external_dir: Path) -> pd.DataFrame:
    """
    Load and merge all external CSV sources from the external data directory.

    Each CSV must have columns: input, output, subset.
    Optional columns (source_name, licence) are included if present.
    Rows with missing required columns are dropped with a warning.

    Args:
        external_dir: Directory containing external CSV files.

    Returns:
        Combined DataFrame, or empty DataFrame if no valid sources found.
    """
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(external_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            missing = {"input", "output", "subset"} - set(df.columns)
            if missing:
                logger.warning(f"Skipping {csv_path.name}: missing {missing}")
                continue
            df["source"] = df.get("source", csv_path.stem)
            frames.append(df)
            logger.info(f"External: {csv_path.name}  ({len(df)} rows)")
        except Exception as exc:
            logger.error(f"Failed to load {csv_path.name}: {exc}")

    if not frames:
        logger.info("No external sources found.")
        return pd.DataFrame(columns=["input", "output", "subset", "source"])

    merged = pd.concat(frames, ignore_index=True)
    bad    = merged[["input", "output", "subset"]].isna().any(axis=1)
    if bad.any():
        logger.warning(f"Dropping {bad.sum()} external rows with missing values")
        merged = merged[~bad].reset_index(drop=True)

    for col in ("input", "output", "subset"):
        merged[col] = merged[col].astype(str).str.strip()
    merged = merged[(merged["input"] != "") & (merged["output"] != "")].reset_index(drop=True)

    logger.success(f"Loaded {len(merged)} external rows from {len(frames)} files")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main augmentation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_augmentation(config_path: str = "src/training/config.yaml") -> None:
    """
    End-to-end augmentation pipeline.

    Order of operations:
      1. Load cleaned training data + external sources
      2. Forward translate English pairs → non-English languages
      3. Back-translate non-English pairs → English → paraphrase → back
      4. Apply temperature sampling to balance language distribution
      5. Save final_train.csv

    All translation steps checkpoint per-language so a crash resumes
    without re-translating completed languages.

    Args:
        config_path: Path to config.yaml.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths      = cfg["paths"]
    nllb_codes = cfg["nllb_codes"]
    aug_cfg    = cfg["augmentation"]
    seed       = int(cfg["seed"])

    cleaned_dir   = Path(paths["data_cleaned"])
    augmented_dir = Path(paths["data_augmented"])
    external_dir  = Path(paths["data_external"])
    augmented_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Check if final output already exists ──────────────────────────────────
    final_path = augmented_dir / "final_train.csv"
    if final_path.exists():
        logger.success(
            f"final_train.csv already exists ({final_path}). "
            "Delete it to re-run augmentation."
        )
        return

    # ── Load base training data ───────────────────────────────────────────────
    train_path = cleaned_dir / "train_clean.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Cleaned training data not found: {train_path}")

    df_train = pd.read_csv(train_path)
    df_train["source"] = "original"
    for col in ("input", "output", "subset"):
        df_train[col] = df_train[col].astype(str).str.strip()
    df_train = df_train[
        (df_train["input"] != "") & (df_train["output"] != "") & (df_train["subset"] != "")
    ].copy()
    logger.info(f"Base training data: {len(df_train)} rows")

    # Language distribution before augmentation
    logger.info("Base language distribution:")
    for lang, cnt in df_train["subset"].value_counts().items():
        is_eng = "  [English — will not be back-translated]" if lang in ENGLISH_SUBSETS else ""
        logger.info(f"  {lang:<12} {cnt:>6} rows{is_eng}")

    # ── Load external sources ─────────────────────────────────────────────────
    if external_dir.exists():
        df_ext = load_external_sources(external_dir)
        if not df_ext.empty:
            df_train = pd.concat([df_train, df_ext], ignore_index=True)
            logger.info(f"After external: {len(df_train)} rows")

    # ── Forward translation ───────────────────────────────────────────────────
    # Collect all English rows (from any English subset) as translation source
    df_english = df_train[df_train["subset"].isin(ENGLISH_SUBSETS)].copy()
    logger.info(f"English source rows for forward translation: {len(df_english)}")

    df_fwd = forward_translate(
        df_english=df_english,
        nllb_codes=nllb_codes,
        cfg=cfg,
        augmented_dir=augmented_dir,
    )
    if not df_fwd.empty:
        df_train = pd.concat([df_train, df_fwd], ignore_index=True)
        logger.info(f"After forward translation: {len(df_train)} rows")

    # ── Back-translation + paraphrase ─────────────────────────────────────────
    df_bt = back_translate_and_paraphrase(
        df=df_train,
        nllb_codes=nllb_codes,
        cfg=cfg,
        augmented_dir=augmented_dir,
    )
    if not df_bt.empty:
        df_train = pd.concat([df_train, df_bt], ignore_index=True)
        logger.info(f"After back-translation: {len(df_train)} rows")

    # ── Temperature sampling ──────────────────────────────────────────────────
    df_final = temperature_sample(
        df=df_train,
        target_ratios=aug_cfg["target_ratio"],
        temperature=cfg["temperature_sampling"]["temperature"],
        seed=seed,
    )

    # Assign IDs to any rows missing them
    if "ID" not in df_final.columns:
        df_final.insert(0, "ID", [f"aug_{i:08d}" for i in range(len(df_final))])
    else:
        mask = df_final["ID"].isna() | (df_final["ID"].astype(str).str.strip() == "")
        if mask.any():
            ids = df_final["ID"].astype(str).copy()
            ids[mask] = [f"aug_{i:08d}" for i in range(mask.sum())]
            df_final["ID"] = ids

    # ── Save final dataset ────────────────────────────────────────────────────
    df_final.to_csv(final_path, index=False)
    logger.success(f"✅ Saved final_train.csv → {final_path}  ({len(df_final)} rows)")

    # Final language distribution
    logger.info("\nFinal language distribution:")
    for lang, cnt in df_final["subset"].value_counts().items():
        pct = cnt / len(df_final) * 100
        logger.info(f"  {lang:<12} {cnt:>7} rows  ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/training/config.yaml"
    run_augmentation(cfg_path)
