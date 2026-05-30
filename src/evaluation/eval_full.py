"""
eval_full.py — full-quality evaluation of a trained mT0 adapter

Runs the production decoding config (beam search, length/repetition penalties,
per-language overrides) on the entire validation set, NOT the 800-sample
training subsample. Reports overall + per-language ROUGE-1 and ROUGE-L, dumps
all predictions for inspection.

This is the metric that matters for the Zindi leaderboard / submission.

Usage:
    python eval_full.py --config config_mt0.yaml
    python eval_full.py --config config_mt0.yaml --adapter /path/to/phase_2_complete
    python eval_full.py --config config_mt0.yaml --batch_size 8 --val_file data/cleaned/val_clean.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANGUAGE_NAMES: dict[str, str] = {
    "Eng_Uga": "English", "Aka_Gha": "Akan",   "Eng_Gha": "English",
    "Eng_Eth": "English", "Lug_Uga": "Luganda", "Eng_Ken": "English",
    "Swa_Ken": "Swahili", "Amh_Eth": "Amharic",
}


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def build_prompt(question: Any, language: Any) -> str:
    lang = str(language)
    return (
        f"Question: {str(question).strip()}\n"
        f"Answer in {LANGUAGE_NAMES.get(lang, lang)}:"
    )


def resolve_adapter_path(cfg: dict, override: Optional[str]) -> Path:
    """Find the trained adapter checkpoint. Priority: CLI override → 'best' →
    phase_2_complete → phase_1_complete."""
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"--adapter path does not exist: {p}")
        return p

    paths = cfg["paths"]
    base_model = cfg["model"]["base_model"]
    output_name = base_model.replace("/", "_")
    suffix = str(cfg["training"].get("output_dir_suffix", "")).strip()
    if suffix:
        output_name = f"{output_name}_{suffix}"
    output_dir = Path(paths["models"]) / output_name

    candidates = [
        output_dir / "best",
        output_dir / "phase_2_complete",
        output_dir / "phase_1_complete",
    ]
    for c in candidates:
        if (c / "adapter_config.json").exists():
            return c
    raise FileNotFoundError(
        f"No trained adapter found under {output_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_model_and_tokenizer(cfg: dict, adapter_path: Path):
    base_model = cfg["model"]["base_model"]
    use_fast = bool(cfg.get("model", {}).get("use_fast_tokenizer", False))

    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=use_fast)

    precision = cfg["training"].get("mixed_precision", "bf16")
    dtype = (torch.bfloat16 if precision == "bf16"
             else torch.float16 if precision == "fp16"
             else torch.float32)

    logger.info(f"Loading base model: {base_model}  dtype={dtype}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).cuda()
    model.config.use_cache = True  # speeds up inference vs training

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # Block sentinel tokens from generation (mT0 inherits mT5's <extra_id_*>)
    if bool(cfg.get("model", {}).get("block_sentinel_tokens", True)):
        bad_words_ids = []
        for i in range(100):
            tid = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            if tid is not None and tid != tokenizer.unk_token_id:
                bad_words_ids.append([tid])
        if bad_words_ids:
            model.generation_config.bad_words_ids = bad_words_ids

    return model, tokenizer


def decoding_kwargs_for_language(cfg: dict, language: str, model_cfg_max_output: int) -> dict:
    """Merge default decoding + per-language overrides from the config."""
    decoding = cfg.get("decoding", {})
    default = decoding.get("default", {})
    per_lang = decoding.get("per_language", {}).get(language, {})
    merged = {**default, **per_lang}

    return {
        "num_beams":            int(merged.get("num_beams", 4)),
        "length_penalty":       float(merged.get("length_penalty", 1.0)),
        "no_repeat_ngram_size": int(merged.get("no_repeat_ngram", 0)),
        "repetition_penalty":   float(merged.get("repetition_penalty", 1.0)),
        "min_new_tokens":       int(merged.get("min_length", 10)),
        "max_new_tokens":       model_cfg_max_output,
        "early_stopping":       True,
        "do_sample":            False,
    }


def generate_batch(model, tokenizer, prompts: list[str], gen_kwargs: dict,
                   max_input_length: int) -> list[str]:
    """Tokenize a batch of prompts and generate completions."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input_length,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs, **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [t.strip() for t in tokenizer.batch_decode(out, skip_special_tokens=True)]


class IntlTokenizer:
    """Unicode-aware tokenizer for ROUGE.

    rouge_score's default tokenizer uses regex [^a-z0-9]+ to strip
    'non-alphanumeric' characters. That regex is ASCII-only, so EVERY
    Amharic character (and every accented char in Akan / Luganda) gets
    stripped, leaving an empty token list. ROUGE then returns 0 even
    for identical text.

    This tokenizer uses the Unicode-aware \\w+ pattern that recognizes
    letters in ALL scripts (Latin, Ethiopic, Arabic, CJK, etc.).
    Works correctly for every language in this project.
    """
    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        text = unicodedata.normalize("NFKC", text.lower())
        return re.findall(r"\w+", text, flags=re.UNICODE)


def compute_rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    # Pass our custom tokenizer — this is the fix for Amharic / Akan / Luganda
    # scoring 0 on identical text due to the rouge_score library default
    # being ASCII-only.
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        tokenizer=IntlTokenizer(),
        use_stemmer=False,
    )
    r1, rl = [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1)) if r1 else 0.0,
        "rougeL": float(np.mean(rl)) if rl else 0.0,
        "n":      len(preds),
    }


def evaluate(cfg: dict, args) -> None:
    setup_logging()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    adapter_path = resolve_adapter_path(cfg, args.adapter)
    logger.info(f"Adapter: {adapter_path}")

    paths = cfg["paths"]
    workspace = Path(paths.get("workspace", "/workspace/Fin-tuning-RundPod"))
    val_path = Path(args.val_file or cfg["dataset"]["val_file"])
    if not val_path.is_absolute():
        val_path = workspace / val_path
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    logger.info(f"Val file: {val_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, adapter_path)

    # ── Load + clean val data ─────────────────────────────────────────────────
    df = pd.read_csv(val_path)
    df = df.dropna(subset=["input", "output", "subset"]).copy()
    for c in ["input", "output", "subset"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "") & (df["subset"] != "")].reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit).copy()
        logger.warning(f"--limit={args.limit} applied; not evaluating full set")
    logger.info(f"Evaluating on {len(df):,} examples across {df['subset'].nunique()} languages")

    df["prompt"] = [build_prompt(q, s) for q, s in zip(df["input"], df["subset"])]

    # ── Generate, grouped by language for per-lang decoding overrides ─────────
    max_in  = cfg["model"]["max_input_length"]
    max_out = cfg["model"]["max_output_length"]
    batch_size = int(args.batch_size)

    predictions: list[Optional[str]] = [None] * len(df)
    by_lang_groups = df.groupby("subset", sort=False).indices  # {lang: np.array_of_indices}

    t0 = time.time()
    total_done = 0
    for lang, indices in by_lang_groups.items():
        gen_kwargs = decoding_kwargs_for_language(cfg, lang, max_out)
        logger.info(
            f"[{lang}] n={len(indices):,}  beams={gen_kwargs['num_beams']}  "
            f"len_pen={gen_kwargs['length_penalty']}  "
            f"no_rep_ng={gen_kwargs['no_repeat_ngram_size']}  "
            f"rep_pen={gen_kwargs['repetition_penalty']}"
        )

        idx_list = list(indices)
        for start in range(0, len(idx_list), batch_size):
            batch_idx = idx_list[start:start + batch_size]
            prompts = [df.at[i, "prompt"] for i in batch_idx]
            try:
                preds = generate_batch(model, tokenizer, prompts, gen_kwargs, max_in)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"OOM with batch_size={len(prompts)} on {lang}. "
                    "Retrying one-by-one for this batch."
                )
                torch.cuda.empty_cache()
                preds = []
                for p in prompts:
                    preds.extend(generate_batch(model, tokenizer, [p], gen_kwargs, max_in))
            for i, p in zip(batch_idx, preds):
                predictions[i] = p
            total_done += len(batch_idx)
            if total_done % (batch_size * 10) == 0:
                rate = total_done / max(time.time() - t0, 1e-9)
                eta_s = (len(df) - total_done) / max(rate, 1e-9)
                logger.info(f"  progress {total_done:,}/{len(df):,}  "
                           f"({rate:.1f} ex/s, ETA {eta_s/60:.1f} min)")

    elapsed = time.time() - t0
    logger.info(f"Generation complete in {elapsed/60:.1f} min "
                f"({len(df)/elapsed:.1f} ex/s)")

    df["prediction"] = predictions

    # ── Scoring ───────────────────────────────────────────────────────────────
    overall = compute_rouge(df["prediction"].tolist(), df["output"].tolist())
    print()
    print("=" * 72)
    print(f"{'OVERALL':<14} n={overall['n']:>5}  "
          f"ROUGE-1={overall['rouge1']:.4f}  ROUGE-L={overall['rougeL']:.4f}")
    print("=" * 72)

    per_lang_rows = []
    for lang in sorted(df["subset"].unique()):
        sub = df[df["subset"] == lang]
        scores = compute_rouge(sub["prediction"].tolist(), sub["output"].tolist())
        per_lang_rows.append({
            "language": lang,
            "n":        scores["n"],
            "rouge1":   scores["rouge1"],
            "rougeL":   scores["rougeL"],
        })
        print(f"  {lang:<10} n={scores['n']:>5}  "
              f"ROUGE-1={scores['rouge1']:.4f}  ROUGE-L={scores['rougeL']:.4f}")
    print()

    # ── Save predictions + per-lang summary ────────────────────────────────────
    out_dir = adapter_path.parent / "final_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "predictions_full.csv"
    df.to_csv(preds_path, index=False, encoding="utf-8")
    logger.info(f"Predictions → {preds_path}")

    summary_path = out_dir / "rouge_by_language.csv"
    summary_df = pd.DataFrame(per_lang_rows + [{
        "language": "OVERALL", "n": overall["n"],
        "rouge1": overall["rouge1"], "rougeL": overall["rougeL"],
    }])
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info(f"Summary    → {summary_path}")

    # Per-language quality buckets (helps see where the work is)
    print("Quality buckets (ROUGE-L):")
    for row in per_lang_rows:
        bar = "█" * int(row["rougeL"] * 50)
        flag = " ← needs attention" if row["rougeL"] < 0.20 else (
               " ← strong" if row["rougeL"] >= 0.35 else "")
        print(f"  {row['language']:<10} {row['rougeL']:.3f}  {bar}{flag}")


def parse_args():
    p = argparse.ArgumentParser(description="Full-quality eval of trained mT0 adapter")
    p.add_argument("--config",  default="/workspace/Fin-tuning-RundPod/src/training/config_mt0.yaml")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter dir (default: auto-detect)")
    p.add_argument("--val_file", default=None, help="Override val CSV (default: from config)")
    p.add_argument("--batch_size", type=int, default=8, help="Generation batch size (default 8)")
    p.add_argument("--limit", type=int, default=0, help="Debug: cap to first N examples (0 = all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    evaluate(config, args)