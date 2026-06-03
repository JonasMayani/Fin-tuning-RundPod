"""
eval_full.py — full evaluation pipeline for trained mT0 adapter.

Replaces both the prior eval_full.py and metrics.py. Single source of truth
for evaluation. Computes:
    • ROUGE-1 F1 and ROUGE-L F1 (Unicode-aware tokenizer — works on Amharic)
    • Per-language breakdown
    • LLM-as-a-Judge (Mistral-7B-Instruct local OR Claude API)
    • BERTScore (optional, Phase 2 metric)
    • Weighted competition score (37% R1 + 37% RL + 26% LLM)
    • Zindi submission CSV (optional)

Usage:
    # Quick eval — ROUGE only on full val set
    python eval_full.py --config config_v3.yaml

    # Full eval with LLM judge (sampled), no submission
    python eval_full.py --config config_v3.yaml --llm_judge

    # Full eval + generate test submission
    python eval_full.py --config config_v3.yaml --llm_judge --make_submission

    # Use Claude API for judge (faster, better quality)
    python eval_full.py --config config_v3.yaml --llm_judge --judge_backend claude

    # Debug: limit val set
    python eval_full.py --config config_v3.yaml --limit 200
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import unicodedata
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
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

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
    """Identical to train_mt0.py's prompt — must match for correct eval."""
    lang = str(language)
    return (
        f"Question: {str(question).strip()}\n"
        f"Answer in {LANGUAGE_NAMES.get(lang, lang)}:"
    )


# ─── ROUGE — Unicode-aware tokenizer ────────────────────────────────────────

class IntlTokenizer:
    """Unicode-aware tokenizer for ROUGE.

    rouge_score's default uses regex [^a-z0-9]+ to strip non-alphanumeric chars.
    That regex is ASCII-only, so every Amharic character gets stripped — both
    predictions AND references tokenize to []. ROUGE then returns 0 on
    identical Amharic text. This tokenizer uses \\w+ (Unicode-aware) instead,
    which works for every language in this project.
    """
    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        text = unicodedata.normalize("NFKC", text.lower())
        return re.findall(r"\w+", text, flags=re.UNICODE)


def compute_rouge_scores(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Macro-average ROUGE-1 F1 and ROUGE-L F1 using IntlTokenizer."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        tokenizer=IntlTokenizer(),
        use_stemmer=False,
    )
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score((ref or "").strip(), (pred or "").strip())
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1_f1": float(np.mean(r1)) if r1 else 0.0,
        "rougeL_f1": float(np.mean(rl)) if rl else 0.0,
        "rouge1_scores": r1,
        "rougeL_scores": rl,
        "n": len(predictions),
    }


def evaluate_per_language(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    ref_col: str = "output",
    subset_col: str = "subset",
) -> pd.DataFrame:
    """Per-language ROUGE breakdown + overall."""
    rows = []
    for lang in sorted(df[subset_col].unique()):
        sub = df[df[subset_col] == lang]
        preds = sub[pred_col].fillna("").tolist()
        refs = sub[ref_col].fillna("").tolist()
        r = compute_rouge_scores(preds, refs)
        rows.append({
            "language": lang,
            "n":        len(sub),
            "rouge1_f1": r["rouge1_f1"],
            "rougeL_f1": r["rougeL_f1"],
        })
    overall = compute_rouge_scores(
        df[pred_col].fillna("").tolist(),
        df[ref_col].fillna("").tolist(),
    )
    rows.append({
        "language": "OVERALL",
        "n":        len(df),
        "rouge1_f1": overall["rouge1_f1"],
        "rougeL_f1": overall["rougeL_f1"],
    })
    return pd.DataFrame(rows)


# ─── Inference — load LoRA adapter, group by language for correct decoding ───

def resolve_adapter_path(cfg: dict, override: Optional[str]) -> Path:
    """Locate the trained LoRA adapter. Priority: override → 'best' →
    'phase_2_complete' → 'phase_1_complete'."""
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
    """Load base mT0 + apply LoRA adapter. Matches training precision (bf16)."""
    base_model = cfg["model"]["base_model"]
    use_fast = bool(cfg.get("model", {}).get("use_fast_tokenizer", False))

    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=use_fast)

    # Match training precision — bf16 is what was used during fine-tuning.
    # fp16 here would risk numerical instability with mT0's large vocab.
    precision = cfg["training"].get("mixed_precision", "bf16")
    dtype = (torch.bfloat16 if precision == "bf16"
             else torch.float16 if precision == "fp16"
             else torch.float32)

    logger.info(f"Loading base model: {base_model}  dtype={dtype}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = True  # speeds up inference vs training

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # Block mT0/mT5 sentinel tokens from generation
    if bool(cfg.get("model", {}).get("block_sentinel_tokens", True)):
        bad_words_ids = []
        for i in range(100):
            tid = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            if tid is not None and tid != tokenizer.unk_token_id:
                bad_words_ids.append([tid])
        if bad_words_ids:
            model.generation_config.bad_words_ids = bad_words_ids
            logger.info(f"Generation guard: blocking {len(bad_words_ids)} sentinel tokens")

    return model, tokenizer


def decoding_kwargs_for_language(cfg: dict, language: str, max_output: int) -> dict:
    """Merge default decoding + per-language overrides."""
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
        "max_new_tokens":       max_output,
        "early_stopping":       True,
        "do_sample":            False,
    }


def generate_batch(
    model, tokenizer, prompts: list[str], gen_kwargs: dict, max_input: int,
) -> list[str]:
    """Generate completions for a batch."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs, **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [t.strip() for t in tokenizer.batch_decode(out, skip_special_tokens=True)]


def generate_predictions(
    df: pd.DataFrame,
    model, tokenizer,
    cfg: dict,
    batch_size: int = 8,
    desc: str = "Generating",
) -> list[str]:
    """Generate predictions for all rows. Groups by language so each language
    gets its own decoding config (bug fix from metrics.py which used the
    majority language of each batch — wrong for mixed batches)."""
    max_in  = cfg["model"]["max_input_length"]
    max_out = cfg["model"]["max_output_length"]

    df = df.copy()
    if "prompt" not in df.columns:
        df["prompt"] = [build_prompt(q, s) for q, s in zip(df["input"], df["subset"])]

    predictions: list[Optional[str]] = [None] * len(df)
    by_lang_groups = df.groupby("subset", sort=False).indices

    t0 = time.time()
    total_done = 0
    for lang, indices in by_lang_groups.items():
        gen_kwargs = decoding_kwargs_for_language(cfg, lang, max_out)
        logger.info(
            f"[{lang}] n={len(indices):,}  beams={gen_kwargs['num_beams']}  "
            f"len_pen={gen_kwargs['length_penalty']}  "
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
                    f"OOM at batch_size={len(prompts)} on {lang}; retrying one-by-one"
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
                eta_min = (len(df) - total_done) / max(rate, 1e-9) / 60
                logger.info(f"  {desc} progress {total_done:,}/{len(df):,}  "
                            f"({rate:.1f} ex/s, ETA {eta_min:.1f} min)")
    elapsed = time.time() - t0
    logger.info(f"{desc} complete in {elapsed/60:.1f} min ({len(df)/elapsed:.1f} ex/s)")
    return predictions


# ─── LLM-as-a-Judge ─────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for multilingual health question-answering systems.
Your task is to evaluate a model's answer against a reference answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Please rate the model answer on a scale of 1-5 for each criterion:
- accuracy: medical correctness and safety (1=dangerous/wrong, 5=correct and safe)
- completeness: how fully the question is addressed (1=missing key info, 5=complete)
- language: grammatical naturalness and fluency in the question's language (1=unreadable, 5=native-like)

Respond ONLY with a valid JSON object, no other text:
{{"accuracy": <int>, "completeness": <int>, "language": <int>, "overall": <float>}}"""


def _parse_judge_response(text: str) -> float:
    """Extract overall score from judge JSON response. Falls back to 3.0 on failure."""
    try:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if "overall" in data:
                return float(data["overall"])
            scores = [data.get(k, 3.0) for k in ("accuracy", "completeness", "language")]
            return float(np.mean(scores))
    except Exception:
        pass
    nums = re.findall(r"\b([1-5](?:\.\d+)?)\b", text)
    if nums:
        # Prefer a decimal value (e.g. "4.5") over a bare integer (e.g. the
        # "5" in "4.5 / 5", which is usually the scale denominator). Among
        # decimals, take the first; otherwise take the first integer.
        decimals = [n for n in nums if "." in n]
        return float(decimals[0]) if decimals else float(nums[0])
    return 3.0


def judge_local_mistral(
    questions: list[str],
    predictions: list[str],
    references: list[str],
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    batch_size: int = 4,
) -> list[float]:
    """Run Mistral-7B-Instruct locally as judge.

    Note: we use a batched chat-template approach. Mistral-7B in bf16 needs
    ~14 GB; runs comfortably alongside mT0-XL if you don't unload it first,
    but we recommend unloading mT0 before judging (call gc + empty_cache)."""
    logger.info(f"Loading judge model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="cuda",
    )
    model.eval()

    scores: list[float] = []
    n = len(questions)
    try:
        for start in tqdm(range(0, n, batch_size), desc="LLM Judge (Mistral)"):
            batch_q = questions[start:start + batch_size]
            batch_p = predictions[start:start + batch_size]
            batch_r = references[start:start + batch_size]

            chat_prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
                        question=q[:300], reference=r[:400], prediction=p[:400])}],
                    tokenize=False, add_generation_prompt=True,
                )
                for q, p, r in zip(batch_q, batch_p, batch_r)
            ]
            enc = tok(chat_prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to("cuda")
            try:
                with torch.inference_mode():
                    out = model.generate(
                        **enc, max_new_tokens=80,
                        do_sample=False, temperature=1.0,
                        pad_token_id=tok.pad_token_id,
                    )
                # Strip the prompt portion per row
                for i in range(out.size(0)):
                    new_tokens = out[i][enc["input_ids"].shape[1]:]
                    text = tok.decode(new_tokens, skip_special_tokens=True)
                    scores.append(_parse_judge_response(text))
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"Judge OOM at batch_size={len(chat_prompts)}; one-by-one")
                torch.cuda.empty_cache()
                for cp in chat_prompts:
                    enc1 = tok(cp, return_tensors="pt", truncation=True,
                               max_length=1024).to("cuda")
                    with torch.inference_mode():
                        out = model.generate(
                            **enc1, max_new_tokens=80,
                            do_sample=False, temperature=1.0,
                            pad_token_id=tok.pad_token_id,
                        )
                    new_tokens = out[0][enc1["input_ids"].shape[1]:]
                    text = tok.decode(new_tokens, skip_special_tokens=True)
                    scores.append(_parse_judge_response(text))
    finally:
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
    return scores


def judge_via_claude_api(
    questions: list[str],
    predictions: list[str],
    references: list[str],
    model: str = "claude-sonnet-4-5",
) -> list[float]:
    """Run judge via Anthropic API. Requires ANTHROPIC_API_KEY env var."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic()
    scores: list[float] = []
    for q, p, r in tqdm(zip(questions, predictions, references),
                        total=len(questions), desc="LLM Judge (Claude API)"):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=q[:300], reference=r[:400], prediction=p[:400])
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model, max_tokens=128,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text if msg.content else ""
                scores.append(_parse_judge_response(text))
                break
            except Exception as exc:
                if attempt == 2:
                    logger.warning(f"Claude failed 3x: {exc}")
                    scores.append(3.0)
                else:
                    time.sleep(2 ** attempt)
    return scores


def sample_for_judge(
    df: pd.DataFrame,
    n_per_lang: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample for judge — n rows per language."""
    pieces = []
    for lang, group in df.groupby("subset"):
        take = min(n_per_lang, len(group))
        pieces.append(group.sample(n=take, random_state=seed))
    return pd.concat(pieces, ignore_index=True)


# ─── BERTScore (Phase 2) ────────────────────────────────────────────────────

def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "Davlan/afro-xlmr-base",
) -> dict[str, float]:
    """BERTScore with an Afro-XLMR encoder (covers all project languages)."""
    try:
        from bert_score import score as bs_score
    except ImportError:
        logger.warning("pip install bert-score to enable Phase 2 BERTScore")
        return {"bertscore_f1": float("nan")}
    try:
        P, R, F1 = bs_score(
            cands=predictions, refs=references,
            model_type=model_type, lang="multilingual",
            verbose=False, batch_size=32,
        )
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall":    float(R.mean()),
            "bertscore_f1":        float(F1.mean()),
        }
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"bertscore_f1": float("nan")}


# ─── Competition score ──────────────────────────────────────────────────────

def compute_competition_score(
    rouge1_f1: float,
    rougeL_f1: float,
    llm_score_1_to_5: float,
    weights: dict | None = None,
) -> float:
    """Weighted Phase 1 competition score.

    Default weights: ROUGE-1 37%, ROUGE-L 37%, LLM judge 26%.
    LLM raw score (1-5) is normalized to [0,1] via (s-1)/4 before weighting,
    matching the metrics.py spec from the project.
    """
    if weights is None:
        weights = {"rouge1_f1": 0.37, "rougeL_f1": 0.37, "llm_judge": 0.26}
    llm_norm = max(0.0, min(1.0, (llm_score_1_to_5 - 1.0) / 4.0))
    return round(
        weights["rouge1_f1"] * rouge1_f1
        + weights["rougeL_f1"] * rougeL_f1
        + weights["llm_judge"] * llm_norm,
        5,
    )


# ─── Zindi submission ──────────────────────────────────────────────────────

def make_submission(
    ids: list,
    predictions: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Zindi format: ID | TargetRLF1 | TargetR1F1 | TargetLLM
    (same predicted text in all three target columns per Phase 1 spec.)"""
    sub = pd.DataFrame({
        "ID":         ids,
        "TargetRLF1": predictions,
        "TargetR1F1": predictions,
        "TargetLLM":  predictions,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    logger.success(f"Submission saved → {output_path}  ({len(sub)} rows)")
    return sub


# ─── Main eval orchestrator ────────────────────────────────────────────────

def evaluate(cfg: dict, args) -> None:
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    paths = cfg["paths"]
    eval_cfg = cfg.get("evaluation", {})
    weights = eval_cfg.get("metric_weights", {"rouge1_f1": 0.37, "rougeL_f1": 0.37, "llm_judge": 0.26})

    workspace = Path(paths.get("workspace", "."))
    adapter_path = resolve_adapter_path(cfg, args.adapter)
    logger.info(f"Adapter: {adapter_path}")

    val_path = Path(args.val_file or cfg["dataset"]["val_file"])
    if not val_path.is_absolute():
        val_path = workspace / val_path
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")
    logger.info(f"Val file: {val_path}")

    # ── Load + clean val data ────────────────────────────────────────────────
    df = pd.read_csv(val_path)
    df = df.dropna(subset=["input", "output", "subset"]).copy()
    for c in ["input", "output", "subset"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "") & (df["subset"] != "")].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()
        logger.warning(f"--limit={args.limit} applied")
    logger.info(f"Evaluating on {len(df):,} rows across {df['subset'].nunique()} languages")

    # ── Generate predictions ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, adapter_path)
    df["prediction"] = generate_predictions(df, model, tokenizer, cfg,
                                           batch_size=args.batch_size, desc="Val inference")

    # ── ROUGE per language ───────────────────────────────────────────────────
    rouge_df = evaluate_per_language(df)
    print()
    print("=" * 72)
    print("ROUGE RESULTS")
    print("=" * 72)
    print(rouge_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    overall = rouge_df[rouge_df["language"] == "OVERALL"].iloc[0]
    rouge1 = float(overall["rouge1_f1"])
    rougeL = float(overall["rougeL_f1"])

    # ── Save outputs near the adapter ────────────────────────────────────────
    out_dir = adapter_path.parent / "final_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "predictions_full.csv", index=False, encoding="utf-8")
    rouge_df.to_csv(out_dir / "rouge_by_language.csv", index=False)
    logger.info(f"Predictions      → {out_dir / 'predictions_full.csv'}")
    logger.info(f"ROUGE summary    → {out_dir / 'rouge_by_language.csv'}")

    # ── LLM judge (optional, sampled) ────────────────────────────────────────
    judge_avg = None
    if args.llm_judge:
        # Free up VRAM from mT0 before loading judge model
        logger.info("Unloading mT0 to free VRAM for judge model …")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        sample_df = sample_for_judge(df, n_per_lang=args.judge_sample, seed=cfg.get("seed", 42))
        logger.info(f"LLM judge sample: {len(sample_df):,} rows "
                    f"({args.judge_sample}/lang × {df['subset'].nunique()} langs)")

        if args.judge_backend == "claude":
            judge_scores = judge_via_claude_api(
                sample_df["input"].tolist(),
                sample_df["prediction"].tolist(),
                sample_df["output"].tolist(),
            )
        else:
            judge_scores = judge_local_mistral(
                sample_df["input"].tolist(),
                sample_df["prediction"].tolist(),
                sample_df["output"].tolist(),
                model_name=args.judge_model,
                batch_size=args.judge_batch_size,
            )
        sample_df = sample_df.copy()
        sample_df["judge_score"] = judge_scores
        judge_avg = float(np.mean(judge_scores))

        # Per-language judge breakdown
        judge_by_lang = sample_df.groupby("subset")["judge_score"].agg(["mean", "count"]).round(3)
        print()
        print("LLM Judge per language (raw 1-5):")
        print(judge_by_lang.to_string())
        print()

        sample_df.to_csv(out_dir / "judge_scores.csv", index=False)
        logger.info(f"Judge scores     → {out_dir / 'judge_scores.csv'}")

    # ── BERTScore (optional) ─────────────────────────────────────────────────
    bs_result = None
    if args.bertscore:
        logger.info("Computing BERTScore (Afro-XLMR base) …")
        bs_result = compute_bertscore(
            df["prediction"].tolist(), df["output"].tolist(),
        )
        print(f"BERTScore F1: {bs_result.get('bertscore_f1', float('nan')):.4f}")

    # ── Competition score ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("FINAL SCORES")
    print("=" * 72)
    print(f"  ROUGE-1 F1   : {rouge1:.4f}")
    print(f"  ROUGE-L F1   : {rougeL:.4f}")
    if judge_avg is not None:
        comp = compute_competition_score(rouge1, rougeL, judge_avg, weights)
        print(f"  LLM Judge    : {judge_avg:.3f} / 5.0 (normalized: {(judge_avg-1)/4:.3f})")
        print(f"  COMPOSITE    : {comp:.4f}")
        print(f"    weights = R1×{weights['rouge1_f1']} + RL×{weights['rougeL_f1']} + LLM×{weights['llm_judge']}")
    if bs_result and not np.isnan(bs_result.get('bertscore_f1', float('nan'))):
        print(f"  BERTScore F1 : {bs_result['bertscore_f1']:.4f}  (Phase 2 metric)")
    print()

    # Save a single JSON summary
    summary = {
        "adapter_path":   str(adapter_path),
        "val_path":       str(val_path),
        "n_examples":     len(df),
        "rouge1_f1":      rouge1,
        "rougeL_f1":      rougeL,
        "per_language":   rouge_df.to_dict(orient="records"),
        "llm_judge_avg":  judge_avg,
        "competition_score": (compute_competition_score(rouge1, rougeL, judge_avg, weights)
                              if judge_avg is not None else None),
        "bertscore":      bs_result,
        "decoding":       cfg.get("decoding", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"Summary JSON     → {out_dir / 'summary.json'}")

    # ── Test set submission (optional) ──────────────────────────────────────
    if args.make_submission:
        test_path = Path(args.test_file or cfg["dataset"]["test_file"])
        if not test_path.is_absolute():
            test_path = workspace / test_path
        if not test_path.exists():
            logger.error(f"Test file not found: {test_path} — skipping submission")
        else:
            # If we unloaded the model for the judge, reload it for test inference
            if args.llm_judge:
                logger.info("Reloading mT0 for test inference …")
                model, tokenizer = load_model_and_tokenizer(cfg, adapter_path)

            logger.info(f"Generating test predictions: {test_path}")
            df_test = pd.read_csv(test_path).dropna(subset=["input", "subset"]).copy()
            for c in ["input", "subset"]:
                df_test[c] = df_test[c].astype(str).str.strip()
            df_test = df_test[(df_test["input"] != "") & (df_test["subset"] != "")].reset_index(drop=True)

            df_test["prediction"] = generate_predictions(
                df_test, model, tokenizer, cfg,
                batch_size=args.batch_size, desc="Test inference",
            )

            sub_dir = Path(paths.get("submissions", workspace / "submissions"))
            id_col = cfg["dataset"].get("id_col", "ID")
            ids = df_test[id_col].tolist() if id_col in df_test.columns else list(range(len(df_test)))
            sub_path = sub_dir / f"submission_{args.submission_tag}.csv"
            make_submission(ids, df_test["prediction"].tolist(), sub_path)
            df_test.to_csv(out_dir / f"test_predictions_{args.submission_tag}.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Full evaluation of trained mT0 adapter")
    p.add_argument("--config",   default="config_v3.yaml")
    p.add_argument("--adapter",  default=None, help="Path to LoRA adapter (default: auto)")
    p.add_argument("--val_file", default=None)
    p.add_argument("--test_file", default=None)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--limit",    type=int, default=0)

    # LLM judge
    p.add_argument("--llm_judge", action="store_true",
                   help="Enable LLM-as-a-Judge (sampled, default Mistral local)")
    p.add_argument("--judge_backend", choices=["mistral", "claude"], default="mistral")
    p.add_argument("--judge_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--judge_sample", type=int, default=100,
                   help="Rows per language to judge (default 100 = ~800 total)")
    p.add_argument("--judge_batch_size", type=int, default=4)

    # BERTScore
    p.add_argument("--bertscore", action="store_true",
                   help="Compute BERTScore (Phase 2 metric, ~30 min on full val)")

    # Submission
    p.add_argument("--make_submission", action="store_true",
                   help="Generate Zindi submission CSV from test set")
    p.add_argument("--submission_tag", default="v2")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    evaluate(config, args)