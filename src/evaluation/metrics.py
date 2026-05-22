"""
src/evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 — Evaluation Pipeline
Multilingual African Health Assistant | Zindi ITU Challenge

Implements:
  - ROUGE-1 F1, ROUGE-L F1  (37% + 37% = 74% of Phase 1 score)
  - LLM-as-a-Judge via instruction-tuned model  (26%)
  - AfroLM BERTScore  (Phase 2)
  - Per-language breakdowns
  - Zindi submission file generation

Submission format (Phase 1):
  ID | TargetRLF1 | TargetR1F1 | TargetLLM
  (predicted answer text goes in all three columns as per competition spec)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


# ─── ROUGE Evaluation ─────────────────────────────────────────────────────────

def compute_rouge_scores(
    predictions: list[str],
    references:  list[str],
) -> dict[str, float]:
    """Compute ROUGE-1 and ROUGE-L F1 (macro average)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_list, rl_list = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref.strip(), pred.strip())
        r1_list.append(scores["rouge1"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f1": float(np.mean(r1_list)),
        "rougeL_f1": float(np.mean(rl_list)),
        "rouge1_scores": r1_list,
        "rougeL_scores": rl_list,
    }


def evaluate_per_language(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    ref_col:  str = "output",
    subset_col: str = "subset",
) -> pd.DataFrame:
    """Compute ROUGE metrics per language and overall."""
    rows = []
    for lang, group in df.groupby(subset_col):
        preds = group[pred_col].fillna("").tolist()
        refs  = group[ref_col].fillna("").tolist()
        rouge = compute_rouge_scores(preds, refs)
        rows.append({
            "language":   lang,
            "n_examples": len(group),
            "rouge1_f1":  rouge["rouge1_f1"],
            "rougeL_f1":  rouge["rougeL_f1"],
        })

    # Overall macro average
    all_preds = df[pred_col].fillna("").tolist()
    all_refs  = df[ref_col].fillna("").tolist()
    overall   = compute_rouge_scores(all_preds, all_refs)
    rows.append({
        "language":   "OVERALL",
        "n_examples": len(df),
        "rouge1_f1":  overall["rouge1_f1"],
        "rougeL_f1":  overall["rougeL_f1"],
    })
    return pd.DataFrame(rows)


# ─── LLM-as-a-Judge ─────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for multilingual health question-answering systems.
Your task is to evaluate a model's answer against a reference answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Please rate the model answer on a scale of 1–5 for each criterion:
- accuracy: medical correctness and safety (1=dangerous/wrong, 5=correct and safe)
- completeness: how fully the question is addressed (1=missing key info, 5=complete)
- language: grammatical naturalness and fluency in the question's language (1=unreadable, 5=native-like)

Respond ONLY with a valid JSON object, no other text:
{{"accuracy": <int>, "completeness": <int>, "language": <int>, "overall": <float>}}"""


def llm_judge_score(
    questions:   list[str],
    predictions: list[str],
    references:  list[str],
    judge_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    batch_size:  int = 4,
    use_api:     bool = False,
    api_key:     Optional[str] = None,
) -> list[float]:
    """
    Score predictions using an instruction-tuned LLM judge.

    Parameters
    ----------
    use_api  : If True, uses Anthropic/OpenAI API; otherwise runs locally.
    """
    if use_api and api_key:
        return _judge_via_api(questions, predictions, references, api_key)
    return _judge_local(questions, predictions, references, judge_model, batch_size)


def _parse_judge_response(text: str) -> float:
    """Extract overall score from judge JSON response."""
    try:
        # Try to find JSON block
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if "overall" in data:
                return float(data["overall"])
            # Fallback: average the three sub-scores
            scores = [data.get(k, 3.0) for k in ("accuracy", "completeness", "language")]
            return float(np.mean(scores))
    except Exception:
        pass
    # Fallback: look for a number 1-5
    nums = re.findall(r"\b([1-5](?:\.\d+)?)\b", text)
    if nums:
        return float(nums[-1])
    return 3.0  # neutral fallback


def _judge_local(
    questions:   list[str],
    predictions: list[str],
    references:  list[str],
    model_name:  str,
    batch_size:  int,
) -> list[float]:
    """Run judge model locally (HuggingFace)."""
    from transformers import pipeline as hf_pipeline
    logger.info(f"Loading judge model: {model_name}")
    judge = hf_pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_new_tokens=128,
    )
    scores = []
    for q, pred, ref in tqdm(
        zip(questions, predictions, references), total=len(questions), desc="LLM Judge"
    ):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=q[:300], reference=ref[:400], prediction=pred[:400]
        )
        try:
            result = judge(prompt, do_sample=False)[0]["generated_text"]
            # Extract the generated part (after the prompt)
            generated = result[len(prompt):].strip()
            score = _parse_judge_response(generated)
        except Exception as e:
            logger.warning(f"Judge failed: {e}")
            score = 3.0
        scores.append(score)
    return scores


def _judge_via_api(
    questions:   list[str],
    predictions: list[str],
    references:  list[str],
    api_key:     str,
) -> list[float]:
    """Run judge via Anthropic Messages API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    scores = []
    for q, pred, ref in tqdm(
        zip(questions, predictions, references), total=len(questions), desc="LLM Judge (API)"
    ):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=q[:300], reference=ref[:400], prediction=pred[:400]
        )
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            text  = message.content[0].text
            score = _parse_judge_response(text)
        except Exception as e:
            logger.warning(f"API judge failed: {e}")
            score = 3.0
        scores.append(score)
        time.sleep(0.1)  # rate-limit safety
    return scores


# ─── BERTScore (Phase 2) ─────────────────────────────────────────────────────

def compute_bertscore(
    predictions: list[str],
    references:  list[str],
    model_type:  str = "Davlan/afro-xlmr-base",
) -> dict[str, float]:
    """Compute BERTScore using an AfroLM/AfroXLMR encoder."""
    try:
        from bert_score import score as bs_score
        P, R, F1 = bs_score(
            cands=predictions,
            refs=references,
            model_type=model_type,
            lang="multilingual",
            verbose=False,
        )
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall":    float(R.mean()),
            "bertscore_f1":        float(F1.mean()),
        }
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"bertscore_f1": 0.0}


# ─── Inference ───────────────────────────────────────────────────────────────

def generate_predictions(
    df:         pd.DataFrame,
    model_dir:  str,
    cfg:        dict,
    batch_size: int = 16,
) -> list[str]:
    """
    Run inference with the fine-tuned model on df["input"] / df["subset"].
    Uses per-language decoding configs from cfg["decoding"]["per_language"].
    """
    from src.training.train import build_prompt, LANGUAGE_NAMES

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    decode_cfgs = cfg["decoding"]["per_language"]
    default_dc  = cfg["decoding"]["default"]
    max_out     = cfg["model"]["max_output_length"]

    all_predictions: list[str] = []

    for i in tqdm(range(0, len(df), batch_size), desc="Inference"):
        batch_df = df.iloc[i : i + batch_size]
        prompts  = [
            build_prompt(row["input"], row["subset"])
            for _, row in batch_df.iterrows()
        ]
        # Use per-language decode config for the majority language in the batch
        lang = batch_df["subset"].mode()[0]
        dc   = decode_cfgs.get(lang, default_dc)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["model"]["max_input_length"],
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=dc.get("num_beams", 8),
                length_penalty=dc.get("length_penalty", 0.8),
                no_repeat_ngram_size=dc.get("no_repeat_ngram", 3),
                max_new_tokens=max_out,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])

    return all_predictions


# ─── Zindi Submission Generator ──────────────────────────────────────────────

def make_submission(
    ids:         list[str],
    predictions: list[str],
    output_path: str,
    version:     str = "v1",
) -> pd.DataFrame:
    """
    Generate Zindi submission CSV.
    Format: ID | TargetRLF1 | TargetR1F1 | TargetLLM
    (predicted answer text in all three columns)
    """
    sub = pd.DataFrame({
        "ID":          ids,
        "TargetRLF1":  predictions,
        "TargetR1F1":  predictions,
        "TargetLLM":   predictions,
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    logger.success(f"Submission saved → {output_path}  ({len(sub)} rows)")
    return sub


# ─── Weighted Competition Score ───────────────────────────────────────────────

def compute_competition_score(
    rouge1_f1: float,
    rougeL_f1: float,
    llm_score: float,
    weights:   dict | None = None,
) -> float:
    """
    Compute weighted Phase 1 competition score.
    Default weights: ROUGE-1 37%, ROUGE-L 37%, LLM-Judge 26%.
    LLM score is normalised from [1,5] → [0,1].
    """
    if weights is None:
        weights = {"rouge1_f1": 0.37, "rougeL_f1": 0.37, "llm_judge": 0.26}
    llm_normalised = (llm_score - 1.0) / 4.0
    score = (
        weights["rouge1_f1"] * rouge1_f1
        + weights["rougeL_f1"] * rougeL_f1
        + weights["llm_judge"] * llm_normalised
    )
    return round(score, 5)


# ─── Full Evaluation Runner ───────────────────────────────────────────────────

def run_evaluation(config_path: str = "src/training/config.yaml") -> None:
    """End-to-end evaluation on validation set and test set submission."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths      = cfg["paths"]
    eval_cfg   = cfg["evaluation"]
    weights    = eval_cfg["metric_weights"]

    model_dir    = str(Path(paths["models"]) / cfg["model"]["base_model"].replace("/", "_") / "best")
    val_path     = Path(paths["data_cleaned"]) / "val_clean.csv"
    test_path    = Path(paths["data_raw"])     / "Test.csv"
    reports_dir  = Path(paths["reports"])
    sub_dir      = Path(paths["submissions"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    # ── Validation set evaluation ────────────────────────────────────────────
    logger.info("Evaluating on validation set …")
    df_val = pd.read_csv(val_path)
    df_val["prediction"] = generate_predictions(df_val, model_dir, cfg)

    rouge_breakdown = evaluate_per_language(df_val)
    rouge_breakdown.to_csv(reports_dir / "evaluation_results.csv", index=False)
    logger.info("\n" + rouge_breakdown.to_string(index=False))

    overall_row = rouge_breakdown[rouge_breakdown["language"] == "OVERALL"].iloc[0]
    rouge1 = overall_row["rouge1_f1"]
    rougeL = overall_row["rougeL_f1"]

    # LLM Judge (sample 200 per language for speed)
    sample = df_val.groupby("subset").apply(
        lambda g: g.sample(min(200, len(g)), random_state=cfg["seed"])
    ).reset_index(drop=True)

    logger.info("Running LLM-as-a-Judge evaluation …")
    judge_scores = llm_judge_score(
        questions=sample["input"].tolist(),
        predictions=sample["prediction"].tolist(),
        references=sample["output"].tolist(),
        judge_model=eval_cfg["judge_model"],
    )
    avg_judge = float(np.mean(judge_scores))
    logger.info(f"LLM Judge average score: {avg_judge:.3f} / 5.0")

    comp_score = compute_competition_score(rouge1, rougeL, avg_judge, weights)
    logger.success(
        f"\n{'─'*50}\n"
        f"  ROUGE-1 F1  : {rouge1:.4f}\n"
        f"  ROUGE-L F1  : {rougeL:.4f}\n"
        f"  LLM Judge   : {avg_judge:.3f} / 5\n"
        f"  Competition : {comp_score:.4f}\n"
        f"{'─'*50}"
    )

    # ── Test set submission ──────────────────────────────────────────────────
    logger.info("Generating test set predictions …")
    df_test = pd.read_csv(test_path)
    test_preds = generate_predictions(df_test, model_dir, cfg)

    sub_path = sub_dir / "submission_v1.csv"
    make_submission(df_test["ID"].tolist(), test_preds, str(sub_path))


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/training/config.yaml"
    run_evaluation(cfg_path)
