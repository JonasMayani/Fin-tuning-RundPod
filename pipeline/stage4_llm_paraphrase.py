"""
pipeline/stage4_llm_paraphrase.py — LLM-based same-language paraphrasing.

For each high-quality row, ask an LLM to rewrite the ANSWER in the same
language, preserving meaning and medical accuracy but varying surface form.
This adds the lexical/syntactic diversity that helps ROUGE F1 (which is
brittle to phrasing).

Backends:
  --backend aya     : Aya-Expanse-8B locally (CohereLabs/aya-expanse-8b)
                      Decoder-only, 8B, supports 23+ languages including all
                      of yours. Free, runs on your A100.
  --backend claude  : Anthropic API. Best quality, costs ~$5-15 for our scale.
                      Requires ANTHROPIC_API_KEY env var.

We rewrite ANSWERS only (not questions), because varying the answer's surface
form while keeping the question constant directly targets ROUGE F1's brittleness.

Run: python -m pipeline.stage4_llm_paraphrase \
        --in   data/cleaned/train_clean_v2.csv \
        --diag reports/stage0/diagnostic_per_row.csv \
        --out  data/augmented/stage4_paraphrase.csv \
        --backend aya
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from pipeline.utils import (
    chrf, clean_whitespace, cosine_sim, embed_texts,
    repetition_ratio, script_ok, token_count,
)


LANGUAGE_NAMES = {
    "Eng_Uga": "English", "Aka_Gha": "Akan", "Eng_Gha": "English",
    "Eng_Eth": "English", "Lug_Uga": "Luganda", "Eng_Ken": "English",
    "Swa_Ken": "Swahili", "Amh_Eth": "Amharic",
}

DEFAULT_PARAPHRASE_TARGETS = {
    "Eng_Uga": 500, "Eng_Gha": 500, "Eng_Eth": 300, "Eng_Ken": 300,
    "Aka_Gha": 400, "Lug_Uga": 400, "Swa_Ken": 400, "Amh_Eth": 200,
}


def build_paraphrase_prompt(question: str, answer: str, language: str) -> str:
    lang_name = LANGUAGE_NAMES.get(language, language)
    return (
        f"You are a medical communication assistant. Rewrite the following answer "
        f"in {lang_name}, preserving the medical meaning and accuracy exactly, "
        f"but varying word choice and sentence structure. Do not add new facts, "
        f"do not remove information, do not add disclaimers, and do not translate "
        f"to another language. Reply with only the rewritten answer.\n\n"
        f"Question: {question}\n"
        f"Original answer: {answer}\n"
        f"Rewritten answer in {lang_name}:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Backend 1: Aya-Expanse-8B (local)
# ──────────────────────────────────────────────────────────────────────────────

class AyaBackend:
    """Aya-Expanse-8B: Cohere's instruction-tuned multilingual decoder model.
    23+ languages with native quality, runs in ~16 GB bf16 on A100."""

    def __init__(self, model_name: str = "CohereLabs/aya-expanse-8b",
                 dtype: torch.dtype = torch.bfloat16,
                 max_new_tokens: int = 400):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading {model_name}  dtype={dtype}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map="cuda",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # Aya uses a chat template
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.inference_mode()
    def paraphrase_batch(self, prompts: list[str]) -> list[str]:
        """Batched generation. Returns the model's reply (the rewritten answer)."""
        # Apply chat template per prompt
        chat_prompts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = self.tok(chat_prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to("cuda")
        try:
            gen = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self.tok.pad_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("Aya OOM, retrying one-by-one")
            results = []
            for cp in chat_prompts:
                enc1 = self.tok(cp, return_tensors="pt", truncation=True,
                                max_length=1024).to("cuda")
                gen = self.model.generate(
                    **enc1, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=self.tok.pad_token_id,
                )
                # Strip the prompt portion
                new = gen[0][enc1["input_ids"].shape[1]:]
                results.append(clean_whitespace(self.tok.decode(new, skip_special_tokens=True)))
            return results

        # Strip the prompt from each output
        outputs = []
        for i in range(gen.size(0)):
            new_tokens = gen[i][enc["input_ids"].shape[1]:]
            text = self.tok.decode(new_tokens, skip_special_tokens=True)
            outputs.append(clean_whitespace(text))
        return outputs

    def close(self) -> None:
        del self.model, self.tok
        import gc; gc.collect(); torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Backend 2: Anthropic API (Claude)
# ──────────────────────────────────────────────────────────────────────────────

class ClaudeBackend:
    """Anthropic API backend. Higher quality but costs money."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY env var not set")
        self.client = anthropic.Anthropic()
        self.model = model

    def paraphrase_batch(self, prompts: list[str]) -> list[str]:
        """Anthropic API is request-at-a-time; we still call it 'batch' for
        a uniform interface."""
        results = []
        for p in prompts:
            for attempt in range(3):
                try:
                    msg = self.client.messages.create(
                        model=self.model,
                        max_tokens=600,
                        messages=[{"role": "user", "content": p}],
                    )
                    text = msg.content[0].text if msg.content else ""
                    results.append(clean_whitespace(text))
                    break
                except Exception as exc:
                    if attempt == 2:
                        logger.warning(f"Claude failed after 3 retries: {exc}")
                        results.append("")
                    else:
                        time.sleep(2 ** attempt)
        return results

    def close(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline driver
# ──────────────────────────────────────────────────────────────────────────────

def select_paraphrase_candidates(
    diag_csv: Path,
    n_per_lang: dict[str, int],
    min_qa_cos: float = 0.45,
) -> pd.DataFrame:
    """Pick high-quality rows for LLM paraphrasing — these should be clean
    examples where the LLM has a fair chance of producing a good rewrite."""
    df = pd.read_csv(diag_csv)
    df = df[
        df["q_script_ok"].astype(bool)
        & df["a_script_ok"].astype(bool)
        & (df["qa_cos"] >= min_qa_cos)
        & (df["a_tokens"] >= 10) & (df["a_tokens"] <= 250)
        & (df["a_repetition"] < 0.25)
    ].copy()

    selected = []
    for lang, n in n_per_lang.items():
        if n <= 0:
            continue
        sub = df[df["subset"] == lang]
        if len(sub) == 0:
            continue
        # Take 2× over-sample (filters will drop some)
        take = sub.sort_values("qa_cos", ascending=False).head(n * 2)
        selected.append(take)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def quality_filter_paraphrases(
    df_orig: pd.DataFrame,
    paraphrases: list[str],
    chrf_keep_range: tuple[float, float] = (0.20, 0.80),
    semantic_floor: float = 0.65,
) -> pd.DataFrame:
    """Filter LLM outputs:
       - Must be in the same script/language as the original.
       - chrF vs original answer in [0.20, 0.80] (different but related).
       - Length not catastrophically off.
       - LaBSE semantic similarity vs original ≥ floor (preserves meaning).
    """
    n_orig = len(df_orig)
    assert len(paraphrases) == n_orig

    out = df_orig.copy().reset_index(drop=True)
    out["paraphrase"] = [clean_whitespace(p) for p in paraphrases]
    out["chrf_aa"] = [chrf(o, p) for o, p in zip(out["output"], out["paraphrase"])]

    # Compute semantic similarity between original and paraphrased answer
    if len(out):
        logger.info("Checking semantic preservation of paraphrases (LaBSE) …")
        orig_emb = embed_texts(out["output"].tolist(), batch_size=128)
        para_emb = embed_texts(out["paraphrase"].tolist(), batch_size=128)
        out["semantic_preserved"] = cosine_sim(orig_emb, para_emb)
    else:
        out["semantic_preserved"] = 0.0

    out["p_tokens"]   = out["paraphrase"].apply(token_count)
    out["p_rep"]      = out["paraphrase"].apply(repetition_ratio)
    out["p_script_ok"]= [script_ok(p, s) for p, s in zip(out["paraphrase"], out["subset"])]

    lo, hi = chrf_keep_range
    keep = (
        (out["paraphrase"].str.len() >= 10)
        & out["p_script_ok"]
        & (out["chrf_aa"] >= lo) & (out["chrf_aa"] <= hi)
        & (out["semantic_preserved"] >= semantic_floor)
        & out["p_tokens"].between(5, 350)
        & (out["p_rep"] < 0.3)
    )
    kept = out[keep].copy()

    # Promote the paraphrase to be the new answer in the output row
    kept_out = pd.DataFrame({
        "input":              kept["input"].values,
        "output":             kept["paraphrase"].values,
        "subset":             kept["subset"].values,
        "aug_method":         "llm_paraphrase",
        "chrf_aa":            kept["chrf_aa"].values,
        "semantic_preserved": kept["semantic_preserved"].values,
    })
    logger.info(f"LLM paraphrase quality gate: {len(kept_out)}/{n_orig} kept")
    return kept_out


def run(
    input_csv: Path,
    diag_csv: Path,
    out_csv: Path,
    targets: dict[str, int],
    backend_name: str,
    batch_size: int,
) -> pd.DataFrame:
    candidates = select_paraphrase_candidates(diag_csv, targets)
    if len(candidates) == 0:
        logger.warning("No paraphrase candidates after filtering.")
        return pd.DataFrame()
    logger.info(f"Selected {len(candidates):,} candidate rows across {candidates['subset'].nunique()} langs")

    # Build prompts
    prompts = [
        build_paraphrase_prompt(q, a, s)
        for q, a, s in zip(candidates["input"], candidates["output"], candidates["subset"])
    ]

    # Choose backend
    if backend_name == "aya":
        backend = AyaBackend()
    elif backend_name == "claude":
        backend = ClaudeBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    paraphrases: list[str] = []
    try:
        for start in tqdm(range(0, len(prompts), batch_size),
                          desc=f"Paraphrasing ({backend_name})"):
            chunk = prompts[start:start + batch_size]
            paraphrases.extend(backend.paraphrase_batch(chunk))
    finally:
        backend.close()

    kept = quality_filter_paraphrases(candidates, paraphrases)

    # Truncate per-language to requested target
    final_pieces = []
    for lang, n_target in targets.items():
        if n_target <= 0:
            continue
        sub = kept[kept["subset"] == lang]
        if len(sub) > n_target:
            sub = sub.sort_values("semantic_preserved", ascending=False).head(n_target)
        final_pieces.append(sub)
    final = pd.concat(final_pieces, ignore_index=True) if final_pieces else pd.DataFrame()

    if len(final) == 0:
        logger.warning("Nothing survived the quality gate.")
        return pd.DataFrame()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(final):,} LLM-paraphrased rows → {out_csv}")
    print("\nPer-language counts:")
    print(final["subset"].value_counts().sort_index().to_string())
    return final


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in",   dest="input", required=True)
    p.add_argument("--diag", required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--backend", choices=["aya", "claude"], default="aya")
    p.add_argument("--batch_size", type=int, default=8,
                   help="For local: GPU batch size. For API: parallel reqs (kept at 1 to avoid rate limits).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = dict(DEFAULT_PARAPHRASE_TARGETS)
    if args.config and Path(args.config).exists():
        cfg = yaml.safe_load(open(args.config)) or {}
        if "paraphrase_targets" in cfg:
            targets.update(cfg["paraphrase_targets"])
    run(Path(args.input), Path(args.diag), Path(args.out),
        targets, args.backend, args.batch_size)
