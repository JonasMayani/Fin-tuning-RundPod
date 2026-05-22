"""
src/evaluation/decode_search.py
─────────────────────────────────────────────────────────────────────────────
Phase 4.3 — Decoding Hyperparameter Optimisation
Multilingual African Health Assistant | Zindi ITU Challenge

Grid-searches decoding hyperparameters per language to maximise ROUGE-L.
Writes best config back to config.yaml under decoding.per_language.

Search space:
  num_beams        ∈ {4, 6, 8, 12}
  length_penalty   ∈ {0.6, 0.8, 1.0, 1.2}
  no_repeat_ngram  ∈ {2, 3, 4}
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
from pathlib import Path

# Add project root to path so `src` is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.train import build_prompt

# Search space
BEAM_SIZES      = [4, 6, 8, 12]
LENGTH_PENALTIES = [0.6, 0.8, 1.0, 1.2]
NO_REPEAT_NGRAMS = [2, 3, 4]


def decode_batch(
    model,
    tokenizer,
    texts:            list[str],
    max_in:           int,
    max_out:          int,
    num_beams:        int,
    length_penalty:   float,
    no_repeat_ngram:  int,
) -> list[str]:
    """Generate predictions for a batch with given decode config."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_in,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram,
            max_new_tokens=max_out,
        )
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def score_predictions(predictions: list[str], references: list[str]) -> float:
    """Return macro ROUGE-L F1."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return float(np.mean(scores))


def run_decode_search(config_path: str = "src/training/config.yaml") -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths     = cfg["paths"]
    model_cfg = cfg["model"]
    model_dir = str(Path(paths["models"]) / model_cfg["base_model"].replace("/", "_") / "best")
    val_path  = Path(paths["data_cleaned"]) / "val_clean.csv"

    logger.info(f"Loading model from {model_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    df_val  = pd.read_csv(val_path)
    max_in  = model_cfg["max_input_length"]
    max_out = model_cfg["max_output_length"]

    # Sample 100 examples per language for speed
    sample = df_val.groupby("subset").apply(
        lambda g: g.sample(min(100, len(g)), random_state=cfg["seed"])
    ).reset_index(drop=True)

    from src.training.train import build_prompt
    sample["prompt"] = sample.apply(
        lambda r: build_prompt(r["input"], r["subset"]), axis=1
    )

    best_configs: dict[str, dict] = {}

    for lang, grp in sample.groupby("subset"):
        logger.info(f"\nSearching decode config for {lang} ({len(grp)} examples) …")
        prompts = grp["prompt"].tolist()
        refs    = grp["output"].tolist()

        best_score  = -1.0
        best_config = {}

        grid = list(itertools.product(BEAM_SIZES, LENGTH_PENALTIES, NO_REPEAT_NGRAMS))
        for nb, lp, nrn in tqdm(grid, desc=lang):
            preds = decode_batch(
                model, tokenizer, prompts,
                max_in, max_out,
                num_beams=nb,
                length_penalty=lp,
                no_repeat_ngram=nrn,
            )
            score = score_predictions(preds, refs)
            if score > best_score:
                best_score  = score
                best_config = {"num_beams": nb, "length_penalty": lp, "no_repeat_ngram": nrn}

        logger.info(f"{lang} → best config: {best_config}  ROUGE-L: {best_score:.4f}")
        best_configs[lang] = best_config

    # Write back to config
    cfg["decoding"]["per_language"] = best_configs
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    logger.success(f"Optimal decode configs written to {config_path}")

    # Print summary table
    rows = [{"language": k, **v} for k, v in best_configs.items()]
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/training/config.yaml"
    run_decode_search(cfg_path)
