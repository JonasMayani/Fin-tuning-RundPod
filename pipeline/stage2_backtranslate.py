"""
pipeline/stage2_backtranslate.py — paraphrase questions via translation pivot.

Strategy:
  For each row with native language L:
    Q_L → translate to English → translate back to L
  The resulting Q'_L is a paraphrase: same meaning, different wording.
  The ANSWER STAYS UNCHANGED — we're only varying the input distribution.

Why this helps:
  - Trains the model to handle phrasing variation for the same intent.
  - Cheap: NLLB-200-3.3B you already have in your config.
  - Doesn't introduce factual drift (answers untouched).

We do NOT back-translate for English subsets (would be useless).
We DO back-translate for African languages, especially low-resource ones.

Run: python -m pipeline.stage2_backtranslate \
        --in  data/cleaned/train_clean_v2.csv \
        --out data/augmented/stage2_backtrans.csv \
        --config pipeline/aug_config.yaml
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from pipeline.utils import chrf, clean_whitespace, script_ok


# NLLB language codes for our subsets. We pivot via English.
NLLB_CODE = {
    "Eng_Uga": "eng_Latn", "Eng_Gha": "eng_Latn", "Eng_Eth": "eng_Latn", "Eng_Ken": "eng_Latn",
    "Aka_Gha": "aka_Latn",
    "Lug_Uga": "lug_Latn",
    "Swa_Ken": "swh_Latn",
    "Amh_Eth": "amh_Ethi",
}

# Default per-language: how many rows to back-translate. Bias towards
# under-represented languages where augmentation is most valuable.
DEFAULT_AUG_TARGETS = {
    "Eng_Uga": 0,       # English already large
    "Eng_Gha": 0,
    "Eng_Eth": 0,
    "Eng_Ken": 0,
    "Aka_Gha": 600,
    "Lug_Uga": 800,
    "Swa_Ken": 800,
    "Amh_Eth": 400,     # cap lower — NLLB Amharic quality is moderate
}


class NLLBTranslator:
    """Thin wrapper around NLLB-200 for paired-direction translation."""

    def __init__(self, model_name: str = "facebook/nllb-200-3.3B",
                 dtype: torch.dtype = torch.bfloat16,
                 batch_size: int = 16,
                 max_length: int = 384,
                 num_beams: int = 4):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        logger.info(f"Loading NLLB: {model_name}  dtype={dtype}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).cuda()
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams

    @torch.inference_mode()
    def translate(self, texts: list[str], src_code: str, tgt_code: str) -> list[str]:
        """Translate a list of texts from src_code → tgt_code."""
        if not texts:
            return []
        self.tok.src_lang = src_code
        outputs: list[str] = []
        # NLLB uses a forced BOS token to select the target language.
        forced_bos = self.tok.convert_tokens_to_ids(tgt_code)
        for start in tqdm(range(0, len(texts), self.batch_size),
                          desc=f"{src_code}→{tgt_code}", leave=False):
            batch = texts[start:start + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=self.max_length).to("cuda")
            try:
                gen = self.model.generate(
                    **enc, forced_bos_token_id=forced_bos,
                    max_new_tokens=self.max_length, num_beams=self.num_beams,
                )
                outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"NLLB OOM at batch_size={len(batch)}, retrying one-by-one")
                torch.cuda.empty_cache()
                for t in batch:
                    enc_1 = self.tok([t], return_tensors="pt", truncation=True,
                                    max_length=self.max_length).to("cuda")
                    gen = self.model.generate(
                        **enc_1, forced_bos_token_id=forced_bos,
                        max_new_tokens=self.max_length, num_beams=self.num_beams,
                    )
                    outputs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outputs

    def close(self) -> None:
        del self.model
        del self.tok
        gc.collect()
        torch.cuda.empty_cache()


def back_translate_language(
    df_lang: pd.DataFrame,
    language: str,
    translator: NLLBTranslator,
    n_target: int,
    chrf_keep_range: tuple[float, float] = (0.35, 0.85),
) -> pd.DataFrame:
    """Back-translate up to n_target rows for one language.
    Returns a DataFrame of NEW augmented rows (with paraphrased questions)."""
    if n_target <= 0 or len(df_lang) == 0:
        return pd.DataFrame(columns=df_lang.columns.tolist() + ["aug_method", "chrf_qq"])
    nllb_lang = NLLB_CODE.get(language)
    if not nllb_lang or nllb_lang == "eng_Latn":
        logger.info(f"[{language}] skipping back-translation (English source)")
        return pd.DataFrame(columns=df_lang.columns.tolist() + ["aug_method", "chrf_qq"])

    # Sample rows to translate — prefer rows with cleaner, medium-length questions
    sample = df_lang.copy()
    sample["q_len"] = sample["input"].str.split().str.len()
    sample = sample[(sample["q_len"] >= 5) & (sample["q_len"] <= 80)]
    if len(sample) == 0:
        return pd.DataFrame(columns=df_lang.columns.tolist() + ["aug_method", "chrf_qq"])
    sample = sample.sample(n=min(n_target * 2, len(sample)),  # 2× over-sample for filter loss
                          random_state=42).reset_index(drop=True)
    logger.info(f"[{language}] back-translating {len(sample)} rows ({nllb_lang} → eng_Latn → {nllb_lang})")

    questions = sample["input"].tolist()

    # Step 1: L → English
    eng = translator.translate(questions, src_code=nllb_lang, tgt_code="eng_Latn")
    # Step 2: English → L
    paraphrased = translator.translate(eng, src_code="eng_Latn", tgt_code=nllb_lang)

    # Compute chrF between original and paraphrased questions
    chrf_scores = [chrf(orig, para) for orig, para in zip(questions, paraphrased)]

    out = sample.copy()
    out["input"] = [clean_whitespace(p) for p in paraphrased]
    out["chrf_qq"] = chrf_scores
    out["aug_method"] = "backtranslation"

    # Quality filter: paraphrase must be different enough but not gibberish
    lo, hi = chrf_keep_range
    kept = out[
        (out["chrf_qq"] >= lo) & (out["chrf_qq"] <= hi)
        & out["input"].str.len().between(5, 500)
        & out["input"].apply(lambda t: script_ok(t, language))
    ].copy()
    logger.info(f"[{language}] kept {len(kept)}/{len(out)} after chrF + script filter")

    # Trim to target
    if len(kept) > n_target:
        kept = kept.sample(n=n_target, random_state=42)

    return kept.drop(columns=["q_len"], errors="ignore")[
        ["input", "output", "subset", "aug_method", "chrf_qq"]
    ]


def run(
    input_csv: Path,
    out_csv: Path,
    targets: dict[str, int],
    nllb_model: str = "facebook/nllb-200-3.3B",
    batch_size: int = 16,
    num_beams: int = 4,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv).dropna(subset=["input", "output", "subset"]).copy()
    logger.info(f"Loaded {len(df):,} rows from {input_csv}")

    translator = NLLBTranslator(
        model_name=nllb_model, batch_size=batch_size, num_beams=num_beams,
    )

    all_aug: list[pd.DataFrame] = []
    try:
        for lang, n_target in targets.items():
            if n_target <= 0:
                continue
            df_lang = df[df["subset"] == lang]
            if len(df_lang) == 0:
                logger.warning(f"[{lang}] no rows in input — skipping")
                continue
            aug = back_translate_language(df_lang, lang, translator, n_target)
            if len(aug):
                all_aug.append(aug)
    finally:
        translator.close()

    if not all_aug:
        logger.warning("No back-translation outputs produced.")
        return pd.DataFrame()

    aug_df = pd.concat(all_aug, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aug_df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(aug_df):,} back-translation rows → {out_csv}")
    print("\nPer-language counts:")
    print(aug_df["subset"].value_counts().sort_index().to_string())
    return aug_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in",     dest="input", required=True, help="Cleaned train CSV from stage 1")
    p.add_argument("--out",    required=True, help="Output augmented CSV")
    p.add_argument("--config", default=None, help="aug_config.yaml (overrides per-lang targets)")
    p.add_argument("--nllb_model", default="facebook/nllb-200-3.3B")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_beams",  type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = dict(DEFAULT_AUG_TARGETS)
    if args.config and Path(args.config).exists():
        cfg = yaml.safe_load(open(args.config)) or {}
        if "backtranslation_targets" in cfg:
            targets.update(cfg["backtranslation_targets"])
    run(Path(args.input), Path(args.out), targets,
        nllb_model=args.nllb_model, batch_size=args.batch_size,
        num_beams=args.num_beams)
