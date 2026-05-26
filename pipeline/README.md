# Multilingual Health QA — Data Augmentation Pipeline

A 5-stage pipeline to clean and expand the training set for the African Health QA project.
Designed to fix specific issues identified in the first ROUGE evaluation:

| Language | First-eval ROUGE-L | Suspected issue | Pipeline response |
|---|---:|---|---|
| Amh_Eth | 0.036 | Wrong script / fundamental noise | Aggressive cleaning + cross-lingual transfer from English |
| Lug_Uga | 0.115 | Data scarcity | Back-translation + cross-lingual transfer |
| Swa_Ken | 0.124 | Noisy training data | Cleaning + back-translation |
| Eng_Uga | 0.123 | Label noise (English shouldn't score this low) | Strict coherence filter + LLM paraphrase of clean subset |
| Eng_Ken | 0.139 | Same as Eng_Uga | Same |
| Eng_Eth | 0.168 | Moderate noise | Coherence filter + LLM paraphrase |
| Aka_Gha | 0.205 | Reasonable baseline | Light back-translation + paraphrase |
| Eng_Gha | 0.246 | Cleanest partition | Used as **seed pool** for cross-lingual transfer |

## Pipeline architecture

```
Stage 0 — Diagnose
        ├─ Per-language stats
        ├─ Script + language ID checks (#1 noise detector for Amharic)
        ├─ Q↔A LaBSE semantic similarity → bottom-quartile review file
        └─ Near-dup rate sampling
        Output: reports/stage0/{diagnostic_per_row.csv, summary.json, embeddings.npz}

Stage 1 — Clean
        Drop rows that fail:
        ├─ Length sanity (Q ≥ 3 tokens, A ≥ 5 tokens, A ≤ 400 tokens)
        ├─ Wrong script (Amharic with Latin chars, etc.)
        ├─ Language ID mismatch (fastText disagrees with subset label)
        ├─ Low Q↔A coherence (per-language LaBSE cosine threshold)
        ├─ High repetition (degenerate text)
        └─ Near-duplicates within language (MinHash Jaccard ≥ 0.85)
        Output: data/augmented/train_clean_v2.csv

Stage 2 — Back-translation (paraphrase questions)
        For each row in target language L:
            Q_L → English → L (via NLLB-200-3.3B)
            Keep answer unchanged.
        Filter: chrF(orig, paraphrase) in [0.35, 0.85] (must differ but not garbage)
        Output: data/augmented/stage2_backtrans.csv

Stage 3 — Cross-lingual transfer (translate English → African)
        Take cleanest English (Eng_Gha) seeds.
        Translate Q AND A to each target language.
        Filter: script + langid + length + semantic preservation (LaBSE)
        Output: data/augmented/stage3_crosslingual.csv

Stage 4 — LLM paraphrase (rewrite answers in same language)
        Backend: Aya-Expanse-8B locally OR Anthropic API.
        Filter: chrF(orig, rewrite) in [0.20, 0.80], LaBSE cos ≥ 0.65
        Output: data/augmented/stage4_paraphrase.csv

Stage 5 — Merge + final quality gate
        Combine originals + augmentations.
        Dedup augmentations against originals.
        Optional per-language caps.
        Output: data/augmented/final_train.csv  ← this is what you train on
```

## Setup

```bash
# Install pipeline dependencies (on top of your training requirements)
pip install -r pipeline/requirements-pipeline.txt
```

## Running

**Full pipeline:**
```bash
bash pipeline/run_pipeline.sh
```

**Skip selected stages** (e.g. you already ran Stage 0):
```bash
SKIP_S0=1 bash pipeline/run_pipeline.sh
```

**Use Anthropic API for LLM paraphrase** (better quality, costs money):
```bash
export ANTHROPIC_API_KEY=...
LLM_BACKEND=claude bash pipeline/run_pipeline.sh
```

**Custom paths:**
```bash
TRAIN_IN=data/cleaned/train_clean.csv \
DATA_OUT=data/augmented \
REPORTS=reports/augmentation \
bash pipeline/run_pipeline.sh
```

## Run individual stages

```bash
# Stage 0 only — find out what's broken
python -m pipeline.stage0_diagnose --train data/cleaned/train_clean.csv --out reports/stage0/

# Read reports/stage0/diagnostic_summary.csv and reports/stage0/suspicious_rows_for_review.csv.
# Adjust pipeline/clean_config.yaml if needed.

# Stage 1 only — apply cleaning
python -m pipeline.stage1_clean \
    --diag reports/stage0/diagnostic_per_row.csv \
    --config pipeline/clean_config.yaml \
    --out data/augmented/train_clean_v2.csv

# etc.
```

## Expected outputs

| Stage | Time on A100 80GB | Output rows |
|---|---:|---:|
| 0 (diagnose) | ~10 min | — |
| 1 (clean) | ~1 min | 18,000–22,000 (down from 29,307) |
| 2 (back-trans) | ~30 min | 4,000–4,500 |
| 3 (cross-lingual) | ~25 min | 3,000–3,500 |
| 4 (LLM paraphrase, Aya) | ~90 min | 2,500–3,500 |
| 5 (merge) | ~2 min | 28,000–32,000 |

Total wall time: roughly **2.5–3 hours**.

## Configuration knobs

- `pipeline/clean_config.yaml` — cleaning thresholds (how strict the filters are)
- `pipeline/aug_config.yaml` — augmentation volume targets per language
- `pipeline/run_pipeline.sh` — paths, backend choice, skip flags

## After running

Edit your training config:

```yaml
# config_mt0.yaml
dataset:
  train_file: data/augmented/final_train.csv     # ← was data/cleaned/train_clean.csv
```

Then retrain.

## Validation suggestion

After Stage 5, run a small sanity check:
```bash
python eval_full.py --config config_mt0.yaml --val_file data/augmented/final_train.csv --limit 200
```
This evaluates your CURRENT trained model on a sample of the AUGMENTED data.
If ROUGE-L is reasonable on the augmented data, you've added rows that are in-distribution.
If it crashes (very low ROUGE), some of the augmentation may be too far from the val distribution.
