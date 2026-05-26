#!/usr/bin/env bash
# run_pipeline.sh — end-to-end orchestration of the augmentation pipeline.
#
# Stages:
#   0. Diagnose         — analyze dataset, find noise
#   1. Clean            — drop noisy rows
#   2. Back-translate   — paraphrase questions in low-resource langs
#   3. Cross-lingual    — translate English seeds → African languages
#   4. LLM paraphrase   — same-language answer rewrites via Aya-Expanse-8B
#   5. Merge            — combine + final quality gate → final_train.csv
#
# Skip stages by setting env vars: SKIP_S2=1 SKIP_S4=1 bash run_pipeline.sh
#
# Adjust paths to your workspace at the top of this file.

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace/Fin-tuning-RundPod}"
TRAIN_IN="${TRAIN_IN:-$WORKSPACE/data/cleaned/train_clean.csv}"
REPORTS="${REPORTS:-$WORKSPACE/reports/augmentation}"
DATA_OUT="${DATA_OUT:-$WORKSPACE/data/augmented}"
FINAL_TRAIN="${FINAL_TRAIN:-$DATA_OUT/final_train.csv}"
LLM_BACKEND="${LLM_BACKEND:-aya}"  # 'aya' or 'claude'

mkdir -p "$REPORTS" "$DATA_OUT"

echo "============================================================"
echo "Augmentation pipeline"
echo "  Input train file : $TRAIN_IN"
echo "  Reports dir      : $REPORTS"
echo "  Data out dir     : $DATA_OUT"
echo "  Final train file : $FINAL_TRAIN"
echo "  LLM backend      : $LLM_BACKEND"
echo "============================================================"
echo

cd "$(dirname "$(readlink -f "$0")")/.."  # repo root, so `python -m pipeline.X` works

# ── Stage 0: diagnose ───────────────────────────────────────────────────────
if [[ "${SKIP_S0:-0}" != "1" ]]; then
  echo "── Stage 0: diagnose ─────────────────────────────────────────"
  python -m pipeline.stage0_diagnose \
    --train "$TRAIN_IN" \
    --out   "$REPORTS/stage0"
  echo
else
  echo "Skipping Stage 0"
fi

# ── Stage 1: clean ──────────────────────────────────────────────────────────
CLEAN_OUT="$DATA_OUT/train_clean_v2.csv"
if [[ "${SKIP_S1:-0}" != "1" ]]; then
  echo "── Stage 1: clean ────────────────────────────────────────────"
  python -m pipeline.stage1_clean \
    --diag   "$REPORTS/stage0/diagnostic_per_row.csv" \
    --emb    "$REPORTS/stage0/embeddings.npz" \
    --config pipeline/clean_config.yaml \
    --out    "$CLEAN_OUT" \
    --report_dir "$REPORTS/stage1"
  echo
else
  echo "Skipping Stage 1 (expecting $CLEAN_OUT to exist)"
fi

# ── Stage 2: back-translation ───────────────────────────────────────────────
S2_OUT="$DATA_OUT/stage2_backtrans.csv"
if [[ "${SKIP_S2:-0}" != "1" ]]; then
  echo "── Stage 2: back-translation ────────────────────────────────"
  python -m pipeline.stage2_backtranslate \
    --in     "$CLEAN_OUT" \
    --out    "$S2_OUT" \
    --config pipeline/aug_config.yaml
  echo
else
  echo "Skipping Stage 2"
fi

# ── Stage 3: cross-lingual transfer ─────────────────────────────────────────
S3_OUT="$DATA_OUT/stage3_crosslingual.csv"
if [[ "${SKIP_S3:-0}" != "1" ]]; then
  echo "── Stage 3: cross-lingual transfer ──────────────────────────"
  python -m pipeline.stage3_crosslingual \
    --in     "$CLEAN_OUT" \
    --diag   "$REPORTS/stage0/diagnostic_per_row.csv" \
    --out    "$S3_OUT" \
    --config pipeline/aug_config.yaml
  echo
else
  echo "Skipping Stage 3"
fi

# ── Stage 4: LLM paraphrase ─────────────────────────────────────────────────
S4_OUT="$DATA_OUT/stage4_paraphrase.csv"
if [[ "${SKIP_S4:-0}" != "1" ]]; then
  echo "── Stage 4: LLM paraphrase (backend=$LLM_BACKEND) ────────────"
  python -m pipeline.stage4_llm_paraphrase \
    --in     "$CLEAN_OUT" \
    --diag   "$REPORTS/stage0/diagnostic_per_row.csv" \
    --out    "$S4_OUT" \
    --backend "$LLM_BACKEND" \
    --config pipeline/aug_config.yaml
  echo
else
  echo "Skipping Stage 4"
fi

# ── Stage 5: merge ──────────────────────────────────────────────────────────
echo "── Stage 5: merge + final quality gate ──────────────────────"
python -m pipeline.stage5_merge \
  --clean  "$CLEAN_OUT" \
  --aug    "$S2_OUT" "$S3_OUT" "$S4_OUT" \
  --out    "$FINAL_TRAIN"

echo
echo "============================================================"
echo "Pipeline complete."
echo "  Final training set: $FINAL_TRAIN"
echo "  Reports dir:        $REPORTS"
echo "============================================================"
echo
echo "Next step: update config_mt0.yaml to point at the new data:"
echo "    dataset:"
echo "      train_file: data/augmented/final_train.csv"
echo "Then retrain with: python train_mt0.py --config config_mt0.yaml"
