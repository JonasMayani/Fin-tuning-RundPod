#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train.sh — Reproduce full training pipeline from scratch
# Multilingual African Health Assistant | Zindi ITU Challenge
#
# Usage:
#   ./src/training/train.sh                          # default config
#   ./src/training/train.sh --model google/mt5-large  # override model
#
# Prerequisites:
#   pip install -r requirements.txt
#   Place Train.csv, Val.csv, Test.csv in data/raw/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG="src/training/config.yaml"
BASE_MODEL=""
SKIP_CLEAN=false
SKIP_AUGMENT=false
SKIP_GATE=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)    CONFIG="$2";     shift 2 ;;
        --model)     BASE_MODEL="$2"; shift 2 ;;
        --skip-clean)   SKIP_CLEAN=true;   shift ;;
        --skip-augment) SKIP_AUGMENT=true; shift ;;
        --skip-gate)    SKIP_GATE=true;    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Multilingual African Health Assistant — Training Pipeline  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Config     : $CONFIG"
echo "Start time : $(date)"
echo ""

# ── Phase 1.2: Data Cleaning ─────────────────────────────────────────────────
if [ "$SKIP_CLEAN" = false ]; then
    echo "━━━ Phase 1.2 — Data Cleaning ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python src/data/clean.py "$CONFIG"
    echo ""
fi

# ── Phase 2: Augmentation ─────────────────────────────────────────────────────
if [ "$SKIP_AUGMENT" = false ]; then
    echo "━━━ Phase 2 — Data Augmentation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python src/data/augment.py "$CONFIG"
    echo ""
fi

# ── Phase 2.3: Quality Gate ───────────────────────────────────────────────────
if [ "$SKIP_GATE" = false ]; then
    echo "━━━ Phase 2.3 — Quality Gate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python src/data/quality_gate.py "$CONFIG"
    echo ""
fi

# ── Phase 3: Training ─────────────────────────────────────────────────────────
echo "━━━ Phase 3 — Model Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "$BASE_MODEL" ]; then
    python src/training/train.py --config "$CONFIG" --base_model "$BASE_MODEL"
else
    python src/training/train.py --config "$CONFIG"
fi
echo ""

# ── Phase 4.3: Decoding Search ────────────────────────────────────────────────
echo "━━━ Phase 4.3 — Decoding Hyperparameter Search ━━━━━━━━━━━━━━━━━"
python src/evaluation/decode_search.py "$CONFIG"
echo ""

# ── Phase 4: Evaluation & Submission ─────────────────────────────────────────
echo "━━━ Phase 4 — Evaluation & Submission Generation ━━━━━━━━━━━━━━━"
python src/evaluation/metrics.py "$CONFIG"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Training pipeline complete!                                ║"
echo "║   Check reports/evaluation_results.csv for metrics.         ║"
echo "║   Submission file is in submissions/submission_v1.csv        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "End time: $(date)"
