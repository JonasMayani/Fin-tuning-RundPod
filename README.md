# 🌍 Multilingual African Health Assistant

**Zindi ITU Challenge — Multilingual Health Question Answering in Low-Resource African Languages**

A production-grade multilingual health QA system serving sub-Saharan communities in **Akan (Twi) · Amharic · Luganda · Swahili · English**.

---

## Overview

This system accepts a health question in any of the five supported African languages and generates a medically accurate, culturally appropriate answer **in the same language**. It is designed for deployment in low-bandwidth, resource-constrained environments (mobile, SMS, offline-capable).

### Supported Languages

| Tag | Language | Region |
|---|---|---|
| `Aka_Gha` | Akan (Twi) | Ghana |
| `Amh_Eth` | Amharic | Ethiopia |
| `Lug_Uga` | Luganda | Uganda |
| `Swa_Eas` | Swahili | East Africa |
| `Eng` | English | (pivot / baseline) |

### Domain

All content covers **Maternal, Sexual & Reproductive Health (MSRH)** — prenatal care, family planning, GBV support, postnatal care, and adolescent health.

---

## Repository Structure

```
african-health-assistant/
├── data/
│   ├── raw/                    # Train.csv, Val.csv, Test.csv (not tracked)
│   ├── cleaned/                # train_clean.csv, val_clean.csv
│   ├── augmented/              # final_train.csv
│   └── external/               # external_sources_register.csv
├── notebooks/
│   └── 01_eda.py               # Phase 1.1 EDA (run as notebook or script)
├── src/
│   ├── data/
│   │   ├── clean.py            # Phase 1.2 — data cleaning
│   │   ├── augment.py          # Phase 2   — augmentation pipeline
│   │   └── quality_gate.py     # Phase 2.3 — quality gate filters
│   ├── training/
│   │   ├── train.py            # Phase 3   — fine-tuning
│   │   ├── config.yaml         # All hyperparameters & paths
│   │   └── train.sh            # One-command pipeline runner
│   ├── evaluation/
│   │   ├── metrics.py          # Phase 4   — ROUGE + LLM Judge + submission
│   │   └── decode_search.py    # Phase 4.3 — decoding hyperparameter search
│   └── serving/
│       ├── api.py              # Phase 5.2 — FastAPI inference server
│       ├── Dockerfile          # Container image
│       └── docker-compose.yml  # Full stack (API + Prometheus + Grafana)
├── models/checkpoints/         # Saved model weights (not tracked in git)
├── submissions/                # Zindi submission CSV files
├── reports/
│   ├── eda_report.html
│   ├── model_card.md
│   ├── quality_gate_report.csv
│   └── evaluation_results.csv
├── governance/
│   ├── oversight_committee_charter.md
│   ├── data_governance_policy.md
│   └── flagged_answers_log.csv
├── requirements.txt
├── LICENSE                     # Apache 2.0
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw data

```
data/raw/Train.csv
data/raw/Val.csv
data/raw/Test.csv
```

### 3. Run the full pipeline (one command)

```bash
./src/training/train.sh
```

This runs: clean → augment → quality gate → train → decode search → evaluate → generate submission.

### 4. Run individual phases

```bash
# Phase 1.1: EDA
python notebooks/01_eda.py

# Phase 1.2: Cleaning
python src/data/clean.py

# Phase 2: Augmentation
python src/data/augment.py

# Phase 2.3: Quality gate
python src/data/quality_gate.py

# Phase 3: Training (override model)
python src/training/train.py --base_model google/mt5-large

# Phase 4.3: Decoding search
python src/evaluation/decode_search.py

# Phase 4: Evaluation + submission
python src/evaluation/metrics.py
```

### 5. Start the API server

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose -f src/serving/docker-compose.yml up -d
```

### 6. Query the API

```bash
# Health check
curl http://localhost:8000/health

# Ask a question (auto-detects language)
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Ni muhimu vipi kufanya mazoezi wakati wa ujauzito?"}'

# Ask with explicit language
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "How important is exercise during pregnancy?", "language": "Eng"}'
```

---

## Supported Base Models

Configure `base_model` in `src/training/config.yaml` or pass `--base_model` flag:

| Model | Params | Coverage | Licence | Notes |
|---|---|---|---|---|
| `castorini/afriteva-v2` | 300M | African-first | Apache 2.0 | **Default — recommended** |
| `google/mt5-base` | 580M | 101 languages | Apache 2.0 | Strong multilingual baseline |
| `google/mt5-large` | 1.2B | 101 languages | Apache 2.0 | Better quality, needs GPU |
| `facebook/nllb-200-1.3B` | 1.3B | 200 languages | CC-BY-NC | Best language coverage |
| `bigscience/bloom-1b7` | 1.7B | 46 languages | RAIL | Includes Swahili |

---

## Success Criteria

| Metric | Minimum | Target |
|---|---|---|
| ROUGE-1 F1 (macro avg) | 0.40 | 0.55+ |
| ROUGE-L F1 (macro avg) | 0.38 | 0.52+ |
| LLM-as-a-Judge score | 3.0 / 5 | 4.0 / 5 |
| Weighted Phase 1 Score | 0.45 | 0.60+ |
| Answer latency (p95) | < 5 sec | < 2 sec |
| Flagged answer rate | < 5% | < 1% |

---

## Ethics & Safety

- All outputs include a localised disclaimer recommending professional consultation
- PII is never stored; IP addresses are one-way hashed before logging
- An independent Oversight Committee reviews flagged answers monthly
- See `governance/oversight_committee_charter.md` and `governance/data_governance_policy.md`

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@misc{african-health-qa-2025,
  title   = {Multilingual African Health Assistant},
  year    = {2025},
  note    = {Zindi ITU Challenge: Multilingual Health QA in Low-Resource African Languages},
  url     = {https://github.com/your-org/african-health-assistant}
}
```
