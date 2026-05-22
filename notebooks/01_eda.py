"""
notebooks/01_eda.py  ←→  01_eda.ipynb
─────────────────────────────────────────────────────────────────────────────
Phase 1.1 — Exploratory Data Analysis
Multilingual African Health Assistant | Zindi ITU Challenge

Run as a notebook:
  jupyter nbconvert --to notebook --execute notebooks/01_eda.py

Or as a script:
  python notebooks/01_eda.py
"""

# %% [markdown]
# # Phase 1.1 — Exploratory Data Analysis
# **Multilingual African Health QA | Zindi ITU Challenge**
#
# This notebook performs a comprehensive EDA on Train, Val, and Test sets.

# %% Imports
import warnings
warnings.filterwarnings("ignore")

import re
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Set style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

DATA_RAW  = Path("data/raw")
REPORTS   = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# 

VALID_SUBSETS = {
    "Eng_Uga", "Aka_Gha", "Eng_Gha", "Eng_Eth",
    "Lug_Uga", "Eng_Ken", "Swa_Ken", "Amh_Eth"
}

LANGUAGE_LABELS = {
    "Eng_Uga": "English\nUganda",
    "Aka_Gha": "Akan (Twi)\nGhana",
    "Eng_Gha": "English\nGhana",
    "Eng_Eth": "English\nEthiopia",
    "Lug_Uga": "Luganda\nUganda",
    "Eng_Ken": "English\nKenya",
    "Swa_Ken": "Swahili\nKenya",
    "Amh_Eth": "Amharic\nEthiopia",
}

LANG_COLORS = {
    "Eng_Uga": "#2A9D8F",
    "Aka_Gha": "#E63946",
    "Eng_Gha": "#F4A261",
    "Eng_Eth": "#457B9D",
    "Lug_Uga": "#8338EC",
    "Eng_Ken": "#06D6A0",
    "Swa_Ken": "#FFB703",
    "Amh_Eth": "#FB5607",
}

# %% Load Data
def load_split(name: str) -> pd.DataFrame:
    path = DATA_RAW / f"{name}.csv"
    if not path.exists():
        logger.warning(f"{path} not found — creating synthetic example set.")
        # Synthetic placeholder so EDA can run without real data
        langs   = ["Aka_Gha", "Amh_Eth", "Lug_Uga", "Swa_Ken", "Eng_Uga"]
        records = []
        for i in range(500):
            lang = langs[i % 5]
            records.append({
                "ID": f"{name[:3].upper()}_{i:06d}",
                "input":  f"Sample health question {i} in {lang}?",
                "output": f"Sample answer {i} with medical information for {lang}.",
                "subset": lang,
            })
        return pd.DataFrame(records)
    return pd.read_csv(path)

df_train = load_split("Train")
df_val   = load_split("Val")
df_test  = load_split("Test")

logger.info(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# %% 1. Record Counts Per Language

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Record Counts per Language Tag", fontsize=14, fontweight="bold", y=1.02)

for ax, (df, title) in zip(axes, [
    (df_train, "Train"), (df_val, "Validation"), (df_test, "Test")
]):
    counts = df["subset"].value_counts()
    colors = [LANG_COLORS.get(l, "#888") for l in counts.index]
    bars   = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Language Tag")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(REPORTS / "language_distribution.png", bbox_inches="tight")
plt.show()
logger.success("Saved language_distribution.png")

# %% 2. Token-length Distributions

def token_count(text: str) -> int:
    return len(str(text).split())

for df, name in [(df_train, "train"), (df_val, "val")]:
    df[f"q_len"] = df["input"].apply(token_count)
    df[f"a_len"] = df["output"].apply(token_count) if "output" in df.columns else 0

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle("Question & Answer Length Distributions (tokens)", fontsize=14, fontweight="bold")

langs = sorted(df_train["subset"].unique())
n_langs = len(langs)
fig, axes = plt.subplots(2, n_langs, figsize=(n_langs * 3.8, 8))
fig.suptitle("Question & Answer Length Distributions (tokens)", fontsize=14, fontweight="bold")

for col, lang in enumerate(langs):
    g = df_train[df_train["subset"] == lang]
    axes[0, col].hist(g["q_len"], bins=40, color=LANG_COLORS.get(lang, "#888"), alpha=0.8)
    axes[0, col].set_title(f"{lang}\nQuestions", fontsize=9)
    axes[0, col].set_xlabel("Tokens")
    axes[1, col].hist(g["a_len"], bins=40, color=LANG_COLORS.get(lang, "#888"), alpha=0.8)
    axes[1, col].set_title(f"{lang}\nAnswers", fontsize=9)
    axes[1, col].set_xlabel("Tokens")

plt.tight_layout()
plt.savefig(REPORTS / "length_distributions.png", bbox_inches="tight")
plt.show()
logger.success("Saved length_distributions.png")

# %% 3. Summary Statistics Table

stats_rows = []
for lang in langs:
    g = df_train[df_train["subset"] == lang]
    stats_rows.append({
        "Language":         lang,
        "Train N":          len(g),
        "Q len mean":       round(g["q_len"].mean(), 1),
        "Q len median":     round(g["q_len"].median(), 1),
        "Q len p95":        round(g["q_len"].quantile(0.95), 1),
        "A len mean":       round(g["a_len"].mean(), 1),
        "A len median":     round(g["a_len"].median(), 1),
        "A len p95":        round(g["a_len"].quantile(0.95), 1),
    })

stats_df = pd.DataFrame(stats_rows)
print("\n── Length Statistics per Language ──")
print(stats_df.to_string(index=False))

# %% 4. Duplicate Detection

import hashlib

def row_hash(row):
    txt = str(row.get("input", "")) + "|||" + str(row.get("output", ""))
    return hashlib.md5(txt.encode()).hexdigest()

df_train["_hash"] = df_train.apply(row_hash, axis=1)
exact_dupes = df_train[df_train.duplicated("_hash", keep=False)].copy()
exact_dupes["duplicate_type"] = "exact"

logger.info(f"Exact duplicates in train: {len(exact_dupes)} rows "
            f"({100*len(exact_dupes)/len(df_train):.2f}%)")

# Near-duplicate detection via cosine sim (sample for speed)
try:
    from sentence_transformers import SentenceTransformer
    import sklearn.metrics.pairwise as pw
    sample = df_train.sample(min(1000, len(df_train)), random_state=42)
    model  = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embs   = model.encode(sample["input"].tolist(), show_progress_bar=True)
    sims   = pw.cosine_similarity(embs)
    np.fill_diagonal(sims, 0)
    near_dup_count = int((sims > 0.90).sum() / 2)
    logger.info(f"Near-duplicate pairs (cosine > 0.90) in sample: {near_dup_count}")
except Exception as e:
    logger.warning(f"Near-duplicate detection skipped: {e}")
    near_dup_count = 0

dupes_report = pd.DataFrame({
    "metric":  ["exact_duplicates", "near_duplicates_sample", "total_train"],
    "count":   [len(exact_dupes), near_dup_count, len(df_train)],
})
dupes_report.to_csv(REPORTS / "duplicates_report.csv", index=False)
logger.success("Saved duplicates_report.csv")

# %% 5. Class Imbalance Analysis

train_counts = df_train["subset"].value_counts()
max_lang     = train_counts.max()
imbalance    = pd.DataFrame({
    "language":      train_counts.index,
    "count":         train_counts.values,
    "pct_of_total":  (train_counts.values / len(df_train) * 100).round(2),
    "ratio_to_max":  (train_counts.values / max_lang).round(3),
})
print("\n── Class Imbalance ──")
print(imbalance.to_string(index=False))

# %% 6. Missing Values & Encoding Anomalies

print("\n── Missing Values ──")
for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
    missing = df.isnull().sum()
    if missing.any():
        print(f"{name}: {missing[missing > 0].to_dict()}")
    else:
        print(f"{name}: ✓ No missing values")

# Unicode anomaly check
def has_anomaly(text: str) -> bool:
    try:
        text.encode("utf-8").decode("utf-8")
        return False
    except Exception:
        return True

anomalies = df_train["input"].apply(has_anomaly).sum()
logger.info(f"Encoding anomalies in train inputs: {anomalies}")

# %% 7. Sample 5 Q&A Pairs per Language

print("\n" + "═" * 70)
print("SAMPLE Q&A PAIRS PER LANGUAGE")
print("═" * 70)

for lang in langs:
    print(f"\n── {lang} ({''.join(['─']*(60-len(lang)))})")
    sample_rows = df_train[df_train["subset"] == lang].sample(
        min(5, sum(df_train["subset"] == lang)), random_state=42
    )
    for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
        print(f"\n  [{i}] Q: {str(row['input'])[:120]}")
        if "output" in row:
            print(f"      A: {str(row['output'])[:200]}")

# %% 8. Vocabulary Overlap (Train vs Val)

def get_vocab(series: pd.Series) -> set:
    tokens = set()
    for text in series.dropna():
        tokens.update(str(text).lower().split())
    return tokens

train_vocab = get_vocab(df_train["input"])
val_vocab   = get_vocab(df_val["input"])
overlap     = train_vocab & val_vocab

print(f"\n── Vocabulary Overlap (Train vs Val) ──")
print(f"  Train vocab size : {len(train_vocab):,}")
print(f"  Val vocab size   : {len(val_vocab):,}")
print(f"  Overlap          : {len(overlap):,}  ({100*len(overlap)/len(val_vocab):.1f}% of val vocab in train)")

# %% 9. Generate HTML EDA Report

try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(
        df_train[["subset", "q_len", "a_len"]],
        title="African Health QA — EDA Report",
        explorative=True,
    )
    profile.to_file(REPORTS / "eda_report.html")
    logger.success("Saved eda_report.html")
except Exception as e:
    logger.warning(f"ydata-profiling not available: {e}. Skipping HTML report.")
    # Fallback: simple HTML summary
    html = f"""<!DOCTYPE html>
<html><head><title>EDA Report</title></head><body>
<h1>EDA Report — African Health QA</h1>
<h2>Dataset Summary</h2>
<p>Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}</p>
{stats_df.to_html(index=False)}
<h2>Class Imbalance</h2>
{imbalance.to_html(index=False)}
</body></html>"""
    (REPORTS / "eda_report.html").write_text(html)
    logger.success("Saved minimal eda_report.html")

print("\n✓ Phase 1.1 EDA complete. Artefacts saved to reports/")
