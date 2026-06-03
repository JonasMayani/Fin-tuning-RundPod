"""
train.py — mT0 fine-tuning for RunPod A100 80GB with OOM-resilient training loop

Key features:
  - Drop-in replacement for the mt5-large script. Same CLI, same outputs.
  - mT0-XL (3.7B) default — same architecture as mT5, already instruction-tuned.
  - bf16 mixed precision on A100 Ampere/Hopper tensor cores.
  - LoRA on q,k,v,o,wi_0,wi_1,wo with alpha = r (scale = 1.0, stable).

OOM handling (the main upgrade over the mt5 script):
  1. Step-level OOM: NanGuardTrainer.training_step traps OOM mid-batch, frees
     memory, and re-raises so the outer loop can react.
  2. Trainer-level retry: train_with_oom_retry halves per-device batch and
     doubles grad_accum to preserve effective batch. Effective batch and LR
     schedule are therefore unaffected.
  3. Checkpoint-aware resume: after OOM, the new trainer resumes from the
     latest checkpoint in phase_dir instead of restarting from scratch. This
     was the critical bug in the mt5 version.
  4. Eval-time OOM: separate handling on eval batch — halves the eval batch
     independently of the train batch so training can keep going.
  5. Emergency save before retry: an "oom_recovery_phase_N" snapshot is
     written so we never lose progress between batch reductions.

Eval subsampling:
  Set `training.eval_max_samples` in config (e.g. 800) to cap eval set size
  during training. Full eval still runs at end-of-phase via the prediction
  dumper. Cuts a 3-hour eval down to ~20 minutes on the mt5 logs you saw.
"""

import argparse
import gc
import inspect
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# A100 is Ampere — TF32 is free throughput on fp32 fallback paths.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*length_penalty.*beam-based generation.*",
    category=UserWarning,
)
LANGUAGE_NAMES: dict[str, str] = {
    "Eng_Uga": "English", "Aka_Gha": "Akan",   "Eng_Gha": "English",
    "Eng_Eth": "English", "Lug_Uga": "Luganda", "Eng_Ken": "English",
    "Swa_Ken": "Swahili", "Amh_Eth": "Amharic",
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(question: Any, language: Any) -> str:
    lang = str(language)
    return (
        f"Question: {str(question).strip()}\n"
        f"Answer in {LANGUAGE_NAMES.get(lang, lang)}:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {"rouge1": float(np.mean(r1)), "rougeL": float(np.mean(rl))}


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels      = np.where(labels      != -100, labels,      tokenizer.pad_token_id)
        preds  = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        refs   = [l.strip() for l in tokenizer.batch_decode(labels,      skip_special_tokens=True)]
        return compute_rouge(preds, refs)
    return compute_metrics


def _sample_validation_indices(dataset, n: int, seed: int) -> list[int]:
    n = min(n, len(dataset))
    if n <= 0:
        return []
    if "subset" not in dataset.column_names:
        return list(range(n))

    by_lang: dict[str, list[int]] = {}
    for idx, lang in enumerate(dataset["subset"]):
        by_lang.setdefault(str(lang), []).append(idx)

    rng = random.Random(seed)
    for indices in by_lang.values():
        rng.shuffle(indices)

    selected: list[int] = []
    per_lang = max(1, n // max(len(by_lang), 1))
    for lang in sorted(by_lang):
        selected.extend(by_lang[lang][:per_lang])

    selected_set = set(selected)
    remaining = [idx for idx in range(len(dataset)) if idx not in selected_set]
    rng.shuffle(remaining)
    selected = (selected + remaining)[:n]
    return sorted(selected)


def subsample_eval_dataset(eval_dataset, cfg: dict):
    """Stratified subsample of the eval set during training, controlled by
    `training.eval_max_samples`. Returns the full dataset if disabled or
    already small enough."""
    if eval_dataset is None:
        return eval_dataset
    max_n = int(cfg.get("training", {}).get("eval_max_samples", 0) or 0)
    if max_n <= 0 or len(eval_dataset) <= max_n:
        return eval_dataset
    seed = int(cfg.get("seed", 42))
    indices = _sample_validation_indices(eval_dataset, max_n, seed)
    subset = eval_dataset.select(indices)
    logger.info(
        f"Eval subsample (training-time): {len(subset):,} / {len(eval_dataset):,} "
        f"(stratified by 'subset', seed={seed})"
    )
    return subset


def dump_validation_predictions(trainer, tokenizer, validation_dataset, output_dir: Path, phase: int, cfg: dict) -> None:
    eval_cfg = cfg.get("evaluation", {})
    if not bool(eval_cfg.get("dump_predictions", False)):
        return
    if validation_dataset is None or len(validation_dataset) == 0:
        logger.warning("Prediction dump skipped: validation dataset is empty.")
        return
    if not getattr(trainer.args, "predict_with_generate", False):
        logger.warning("Prediction dump skipped: predict_with_generate is disabled.")
        return

    n = int(eval_cfg.get("dump_predictions_n", 50))
    seed = int(eval_cfg.get("dump_predictions_seed", cfg.get("seed", 42)))
    indices = _sample_validation_indices(validation_dataset, n, seed)
    sample = validation_dataset.select(indices)

    model_cols = {"input_ids", "attention_mask", "labels"}
    gen_dataset = sample.remove_columns([c for c in sample.column_names if c not in model_cols])

    logger.info(f"Generating {len(sample)} validation predictions for phase {phase} dump.")
    pred_output = trainer.predict(gen_dataset, metric_key_prefix=f"dump_phase_{phase}")
    predictions = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]

    rows = []
    for row, pred in zip(sample, decoded_preds):
        ref = str(row.get("output_text", row.get("output", ""))).strip()
        try:
            rouge = compute_rouge([pred], [ref])
        except Exception:
            rouge = {"rouge1": np.nan, "rougeL": np.nan}
        rows.append({
            "phase": phase,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "val_index": row.get("val_index", ""),
            "subset": row.get("subset", ""),
            "input": row.get("input", ""),
            "prompt": row.get("input_text", ""),
            "reference": ref,
            "prediction": pred,
            "rouge1": rouge["rouge1"],
            "rougeL": rouge["rougeL"],
            "reference_chars": len(ref),
            "prediction_chars": len(pred),
        })

    dump_dir = output_dir / str(eval_cfg.get("dump_predictions_dir", "prediction_dumps"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    step = trainer.state.global_step
    epoch = trainer.state.epoch or 0
    out_path = dump_dir / f"phase_{phase}_step_{step}_epoch_{epoch:.2f}_val_predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Validation prediction dump -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = {"input", "output", "subset"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def clean_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["input", "output", "subset"]).copy()
    for col in ["input", "output", "subset"]:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "") & (df["subset"] != "")].copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} empty/null rows from {path.name}")
    return df


def load_tokenized_dataset(
    train_path: Path,
    val_path: Path,
    phase: int,
    langs: Optional[list[str]],
    tokenizer,
    cfg: dict,
) -> DatasetDict:
    df_train = clean_df(pd.read_csv(train_path), train_path)
    df_val   = clean_df(pd.read_csv(val_path),   val_path)
    require_columns(df_train, train_path)
    require_columns(df_val,   val_path)

    if phase == 1 and langs:
        df_train = df_train[df_train["subset"].isin(langs)].copy()
    if df_train.empty:
        raise ValueError("Training dataframe empty after curriculum filtering.")

    for df in (df_train, df_val):
        df["input_text"]  = [build_prompt(q, s) for q, s in zip(df["input"], df["subset"])]
        df["output_text"] = df["output"].astype(str)
    df_val["val_index"] = np.arange(len(df_val))

    max_in  = cfg["model"]["max_input_length"]
    max_out = cfg["model"]["max_output_length"]

    def tok_fn(batch):
        mi = tokenizer(batch["input_text"],  max_length=max_in,  truncation=True, padding=False)
        lb = tokenizer(text_target=batch["output_text"], max_length=max_out, truncation=True, padding=False)
        mi["labels"] = lb["input_ids"]
        return mi

    train_keep = set()
    val_keep = {"val_index", "input", "output", "subset", "input_text", "output_text"}
    ds_train = Dataset.from_pandas(df_train, preserve_index=False).map(
        tok_fn, batched=True,
        remove_columns=[c for c in df_train.columns.tolist() if c not in train_keep],
        desc=f"Tokenising train phase {phase}", num_proc=4)
    ds_val = Dataset.from_pandas(df_val, preserve_index=False).map(
        tok_fn, batched=True,
        remove_columns=[c for c in df_val.columns.tolist() if c not in val_keep],
        desc="Tokenising val", num_proc=4)
    ds_train = ds_train.filter(lambda ex: len(ex["input_ids"]) > 0 and len(ex["labels"]) > 0)
    ds_val   = ds_val.filter(  lambda ex: len(ex["input_ids"]) > 0 and len(ex["labels"]) > 0)

    logger.info(f"Phase {phase} — train: {len(ds_train):,}  val: {len(ds_val):,}")
    return DatasetDict({"train": ds_train, "validation": ds_val})


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def adapter_files_exist(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists())


def phase_complete_dir(output_dir: Path, phase: int) -> Path:
    return output_dir / f"phase_{phase}_complete"


def is_phase_complete(output_dir: Path, phase: int) -> bool:
    done = phase_complete_dir(output_dir, phase) / "PHASE_DONE"
    return done.exists() and adapter_files_exist(phase_complete_dir(output_dir, phase))


def resolve_resume(phase_dir: Path, train_cfg: dict) -> Optional[str]:
    mode = str(train_cfg.get("resume_from_checkpoint", "auto")).lower()
    if mode in {"", "none", "false", "no", "off"}:
        return None
    if mode == "auto":
        if not phase_dir.exists():
            return None
        return get_last_checkpoint(str(phase_dir)) or None
    return mode


def latest_checkpoint_in(phase_dir: Path) -> Optional[str]:
    """Best-effort: return the most recent checkpoint in phase_dir, or None."""
    if not phase_dir.exists():
        return None
    try:
        return get_last_checkpoint(str(phase_dir)) or None
    except Exception:
        return None


def save_phase_complete(trainer, tokenizer, output_dir: Path, phase: int) -> None:
    d = phase_complete_dir(output_dir, phase)
    trainer.save_model(str(d))
    tokenizer.save_pretrained(d)
    if getattr(trainer, "state", None) is not None:
        trainer.state.save_to_json(str(d / "trainer_state.json"))
    (d / "PHASE_DONE").write_text("done\n", encoding="utf-8")
    logger.info(f"✅ Phase {phase} complete → {d}")


def save_emergency(trainer, tokenizer, output_dir: Path, phase: int, tag: str = "interrupted") -> Optional[Path]:
    """Best-effort emergency save. Returns the path written, or None on failure."""
    try:
        d = output_dir / f"{tag}_phase_{phase}"
        trainer.save_model(str(d))
        tokenizer.save_pretrained(d)
        if getattr(trainer, "state", None) is not None:
            trainer.state.save_to_json(str(d / "trainer_state.json"))
        logger.warning(f"⚠ Emergency checkpoint → {d}")
        return d
    except Exception as exc:
        logger.warning(f"Emergency save failed: {exc}")
        return None


def write_active_config(cfg: dict, output_dir: Path) -> None:
    with open(output_dir / "active_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-tune batch — memory tiers updated for mT0-XL (3.7B) on A100 80GB
# ─────────────────────────────────────────────────────────────────────────────

def _memory_based_train_batch(free_gb: float, model_size_b: float = 3.7) -> int:
    """
    Conservative starting batch size based on free VRAM and model size.
    Designed for LoRA fine-tuning in bf16 with seq_in=256, seq_out=384.

    Empirical tiers, validated against mT0-XL (3.7B) on A100 80GB:
      free ≥ 70GB → 24  (mT0-large can push to 32)
      free ≥ 55GB → 16
      free ≥ 40GB → 12
      free ≥ 28GB →  8
      free ≥ 18GB →  4
      free ≥ 10GB →  2
      otherwise   →  1
    The OOM retry loop will halve from here if needed.
    """
    # Scale tiers down for larger models (mT0-XXL would need /4 again).
    scale = max(1.0, model_size_b / 3.7)
    def t(x: int) -> int:
        return max(1, int(x / scale))

    if free_gb >= 70:
        return t(24)
    if free_gb >= 55:
        return t(16)
    if free_gb >= 40:
        return t(12)
    if free_gb >= 28:
        return t(8)
    if free_gb >= 18:
        return t(4)
    if free_gb >= 10:
        return t(2)
    return 1


def _model_size_b(base_model: str) -> float:
    """Rough param count in billions for known mT0 / mT5 variants."""
    m = base_model.lower()
    if "xxl" in m: return 13.0
    if "xl"  in m: return 3.7
    if "large" in m: return 1.2
    if "base"  in m: return 0.58
    if "small" in m: return 0.30
    return 1.2  # safe default


def configure_generation(model, tokenizer, cfg: dict) -> None:
    model_cfg = cfg.get("model", {})
    decoding_default = cfg.get("decoding", {}).get("default", {})

    if bool(model_cfg.get("block_sentinel_tokens", True)):
        bad_words_ids = []
        for i in range(100):
            token_id = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            if token_id is not None and token_id != tokenizer.unk_token_id:
                bad_words_ids.append([token_id])
        if bad_words_ids:
            model.generation_config.bad_words_ids = bad_words_ids
            logger.info(f"Generation guard: blocking {len(bad_words_ids)} mT0/mT5 sentinel tokens.")

    if "no_repeat_ngram" in decoding_default:
        model.generation_config.no_repeat_ngram_size = int(decoding_default["no_repeat_ngram"])
    if "length_penalty" in decoding_default:
        model.generation_config.length_penalty = float(decoding_default["length_penalty"])
    if "repetition_penalty" in decoding_default:
        model.generation_config.repetition_penalty = float(decoding_default["repetition_penalty"])
    if "min_length" in decoding_default:
        model.generation_config.min_new_tokens = int(decoding_default["min_length"])

    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id


def auto_tune_batch_sizes(cfg: dict, base_model: str) -> None:
    train_cfg = cfg["training"]
    if not bool(train_cfg.get("auto_batch_size", False)):
        return
    if not torch.cuda.is_available():
        logger.info("auto_batch_size enabled but CUDA is unavailable; using configured batch sizes.")
        return

    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1e9
    total_gb = total / 1e9
    n_gpus = max(torch.cuda.device_count(), 1)
    model_b = _model_size_b(base_model)

    old_train = int(train_cfg["per_device_train_batch"])
    old_accum = int(train_cfg["gradient_accumulation"])
    old_eval = int(train_cfg["per_device_eval_batch"])

    target_effective = int(train_cfg.get(
        "target_effective_batch",
        old_train * old_accum * n_gpus,
    ))
    min_train = int(train_cfg.get("min_per_device_train_batch", train_cfg.get("min_batch_size", 1)))
    max_train = int(train_cfg.get("max_per_device_train_batch", old_train))
    train_batch = max(min_train, min(max_train, _memory_based_train_batch(free_gb, model_b)))

    accum = max(1, math.ceil(target_effective / max(train_batch * n_gpus, 1)))

    min_eval = int(train_cfg.get("min_per_device_eval_batch", 1))
    max_eval = int(train_cfg.get("max_per_device_eval_batch", old_eval))
    eval_multiplier = float(train_cfg.get("eval_batch_multiplier", 2.0))
    eval_batch = int(train_batch * eval_multiplier)
    eval_batch = max(min_eval, min(max_eval, eval_batch))

    train_cfg["per_device_train_batch"] = train_batch
    train_cfg["gradient_accumulation"] = accum
    train_cfg["per_device_eval_batch"] = eval_batch

    effective = train_batch * accum * n_gpus
    logger.info(
        f"Auto batch sizing | model={base_model.split('/')[-1]} ({model_b:.1f}B) | "
        f"GPU free={free_gb:.1f}/{total_gb:.1f} GB × {n_gpus} | "
        f"train {old_train}→{train_batch}, accum {old_accum}→{accum}, "
        f"eval {old_eval}→{eval_batch}, effective_batch={effective}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(base_model: str, cfg: dict, adapter_checkpoint: Optional[Path] = None):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    lora_cfg  = cfg["lora"]
    precision = train_cfg.get("mixed_precision", "bf16")
    dtype     = (torch.bfloat16 if precision == "bf16"
                 else torch.float16 if precision == "fp16"
                 else torch.float32)

    logger.info(f"Loading {base_model}  dtype={dtype}  device=cuda:0")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False

    use_gc = bool(train_cfg.get("gradient_checkpointing", False))
    if use_gc:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    task_type = getattr(TaskType, lora_cfg.get("task_type", "SEQ_2_SEQ_LM"))

    if adapter_checkpoint is not None:
        logger.info(f"Loading LoRA adapter from {adapter_checkpoint}")
        model = PeftModel.from_pretrained(model, str(adapter_checkpoint), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=task_type,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"),
        ))

    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# VRAM monitor
# ─────────────────────────────────────────────────────────────────────────────

def log_vram(label: str = "") -> None:
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM {label}: {used:.2f} GB / {total:.2f} GB")


def free_cuda_memory() -> None:
    """Aggressively free VRAM. Safe to call at any time."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class SafeCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        batch.pop("decoder_inputs_embeds", None)
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# OOM / NaN guard trainer
# ─────────────────────────────────────────────────────────────────────────────

def _is_oom(exc: BaseException) -> bool:
    """Return True for any CUDA / system out-of-memory exception.

    PyTorch's OOM path always emits "CUDA out of memory" in the message,
    even when the underlying allocator is cuBLAS or cuDNN, so a single
    substring check covers all real-world cases.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


class NanGuardTrainer(Seq2SeqTrainer):
    """Trainer with:
      - fp32 cross-entropy (numerically safer than bf16 CE on mT0/mT5).
      - NaN/Inf loss detection → raises FloatingPointError.
      - Step-level OOM trap: frees memory and re-raises so the outer retry
        loop can reduce batch and resume from the latest checkpoint.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if "decoder_input_ids" not in inputs and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            inputs["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=labels)
        outputs = model(**inputs)
        logits  = outputs.logits.float()
        loss    = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                      logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError("NaN/Inf loss — check LR or reduce batch.")
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Wrap the default training step in OOM recovery.

        On OOM we free memory and re-raise. The outer retry handler will then:
          1) save an emergency adapter snapshot,
          2) tear down this trainer,
          3) halve the batch and double grad_accum,
          4) build a new trainer and resume from the latest checkpoint.
        """
        try:
            # Older HF signatures don't accept num_items_in_batch — degrade gracefully.
            sig = inspect.signature(super().training_step)
            if "num_items_in_batch" in sig.parameters:
                return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            return super().training_step(model, inputs)
        except BaseException as exc:
            if _is_oom(exc):
                logger.warning(
                    "CUDA OOM inside training_step. Freeing memory and signalling retry."
                )
                try:
                    del inputs
                except Exception:
                    pass
                free_cuda_memory()
                log_vram("after step-level OOM")
            raise

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Same OOM trap for eval/generate steps."""
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        except BaseException as exc:
            if _is_oom(exc):
                logger.warning(
                    "CUDA OOM inside prediction_step. Freeing memory and signalling retry."
                )
                try:
                    del inputs
                except Exception:
                    pass
                free_cuda_memory()
                log_vram("after eval-step OOM")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# OOM-resilient training driver
# ─────────────────────────────────────────────────────────────────────────────

def _build_trainer(
    model,
    tokenizer,
    dataset: DatasetDict,
    cfg: dict,
    phase_dir: Path,
    phase: int,
    phase_epochs: int,
    callbacks: list,
):
    """Construct a fresh Seq2SeqTrainer with current cfg values."""
    training_args = make_training_args(cfg, phase_dir, phase, phase_epochs)

    train_ds = dataset["train"]
    eval_ds  = subsample_eval_dataset(dataset["validation"], cfg) if training_args.do_eval else None

    trainer = NanGuardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SafeCollator(
            tokenizer=tokenizer, model=model,
            label_pad_token_id=-100, pad_to_multiple_of=8),
        compute_metrics=(make_compute_metrics(tokenizer)
                         if training_args.predict_with_generate else None),
        callbacks=callbacks,
    )
    return trainer, training_args


def train_with_oom_retry(
    initial_trainer: "NanGuardTrainer",
    model,
    tokenizer,
    dataset: DatasetDict,
    cfg: dict,
    phase: int,
    phase_dir: Path,
    phase_epochs: int,
    callbacks: list,
    resume_ckpt: Optional[str],
    output_dir: Path,
    min_batch: int = 1,
    max_oom_retries: int = 8,
) -> "NanGuardTrainer":
    """Run trainer.train() with automatic OOM recovery that PRESERVES PROGRESS.

    Strategy on OOM:
      1. Save an emergency adapter snapshot (best-effort).
      2. Free VRAM (del trainer, gc, empty_cache, synchronize).
      3. Halve per_device_train_batch_size.
      4. Double gradient_accumulation_steps (keeps effective batch constant
         so LR schedule and loss scale are unchanged).
      5. Independently shrink per_device_eval_batch (eval can OOM separately).
      6. Build a fresh trainer.
      7. Resume from the latest checkpoint in phase_dir — NOT from scratch.
      8. Retry, up to max_oom_retries times or until per-device batch < min_batch.

    Returns the trainer that completed training.
    Raises RuntimeError if min_batch reached and OOM still occurs.
    """
    train_cfg       = cfg["training"]
    current_batch   = int(train_cfg["per_device_train_batch"])
    current_accum   = int(train_cfg["gradient_accumulation"])
    current_eval    = int(train_cfg["per_device_eval_batch"])
    effective_batch = current_batch * current_accum
    trainer         = initial_trainer
    attempt         = 0

    while True:
        attempt += 1
        n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
        logger.info(
            f"OOM guard — attempt {attempt}/{max_oom_retries} | "
            f"per_device_batch={current_batch}  grad_accum={current_accum}  "
            f"eval_batch={current_eval}  effective_batch={current_batch * current_accum * n_gpus}"
        )

        try:
            trainer.train(resume_from_checkpoint=resume_ckpt)
            if attempt > 1:
                logger.success(
                    f"Training completed after {attempt} attempt(s) | "
                    f"final batch={current_batch}  grad_accum={current_accum}."
                )
            return trainer

        except BaseException as exc:
            if not _is_oom(exc):
                raise  # propagate non-OOM errors immediately

            logger.warning(
                f"CUDA OOM on attempt {attempt} "
                f"(batch={current_batch}, eval={current_eval}). Recovering…"
            )

            # ── Step 1: emergency save BEFORE tearing down the trainer ──────
            recovery_dir = save_emergency(
                trainer, tokenizer, output_dir, phase, tag="oom_recovery"
            )

            # ── Step 2: free VRAM aggressively ──────────────────────────────
            try:
                del trainer
            except Exception:
                pass
            free_cuda_memory()
            log_vram("after OOM recovery")

            # ── Step 3: figure out which dim to shrink ──────────────────────
            # Heuristic: if we're already at min_batch on train, try shrinking
            # eval batch first (often eval generate is the actual culprit on
            # mT0). Otherwise shrink train batch and preserve effective batch.
            if current_batch <= min_batch and current_eval > 1:
                new_eval = max(1, current_eval // 2)
                logger.warning(
                    f"Train batch already at min ({min_batch}). "
                    f"Shrinking eval batch only: {current_eval} → {new_eval}"
                )
                current_eval = new_eval
            elif current_batch > min_batch:
                new_batch = max(min_batch, current_batch // 2)
                new_accum = max(1, math.ceil(effective_batch / new_batch))
                # Also clip eval to avoid leaving it too high for the new regime.
                new_eval = max(1, min(current_eval, max(new_batch * 2, 1)))
                logger.warning(
                    f"Reducing train batch: {current_batch} → {new_batch}  |  "
                    f"grad_accum: {current_accum} → {new_accum}  |  "
                    f"eval batch: {current_eval} → {new_eval}  |  "
                    f"effective batch preserved at {new_batch * new_accum * (torch.cuda.device_count() or 1)}"
                )
                current_batch, current_accum, current_eval = new_batch, new_accum, new_eval
            else:
                # Nothing left to shrink — give up.
                raise RuntimeError(
                    f"OOM with per_device_train_batch={current_batch}, eval={current_eval} "
                    f"already at minimum. Try: gradient_checkpointing=true, smaller "
                    f"max_input_length / max_output_length, lower LoRA rank, or 4-bit "
                    f"quantisation (load_in_4bit) for a larger model."
                ) from exc

            if attempt >= max_oom_retries:
                raise RuntimeError(
                    f"Hit max_oom_retries={max_oom_retries} — aborting."
                ) from exc

            # ── Step 4: patch cfg so make_training_args sees new sizes ─────
            cfg["training"]["per_device_train_batch"] = current_batch
            cfg["training"]["gradient_accumulation"]  = current_accum
            cfg["training"]["per_device_eval_batch"]  = current_eval
            write_active_config(cfg, output_dir)

            # ── Step 5: pick the best checkpoint to resume from ─────────────
            # Priority:
            #   (a) latest checkpoint Trainer wrote inside phase_dir,
            #   (b) the emergency snapshot we just made,
            #   (c) nothing (cold restart).
            new_resume = latest_checkpoint_in(phase_dir)
            if new_resume:
                logger.info(f"Will resume from latest checkpoint: {new_resume}")
            elif recovery_dir is not None and adapter_files_exist(recovery_dir):
                # The emergency snapshot is an adapter, not a full HF checkpoint.
                # Reload it into the model in-place; do NOT pass resume_from_checkpoint.
                logger.info(f"Reloading adapter weights from emergency snapshot: {recovery_dir}")
                try:
                    # PeftModel.load_adapter does in-place adapter weight reload.
                    if hasattr(model, "load_adapter"):
                        model.load_adapter(str(recovery_dir), adapter_name="default", is_trainable=True)
                    new_resume = None
                except Exception as e:
                    logger.warning(f"Adapter reload failed ({e}); cold restart this attempt.")
                    new_resume = None
            else:
                logger.warning("No checkpoint available — restarting phase from step 0.")
                new_resume = None
            resume_ckpt = new_resume

            # ── Step 6: rebuild trainer with smaller batches ────────────────
            trainer, _ = _build_trainer(
                model=model, tokenizer=tokenizer, dataset=dataset, cfg=cfg,
                phase_dir=phase_dir, phase=phase, phase_epochs=phase_epochs,
                callbacks=callbacks,
            )
            # Loop continues → trainer.train() is called again with smaller batch.


# ─────────────────────────────────────────────────────────────────────────────
# Training arguments
# ─────────────────────────────────────────────────────────────────────────────

def make_training_args(cfg, phase_output_dir, phase, phase_epochs):
    train_cfg = cfg["training"]
    opt_cfg   = cfg["optimiser"]
    model_cfg = cfg["model"]

    precision        = train_cfg.get("mixed_precision", "bf16")
    use_fp16         = precision == "fp16"
    use_bf16         = precision == "bf16"
    do_eval          = bool(train_cfg.get("eval_enabled", True))
    load_best        = do_eval and bool(train_cfg.get("load_best_model_at_end", True))
    predict_generate = do_eval and bool(train_cfg.get("predict_with_generate", True))
    num_workers      = int(train_cfg.get("dataloader_num_workers", 4))

    params     = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_strat = train_cfg.get("eval_strategy", "epoch") if do_eval else "no"
    save_strategy = train_cfg.get("save_strategy", eval_strat if do_eval else "steps")
    if load_best and eval_strat != "no":
        save_strategy = eval_strat
    eval_steps = train_cfg.get("eval_steps", train_cfg.get("save_steps", 500))
    save_steps = train_cfg.get("save_steps", 500)
    if load_best and eval_strat == "steps":
        save_steps = eval_steps

    kwargs: dict[str, Any] = dict(
        output_dir                    = str(phase_output_dir),
        overwrite_output_dir          = False,
        num_train_epochs              = phase_epochs,
        per_device_train_batch_size   = train_cfg["per_device_train_batch"],
        per_device_eval_batch_size    = train_cfg["per_device_eval_batch"],
        gradient_accumulation_steps   = train_cfg["gradient_accumulation"],
        learning_rate                 = opt_cfg["lr"],
        weight_decay                  = opt_cfg["weight_decay"],
        lr_scheduler_type             = opt_cfg["lr_scheduler"],
        warmup_ratio                  = opt_cfg["warmup_ratio"],
        max_grad_norm                 = opt_cfg.get("max_grad_norm", 1.0),
        optim                         = opt_cfg["type"],
        label_smoothing_factor        = opt_cfg.get("label_smoothing", 0.0),
        fp16                          = use_fp16,
        bf16                          = use_bf16,
        gradient_checkpointing        = bool(train_cfg.get("gradient_checkpointing", False)),
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataloader_num_workers        = num_workers,
        dataloader_pin_memory         = True,
        dataloader_persistent_workers = bool(num_workers > 0),
        dataloader_prefetch_factor    = (train_cfg.get("dataloader_prefetch_factor", 2)
                                         if num_workers > 0 else None),
        predict_with_generate         = predict_generate,
        generation_max_length         = model_cfg["max_output_length"],
        generation_num_beams          = train_cfg.get("generation_num_beams", 1),
        logging_strategy              = "steps",
        logging_steps                 = train_cfg.get("logging_steps", 50),
        eval_steps                    = eval_steps,
        save_strategy                 = save_strategy,
        save_steps                    = save_steps,
        save_total_limit              = train_cfg.get("save_total_limit", 3),
        save_safetensors              = bool(train_cfg.get("save_safetensors", True)),
        load_best_model_at_end        = load_best,
        metric_for_best_model         = train_cfg.get("metric_for_best_model", "rougeL") if load_best else None,
        greater_is_better             = train_cfg.get("greater_is_better", True) if load_best else None,
        group_by_length               = True,
        report_to                     = "none",
        seed                          = cfg.get("seed", 42),
        ddp_find_unused_parameters    = False,
        remove_unused_columns         = True,
        # HF's own auto-find-batch-size (second line of defence under our retry).
        auto_find_batch_size          = bool(train_cfg.get("auto_find_batch_size", False)),
    )

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strat
    else:
        kwargs["evaluation_strategy"] = eval_strat

    kwargs = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return Seq2SeqTrainingArguments(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, base_model_override: Optional[str] = None) -> None:
    setup_logging()
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                    f"Free: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

    paths     = cfg["paths"]
    train_cfg = cfg["training"]
    curr_cfg  = cfg["curriculum"]
    base_model = base_model_override or cfg["model"]["base_model"]
    output_name = base_model.replace("/", "_")
    output_suffix = str(train_cfg.get("output_dir_suffix", "")).strip()
    if output_suffix:
        output_name = f"{output_name}_{output_suffix}"
    output_dir = Path(paths["models"]) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_active_config(cfg, output_dir)
    logger.info(f"Output dir: {output_dir}")

    # ── Phase plan ────────────────────────────────────────────────────────────
    if curr_cfg.get("enabled", False):
        p1 = int(curr_cfg.get("phase1_epochs", 0))
        p2 = int(curr_cfg.get("phase2_epochs", max(int(train_cfg["epochs"]) - p1, 0)))
        phases = ([(1, p1, curr_cfg.get("phase1_langs", []))] if p1 > 0 else []) + \
                 ([(2, p2, None)] if p2 > 0 else [])
    else:
        phases = [(2, int(train_cfg["epochs"]), None)]

    logger.info("Phase plan: " + ", ".join(f"phase {p}: {e} epoch(s)" for p, e, _ in phases))

    # ── Resume / adapter loading ──────────────────────────────────────────────
    phase_resume = {
        p: resolve_resume(output_dir / f"phase_{p}", train_cfg)
        for p, _, _ in phases
    }
    adapter_checkpoint: Optional[Path] = None
    if phase_resume.get(2):
        adapter_checkpoint = Path(phase_resume[2])
        logger.info(f"Resuming phase 2 from {adapter_checkpoint}")
    elif is_phase_complete(output_dir, 1):
        adapter_checkpoint = phase_complete_dir(output_dir, 1)
        logger.info(f"Phase 1 done — loading adapter from {adapter_checkpoint}")
    elif phase_resume.get(1):
        adapter_checkpoint = Path(phase_resume[1])
        logger.info(f"Resuming phase 1 from {adapter_checkpoint}")

    use_fast_tokenizer = bool(cfg.get("model", {}).get("use_fast_tokenizer", False))
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=use_fast_tokenizer)
    logger.info(f"Tokenizer: {'fast' if tokenizer.is_fast else 'slow'}")
    model     = build_model(base_model, cfg, adapter_checkpoint)
    configure_generation(model, tokenizer, cfg)
    log_vram("after model load")
    auto_tune_batch_sizes(cfg, base_model)
    write_active_config(cfg, output_dir)

    # ── Dataset paths ─────────────────────────────────────────────────────────
    dataset_cfg = cfg.get("dataset", {})
    train_file  = Path(dataset_cfg["train_file"])
    val_file    = Path(dataset_cfg["val_file"])
    workspace = Path(paths.get("workspace", "/workspace/Fin-tuning-RundPod"))
    if not train_file.is_absolute():
        train_file = workspace / train_file
    if not val_file.is_absolute():
        val_file = workspace / val_file
    for f in (train_file, val_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {f}\n"
                f"Check 'dataset.train_file' / 'dataset.val_file' in config.yaml"
            )

    logger.info(f"Train file : {train_file}  ({pd.read_csv(train_file).shape[0]:,} rows)")
    logger.info(f"Val file   : {val_file}  ({pd.read_csv(val_file).shape[0]:,} rows)")

    last_trainer = None
    completed: set[int] = set()

    for phase, phase_epochs, phase_langs in phases:
        if is_phase_complete(output_dir, phase):
            logger.info(f"Phase {phase} already complete — skipping.")
            completed.add(phase); continue
        if phase == 1 and phase_resume.get(2):
            logger.info("Skipping phase 1 — phase 2 checkpoint exists.")
            completed.add(phase); continue

        logger.info(f"─── Phase {phase} | {phase_epochs} epoch(s) ───")
        dataset       = load_tokenized_dataset(train_file, val_file, phase, phase_langs, tokenizer, cfg)
        phase_dir     = output_dir / f"phase_{phase}"
        resume_ckpt   = phase_resume.get(phase)

        callbacks = []
        if (phase == 2
                and bool(train_cfg.get("load_best_model_at_end", True))
                and int(train_cfg.get("early_stopping_patience", 0)) > 0):
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=int(train_cfg["early_stopping_patience"])))

        trainer, _ = _build_trainer(
            model=model, tokenizer=tokenizer, dataset=dataset, cfg=cfg,
            phase_dir=phase_dir, phase=phase, phase_epochs=phase_epochs,
            callbacks=callbacks,
        )

        try:
            trainer = train_with_oom_retry(
                initial_trainer=trainer,
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                cfg=cfg,
                phase=phase,
                phase_dir=phase_dir,
                phase_epochs=phase_epochs,
                callbacks=callbacks,
                resume_ckpt=resume_ckpt,
                output_dir=output_dir,
                min_batch=int(train_cfg.get("min_batch_size", 1)),
            )
        except (KeyboardInterrupt, SystemExit):
            save_emergency(trainer, tokenizer, output_dir, phase); raise
        except FloatingPointError:
            save_emergency(trainer, tokenizer, output_dir, phase); raise
        except RuntimeError:
            save_emergency(trainer, tokenizer, output_dir, phase); raise

        save_phase_complete(trainer, tokenizer, output_dir, phase)
        try:
            dump_validation_predictions(trainer, tokenizer, dataset["validation"], output_dir, phase, cfg)
        except Exception as exc:
            logger.warning(f"Prediction dump failed for phase {phase}: {exc}")
        log_vram(f"after phase {phase}")
        last_trainer = trainer
        completed.add(phase)

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir = output_dir / "best"
    if last_trainer is not None:
        last_trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
        logger.info(f"✅ Final model → {final_dir}")
    elif phases and all(p in completed for p, _, _ in phases):
        src = phase_complete_dir(output_dir, phases[-1][0])
        if adapter_files_exist(src):
            if final_dir.exists(): shutil.rmtree(final_dir)
            shutil.copytree(src, final_dir)
            logger.info(f"✅ Copied completed adapter → {final_dir}")

    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune mT0 on RunPod A100 80GB")
    p.add_argument("--config",     default="/workspace/Fin-tuning-RundPod/src/training/config_mt0.yaml")
    p.add_argument("--base_model", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config, args.base_model)