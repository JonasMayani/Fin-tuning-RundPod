"""
train.py — mt5-base fine-tuning for RTX 4060 Laptop GPU (8 GB VRAM)
- No bitsandbytes / 8-bit quantisation needed for mt5-base
- fp16 mixed precision (stable without quantisation)
- LoRA keeps trainable params tiny (~0.5 % of base model)
- Full checkpoint/resume: safe to Ctrl-C and restart at any time
"""

import argparse
import inspect
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

# ── Env flags (set before torch import) ────────────────────────────────────
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

# ── RTX 40-series: enable TF32 for matmuls (free throughput on Ampere+) ───
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_NAMES: dict[str, str] = {
    "Eng_Uga": "English",
    "Aka_Gha": "Akan",
    "Eng_Gha": "English",
    "Eng_Eth": "English",
    "Lug_Uga": "Luganda",
    "Eng_Ken": "English",
    "Swa_Ken": "Swahili",
    "Amh_Eth": "Amharic",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def build_prompt(question: Any, language: Any) -> str:
    lang = str(language)
    return f"Answer in {LANGUAGE_NAMES.get(lang, lang)}: {str(question)}"


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("pip install rouge-score") from exc
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
        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels,      skip_special_tokens=True)]
        return compute_rouge(decoded_preds, decoded_labels)
    return compute_metrics


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
    curriculum_phase: int,
    curriculum_langs: Optional[list[str]],
    tokenizer,
    cfg: dict,
) -> DatasetDict:
    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    require_columns(df_train, train_path)
    require_columns(df_val,   val_path)
    df_train = clean_df(df_train, train_path)
    df_val   = clean_df(df_val,   val_path)

    if curriculum_phase == 1 and curriculum_langs:
        df_train = df_train[df_train["subset"].isin(curriculum_langs)].copy()
    if df_train.empty:
        raise ValueError("Training dataframe is empty after curriculum filtering.")

    df_train["input_text"]  = [build_prompt(q, s) for q, s in zip(df_train["input"], df_train["subset"])]
    df_train["output_text"] = df_train["output"].astype(str)
    df_val["input_text"]    = [build_prompt(q, s) for q, s in zip(df_val["input"],   df_val["subset"])]
    df_val["output_text"]   = df_val["output"].astype(str)

    max_in  = cfg["model"]["max_input_length"]
    max_out = cfg["model"]["max_output_length"]

    def tok_fn(batch):
        model_inputs = tokenizer(batch["input_text"],  max_length=max_in,  truncation=True, padding=False)
        labels       = tokenizer(text_target=batch["output_text"], max_length=max_out, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds_train = Dataset.from_pandas(df_train, preserve_index=False).map(
        tok_fn, batched=True, remove_columns=df_train.columns.tolist(),
        desc=f"Tokenising train phase {curriculum_phase}",
    )
    ds_val = Dataset.from_pandas(df_val, preserve_index=False).map(
        tok_fn, batched=True, remove_columns=df_val.columns.tolist(),
        desc="Tokenising validation",
    )
    ds_train = ds_train.filter(lambda ex: len(ex["input_ids"]) > 0 and len(ex["labels"]) > 0)
    ds_val   = ds_val.filter(  lambda ex: len(ex["input_ids"]) > 0 and len(ex["labels"]) > 0)

    logger.info(f"Phase {curriculum_phase} — train: {len(ds_train):,}  val: {len(ds_val):,}")
    return DatasetDict({"train": ds_train, "validation": ds_val})


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def adapter_files_exist(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


def phase_complete_dir(output_dir: Path, phase: int) -> Path:
    return output_dir / f"phase_{phase}_complete"


def phase_done_file(output_dir: Path, phase: int) -> Path:
    return phase_complete_dir(output_dir, phase) / "PHASE_DONE"


def is_phase_complete(output_dir: Path, phase: int) -> bool:
    return phase_done_file(output_dir, phase).exists() and \
           adapter_files_exist(phase_complete_dir(output_dir, phase))


def resolve_resume_checkpoint(phase_dir: Path, train_cfg: dict) -> Optional[str]:
    mode = train_cfg.get("resume_from_checkpoint", "auto")
    if isinstance(mode, bool):
        mode = "auto" if mode else "none"
    mode = str(mode).lower()
    if mode in {"", "none", "false", "no", "off"}:
        return None
    if mode == "auto":
        if not phase_dir.exists():
            return None
        ckpt = get_last_checkpoint(str(phase_dir))
        return ckpt if ckpt else None
    return mode  # explicit path


def save_phase_complete(trainer: Seq2SeqTrainer, tokenizer, output_dir: Path, phase: int) -> None:
    complete_dir = phase_complete_dir(output_dir, phase)
    trainer.save_model(str(complete_dir))
    tokenizer.save_pretrained(complete_dir)
    phase_done_file(output_dir, phase).write_text("done\n", encoding="utf-8")
    logger.info(f"✅ Phase {phase} complete → {complete_dir}")


def save_emergency_checkpoint(trainer: Seq2SeqTrainer, tokenizer, output_dir: Path, phase: int) -> None:
    try:
        dst = output_dir / f"interrupted_phase_{phase}"
        trainer.save_model(str(dst))
        tokenizer.save_pretrained(dst)
        logger.warning(f"⚠ Emergency checkpoint saved → {dst}")
    except Exception as exc:
        logger.warning(f"Emergency save failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(base_model: str, cfg: dict, adapter_checkpoint: Optional[Path] = None):
    """
    Load mt5-base in fp16 — NO 8-bit quantisation needed.
    mt5-base fp16 = ~1.1 GB, well within RTX 4060 8 GB.
    """
    model_cfg  = cfg["model"]
    train_cfg  = cfg["training"]
    lora_cfg   = cfg["lora"]
    precision  = train_cfg.get("mixed_precision", "bf16")
    use_fp16   = precision == "fp16"
    use_bf16   = precision == "bf16"
    dtype      = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    logger.info(f"Loading {base_model}  dtype={dtype}  device=cuda:0")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).cuda()

    model.config.use_cache = False

    # enable_input_require_grads is needed when gradient checkpointing is on
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    task_type = lora_cfg.get("task_type", "SEQ_2_SEQ_LM")
    if isinstance(task_type, str):
        task_type = getattr(TaskType, task_type)

    if adapter_checkpoint is not None:
        logger.info(f"Loading LoRA adapter from {adapter_checkpoint}")
        model = PeftModel.from_pretrained(model, str(adapter_checkpoint), is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=task_type,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"),
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# VRAM monitor
# ─────────────────────────────────────────────────────────────────────────────
def log_vram(label: str = "") -> None:
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated()  / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM {label}: {used:.2f} GB / {total:.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# Collator with safety checks
# ─────────────────────────────────────────────────────────────────────────────
class SafeCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        batch.pop("decoder_inputs_embeds", None)
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# NaN guard trainer
# ─────────────────────────────────────────────────────────────────────────────
class NanGuardTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if "decoder_input_ids" not in inputs and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            inputs["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=labels)
        outputs = model(**inputs)
        logits  = outputs.logits.float()           # upcast to fp32 for stable loss
        loss    = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                      logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(
                "NaN/Inf loss detected. Try: lower lr, smaller batch, or disable fp16.")
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# Training arguments builder
# ─────────────────────────────────────────────────────────────────────────────
def make_training_args(
    cfg: dict,
    phase_output_dir: Path,
    phase: int,
    phase_epochs: int,
) -> Seq2SeqTrainingArguments:
    train_cfg = cfg["training"]
    opt_cfg   = cfg["optimiser"]
    model_cfg = cfg["model"]

    is_phase2        = phase == 2
    precision        = train_cfg.get("mixed_precision", "bf16")
    use_fp16         = precision == "fp16"
    use_bf16         = precision == "bf16"
    do_eval          = is_phase2
    load_best        = is_phase2 and bool(train_cfg.get("load_best_model_at_end", True))
    predict_generate = is_phase2 and bool(train_cfg.get("predict_with_generate", True))
    num_workers      = int(train_cfg.get("dataloader_num_workers", 2))

    # ── eval_strat must be defined BEFORE kwargs so save_strategy can use it ─
    params     = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_strat = train_cfg.get("eval_strategy", "epoch") if do_eval else "no"

    # When load_best_model_at_end=True the save and eval strategies must match.
    # Phase 1 has no eval so it always saves by steps (frequent = crash-safe).
    # Phase 2 aligns save_strategy with eval_strategy automatically.
    save_strategy = (
        eval_strat
        if (is_phase2 and load_best and eval_strat != "no")
        else train_cfg.get("save_strategy", "steps")
    )

    kwargs: dict[str, Any] = dict(
        output_dir                   = str(phase_output_dir),
        overwrite_output_dir         = False,
        num_train_epochs             = phase_epochs,
        # ── batch ────────────────────────────────────────────────────────────
        per_device_train_batch_size  = train_cfg["per_device_train_batch"],
        per_device_eval_batch_size   = train_cfg["per_device_eval_batch"],
        gradient_accumulation_steps  = train_cfg["gradient_accumulation"],
        # ── optimiser ────────────────────────────────────────────────────────
        learning_rate                = opt_cfg["lr"],
        weight_decay                 = opt_cfg["weight_decay"],
        lr_scheduler_type            = opt_cfg["lr_scheduler"],
        warmup_ratio                 = opt_cfg["warmup_ratio"],
        max_grad_norm                = opt_cfg.get("max_grad_norm", 1.0),
        optim                        = opt_cfg["type"],
        label_smoothing_factor       = opt_cfg.get("label_smoothing", 0.0),
        # ── precision ────────────────────────────────────────────────────────
        fp16                         = use_fp16,
        bf16                         = use_bf16,
        # ── gradient checkpointing ───────────────────────────────────────────
        gradient_checkpointing       = train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs= {"use_reentrant": False},
        # ── data loading ─────────────────────────────────────────────────────
        # num_workers=0 required on Windows (Python 3.12+): local closure from
        # enable_input_require_grads cannot be pickled by spawn multiprocessing.
        dataloader_num_workers        = num_workers,
        dataloader_pin_memory         = True,
        dataloader_persistent_workers = bool(num_workers > 0),
        # prefetch_factor is only valid when num_workers > 0
        dataloader_prefetch_factor    = (
            train_cfg.get('dataloader_prefetch_factor', 2)
            if num_workers > 0 else None
        ),
        # ── generation ───────────────────────────────────────────────────────
        predict_with_generate        = predict_generate,
        generation_max_length        = model_cfg["max_output_length"],
        generation_num_beams         = train_cfg.get("generation_num_beams", 1),
        # ── logging ──────────────────────────────────────────────────────────
        logging_strategy             = "steps",
        logging_steps                = train_cfg.get("logging_steps", 50),
        # ── checkpointing ────────────────────────────────────────────────────
        save_strategy                = save_strategy,   # aligned with eval for phase 2
        save_steps                   = train_cfg.get("save_steps", 200),  # phase 1 only
        save_total_limit             = train_cfg.get("save_total_limit", 3),
        # ── eval / best model ────────────────────────────────────────────────
        load_best_model_at_end       = load_best,
        metric_for_best_model        = train_cfg.get("metric_for_best_model", "rougeL") if load_best else None,
        greater_is_better            = True if load_best else None,
        # ── misc ─────────────────────────────────────────────────────────────
        group_by_length              = True,
        report_to                    = "none",
        seed                         = cfg.get("seed", 42),
        ddp_find_unused_parameters   = False,
        remove_unused_columns        = True,
    )

    # eval_strategy compat (older transformers uses evaluation_strategy)
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strat
    else:
        kwargs["evaluation_strategy"] = eval_strat

    # Drop None values — some older versions reject None for optional args
    kwargs = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return Seq2SeqTrainingArguments(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict, base_model_override: Optional[str] = None) -> None:
    setup_logging()
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"VRAM cleared. Free: "
                    f"{torch.cuda.mem_get_info()[0]/1e9:.1f} GB / "
                    f"{torch.cuda.mem_get_info()[1]/1e9:.1f} GB")

    paths     = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    curr_cfg  = cfg["curriculum"]

    base_model = base_model_override or model_cfg["base_model"]
    output_dir = Path(paths["models"]) / base_model.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the active config for reproducibility
    with open(output_dir / "active_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f"Output dir : {output_dir}")

    # ── Build training phase plan ─────────────────────────────────────────────
    if curr_cfg.get("enabled", False):
        p1_epochs = int(curr_cfg.get("phase1_epochs", 0))
        p2_epochs = int(curr_cfg.get("phase2_epochs",
                        max(int(train_cfg["epochs"]) - p1_epochs, 0)))
        phases = []
        if p1_epochs > 0:
            phases.append((1, p1_epochs, curr_cfg.get("phase1_langs", [])))
        if p2_epochs > 0:
            phases.append((2, p2_epochs, None))
    else:
        phases = [(2, int(train_cfg["epochs"]), None)]

    logger.info("Phase plan: " + ", ".join(f"phase {p}: {e} epoch(s)" for p, e, _ in phases))

    # ── Decide which adapter to load (resume or fresh) ────────────────────────
    adapter_checkpoint: Optional[Path] = None
    phase_resume: dict[int, Optional[str]] = {}
    for phase, _, _ in phases:
        phase_dir = output_dir / f"phase_{phase}"
        phase_resume[phase] = resolve_resume_checkpoint(phase_dir, train_cfg)

    if phase_resume.get(2):
        adapter_checkpoint = Path(phase_resume[2])
        logger.info(f"Resuming phase 2 from {adapter_checkpoint}")
    elif is_phase_complete(output_dir, 1):
        adapter_checkpoint = phase_complete_dir(output_dir, 1)
        logger.info(f"Phase 1 done — loading adapter from {adapter_checkpoint}")
    elif phase_resume.get(1):
        adapter_checkpoint = Path(phase_resume[1])
        logger.info(f"Resuming phase 1 from {adapter_checkpoint}")

    # ── Load tokenizer and model ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model     = build_model(base_model, cfg, adapter_checkpoint=adapter_checkpoint)
    log_vram("after model load")

    # ── Data files ───────────────────────────────────────────────────────────
    train_file = Path(paths["data_augmented"]) / "final_train.csv"
    val_file   = Path(paths["data_cleaned"])   / "val_clean.csv"
    for f in (train_file, val_file):
        if not f.exists():
            raise FileNotFoundError(f"Required data file not found: {f}")

    # ── Phase loop ───────────────────────────────────────────────────────────
    last_trainer = None
    completed: set[int] = set()

    for phase, phase_epochs, phase_langs in phases:
        # Skip completed phases
        if is_phase_complete(output_dir, phase):
            logger.info(f"Phase {phase} already complete — skipping.")
            completed.add(phase)
            continue
        if phase == 1 and phase_resume.get(2):
            logger.info("Skipping phase 1 — phase 2 checkpoint exists.")
            completed.add(phase)
            continue

        logger.info(f"─── Phase {phase} | {phase_epochs} epoch(s) ───")
        dataset = load_tokenized_dataset(
            train_file, val_file, phase, phase_langs, tokenizer, cfg)

        phase_dir     = output_dir / f"phase_{phase}"
        resume_ckpt   = phase_resume.get(phase)
        training_args = make_training_args(cfg, phase_dir, phase, phase_epochs)

        callbacks = []
        if (phase == 2
                and training_args.load_best_model_at_end
                and int(train_cfg.get("early_stopping_patience", 0)) > 0):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=int(train_cfg["early_stopping_patience"])))

        trainer = NanGuardTrainer(
            model          = model,
            args           = training_args,
            train_dataset  = dataset["train"],
            eval_dataset   = dataset["validation"] if training_args.do_eval else None,
            data_collator  = SafeCollator(
                tokenizer=tokenizer, model=model,
                label_pad_token_id=-100, pad_to_multiple_of=8),
            compute_metrics= make_compute_metrics(tokenizer) if training_args.predict_with_generate else None,
            callbacks      = callbacks,
        )

        try:
            trainer.train(resume_from_checkpoint=resume_ckpt)
        except (KeyboardInterrupt, SystemExit):
            logger.warning("Training interrupted — saving emergency checkpoint …")
            save_emergency_checkpoint(trainer, tokenizer, output_dir, phase)
            raise
        except FloatingPointError:
            save_emergency_checkpoint(trainer, tokenizer, output_dir, phase)
            raise

        save_phase_complete(trainer, tokenizer, output_dir, phase)
        log_vram(f"after phase {phase}")
        last_trainer = trainer
        completed.add(phase)

    # ── Final save ───────────────────────────────────────────────────────────
    final_dir = output_dir / "best"
    if last_trainer is not None:
        last_trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
        logger.info(f"✅ Final model saved → {final_dir}")
    elif phases and all(p in completed for p, _, _ in phases):
        # All phases were already complete from a previous run
        src = phase_complete_dir(output_dir, phases[-1][0])
        if adapter_files_exist(src):
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(src, final_dir)
            logger.info(f"✅ Copied completed adapter → {final_dir}")

    logger.info("Training finished.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mt5-base with LoRA")
    parser.add_argument("--config",     default="src/training/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--base_model", default=None,
                        help="Override base_model in config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config, args.base_model)