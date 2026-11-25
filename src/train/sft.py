"""QLoRA fine-tuning CLI (`python -m src.train.sft`) with Unsloth + TRL/PEFT fallback."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.train.masking import pad_sft_batch, tokenize_and_mask_assistant

if TYPE_CHECKING:
    import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_train_config(config_path: Path = DEFAULT_TRAIN_CONFIG) -> dict[str, Any]:
    """Load training YAML configuration."""
    return dict(yaml.safe_load(config_path.read_text(encoding="utf-8")))


def _load_chat_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chat dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _set_seed(seed: int) -> None:
    import torch  # type: ignore[import-not-found]

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any, str]:
    """Load 4-bit NF4 base + LoRA model via Unsloth, falling back to HF TRL/PEFT."""
    import torch  # type: ignore[import-not-found]

    model_name = cfg["model_name"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    max_seq_len = int(train_cfg["max_seq_length"])

    if train_cfg.get("use_unsloth", True):
        try:
            from unsloth import FastLanguageModel  # type: ignore[import-not-found]

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_len,
                dtype=torch.bfloat16,
                load_in_4bit=bool(quant_cfg["load_in_4bit"]),
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(lora_cfg["r"]),
                target_modules=list(lora_cfg["target_modules"]),
                lora_alpha=int(lora_cfg["lora_alpha"]),
                lora_dropout=float(lora_cfg["lora_dropout"]),
                bias=str(lora_cfg.get("bias", "none")),
                use_gradient_checkpointing="unsloth" if train_cfg["gradient_checkpointing"] else False,
                random_state=int(cfg["seed"]),
            )
            return model, tokenizer, "unsloth"
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Unsloth load failed ({exc}); falling back to HF Transformers + PEFT.")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=str(quant_cfg["quant_type"]),
        bnb_4bit_use_double_quant=bool(quant_cfg["use_double_quant"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer, "hf_peft"


def run_training(
    config_path: Path = DEFAULT_TRAIN_CONFIG,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Execute QLoRA SFT training and write adapter + run_summary.json."""
    import torch  # type: ignore[import-not-found]
    from datasets import Dataset  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    cfg = load_train_config(config_path)
    _set_seed(int(cfg["seed"]))

    train_cfg = cfg["training"]
    run_name = f"{cfg['run_name']}_dryrun" if dry_run else cfg["run_name"]
    adapter_out = _resolve(cfg["adapters_dir"]) / run_name
    adapter_out.mkdir(parents=True, exist_ok=True)
    reports_dir = _resolve(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    max_examples = int(train_cfg["dry_run_max_examples"]) if dry_run else None
    train_rows = _load_chat_jsonl(_resolve(cfg["train_chat_jsonl"]), limit=max_examples)
    val_rows = _load_chat_jsonl(_resolve(cfg["val_chat_jsonl"]), limit=max_examples)

    model, tokenizer, backend_used = _load_model_and_tokenizer(cfg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_seq_len = int(train_cfg["max_seq_length"])
    train_features = [tokenize_and_mask_assistant(r["messages"], tokenizer, max_seq_len) for r in train_rows]
    val_features = [tokenize_and_mask_assistant(r["messages"], tokenizer, max_seq_len) for r in val_rows]

    train_ds = Dataset.from_list(train_features)
    val_ds = Dataset.from_list(val_features)

    total_tokens = sum(len(f["input_ids"]) for f in train_features)
    report_to = ["tensorboard"]
    if train_cfg.get("wandb_enabled", False):
        os.environ.setdefault("WANDB_PROJECT", str(train_cfg.get("wandb_project", "launchforge-ai")))
        report_to.append("wandb")

    eval_steps = 5 if dry_run else int(train_cfg["eval_steps"])
    save_steps = 5 if dry_run else int(train_cfg["save_steps"])

    args = TrainingArguments(
        output_dir=str(adapter_out / "checkpoints"),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["dry_run_max_steps"]) if dry_run else -1,
        optim=str(train_cfg["optim"]),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=1 if dry_run else int(train_cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        logging_dir=str(adapter_out / "tb_logs"),
        seed=int(cfg["seed"]),
        remove_unused_columns=False,
    )

    step_times: list[float] = []

    class _PerfCallback(TrainerCallback):
        def __init__(self) -> None:
            self._step_t0 = 0.0

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._step_t0 = time.perf_counter()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step_times.append(time.perf_counter() - self._step_t0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda feats: pad_sft_batch(feats, tokenizer.pad_token_id),
        callbacks=[
            _PerfCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=int(train_cfg["early_stopping_patience"])
            ),
        ],
    )

    t0 = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.perf_counter() - t0

    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if torch.cuda.is_available()
        else 0.0
    )
    mean_step_sec = round(sum(step_times) / len(step_times), 4) if step_times else 0.0
    tokens_per_sec = round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0

    summary = {
        "run_name": run_name,
        "backend": backend_used,
        "model_name": cfg["model_name"],
        "dry_run": dry_run,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_loss": round(float(train_result.training_loss), 4),
        "elapsed_sec": round(elapsed, 2),
        "mean_step_sec": mean_step_sec,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_gb": peak_vram_gb,
        "adapter_dir": str(adapter_out),
    }
    summary_path = adapter_out / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (reports_dir / f"{run_name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="sft",
        help="QLoRA fine-tuning for LaunchForge AI (Unsloth + TRL/PEFT fallback).",
        add_completion=False,
    )

    @cli.command("train")
    def train_cmd(
        config: Path = typer.Option(DEFAULT_TRAIN_CONFIG, "--config", "-c"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Train 10 steps on 20 examples."),
        resume: str | None = typer.Option(
            None, "--resume-from-checkpoint", help="Checkpoint path or 'auto'."
        ),
    ) -> None:
        """Run QLoRA SFT training."""
        summary = run_training(
            config_path=config,
            dry_run=dry_run,
            resume_from_checkpoint=resume,
        )
        typer.echo(json.dumps(summary, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
