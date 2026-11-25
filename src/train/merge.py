"""Memory-safe LoRA adapter merge CLI (`python -m src.train.merge`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def merge_adapter(
    config_path: Path = DEFAULT_TRAIN_CONFIG,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Merge LoRA adapter into bf16/fp16 base weights without loading fp32 on CPU."""
    import torch  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    active_run = run_name or cfg["run_name"]
    adapter_dir = _resolve(cfg["adapters_dir"]) / active_run
    merged_dir = _resolve(cfg["merged_dir"]) / active_run
    merged_dir.mkdir(parents=True, exist_ok=True)

    merge_cfg = cfg.get("merge", {})
    dtype_str = str(merge_cfg.get("torch_dtype", "bfloat16"))
    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    max_shard_size = str(merge_cfg.get("max_shard_size", "2GB"))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(
        str(merged_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(str(merged_dir))

    return {
        "run_name": active_run,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "torch_dtype": dtype_str,
    }


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="merge",
        help="Merge a trained LoRA adapter into base weights (memory-safe bf16/fp16).",
        add_completion=False,
    )

    @cli.command("merge")
    def merge_cmd(
        config: Path = typer.Option(DEFAULT_TRAIN_CONFIG, "--config", "-c"),
        run_name: str | None = typer.Option(None, "--run-name", "-r"),
    ) -> None:
        """Merge adapter and save sharded safetensors to models/merged/<run>."""
        res = merge_adapter(config_path=config, run_name=run_name)
        typer.echo(json.dumps(res, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
