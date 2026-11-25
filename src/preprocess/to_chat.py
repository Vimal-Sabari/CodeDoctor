"""Convert processed DatasetRecord JSONL splits into no-RAG chat SFT JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.schema import DatasetRecord, ModelOutput

if TYPE_CHECKING:
    import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PREPROCESS_CONFIG = PROJECT_ROOT / "configs" / "preprocess.yaml"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def record_to_chat_example(
    raw_record: dict[str, Any],
    prompts: dict[str, str],
) -> dict[str, Any]:
    """Convert a DatasetRecord dict into a system/user/assistant chat SFT dict (no-RAG)."""
    rec = DatasetRecord.model_validate(raw_record)
    user_content = prompts["user_no_rag"].format(
        framework=rec.framework,
        package_manager=rec.package_manager,
        log_excerpt=rec.log_excerpt,
    )
    target = ModelOutput(
        error_category=rec.error_category,
        root_cause=rec.root_cause,
        evidence=rec.evidence_lines,
        fix_steps=rec.fix_steps,
        confidence=rec.confidence,
    )
    return {
        "id": rec.id,
        "messages": [
            {"role": "system", "content": prompts["system"].strip()},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": target.model_dump_json()},
        ],
    }


def convert_jsonl_to_chat(
    input_jsonl: Path,
    output_jsonl: Path,
    prompts_path: Path,
) -> int:
    """Convert all records in `input_jsonl` to chat-format JSONL at `output_jsonl`."""
    if not input_jsonl.exists():
        return 0

    prompts: dict[str, str] = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with (
        input_jsonl.open("r", encoding="utf-8") as src,
        output_jsonl.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            line = line.strip()
            if not line:
                continue
            chat_obj = record_to_chat_example(json.loads(line), prompts)
            dst.write(json.dumps(chat_obj) + "\n")
            count += 1
    return count


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="to-chat",
        help="Convert train/val JSONL splits to no-RAG chat SFT JSONL.",
        add_completion=False,
    )

    @cli.command("convert")
    def convert_cmd(
        config: Path = typer.Option(DEFAULT_PREPROCESS_CONFIG, "--config", "-c"),
    ) -> None:
        """Convert train and val splits into chat-format SFT JSONL."""
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        outputs = cfg["outputs"]
        chat_cfg = cfg["chat"]
        prompts_path = _resolve(chat_cfg["prompts_path"])

        train_cnt = convert_jsonl_to_chat(
            _resolve(outputs["train_jsonl"]),
            _resolve(chat_cfg["train_chat_jsonl"]),
            prompts_path,
        )
        val_cnt = convert_jsonl_to_chat(
            _resolve(outputs["val_jsonl"]),
            _resolve(chat_cfg["val_chat_jsonl"]),
            prompts_path,
        )
        typer.echo(json.dumps({"train_chat": train_cnt, "val_chat": val_cnt}, indent=2))

    return cli


if __name__ == "__main__":
    get_cli_app()()
