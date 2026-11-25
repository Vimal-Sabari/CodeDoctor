"""Evaluation runner CLI: `python -m src.eval.run_eval`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.eval.backends import HF4BitBackend, LlamaServerBackend, ModelBackend
from src.eval.metrics import compute_metrics

if TYPE_CHECKING:
    import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "eval.yaml"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation split not found: {path}")
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_backend(model_key: str, cfg: dict[str, Any]) -> ModelBackend:
    """Instantiate the configured backend for `model_key`."""
    baselines = cfg["baselines"]
    if model_key not in baselines:
        available = ", ".join(sorted(baselines.keys()))
        raise KeyError(f"Unknown model '{model_key}'. Available: {available}")

    spec = baselines[model_key]
    inf = cfg["inference"]
    backend_type = spec["backend"]

    if backend_type == "hf_4bit":
        return HF4BitBackend(
            model_id=spec["model_id"],
            temperature=float(inf["temperature"]),
            max_new_tokens=int(inf["max_new_tokens"]),
        )
    if backend_type == "llama_server":
        return LlamaServerBackend(
            base_url=spec["base_url"],
            model_id=spec["model_id"],
            temperature=float(inf["temperature"]),
            max_new_tokens=int(inf["max_new_tokens"]),
            use_json_schema=bool(spec.get("use_json_schema", True)),
        )
    raise ValueError(f"Unsupported backend type: {backend_type}")


def format_metrics_markdown(model_name: str, split: str, metrics: dict[str, Any]) -> str:
    """Render a concise markdown summary table of evaluation metrics."""
    lines = [
        f"### Evaluation: `{model_name}` on `{split}` (n={metrics.get('total', 0)})",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Category Accuracy | {metrics.get('category_accuracy', 0.0):.4f} |",
        f"| Macro-F1 | {metrics.get('macro_f1', 0.0):.4f} |",
        f"| JSON Validity Rate | {metrics.get('json_validity_rate', 0.0):.4f} |",
        f"| Root-Cause Entity Match | {metrics.get('root_cause_entity_match', 0.0):.4f} |",
        f"| Hallucination Rate | {metrics.get('hallucination_rate', 0.0):.4f} |",
        f"| Unknown When Wrong | {metrics.get('unknown_when_wrong_rate', 0.0):.4f} |",
        f"| Mean Latency (s) | {metrics.get('latency', {}).get('mean_sec', 0.0):.4f} |",
    ]
    return "\n".join(lines)


def run_evaluation(
    config_path: Path = DEFAULT_EVAL_CONFIG,
    model_key: str = "qwen3_4b_instruct",
    split: str = "gold",
    rag: bool = False,
    limit: int | None = None,
    backend_override: ModelBackend | None = None,
) -> dict[str, Any]:
    """Run batch_size=1 evaluation on `split` and save predictions + metrics JSON."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompts = yaml.safe_load(_resolve(cfg["prompts_path"]).read_text(encoding="utf-8"))
    schema = json.loads(_resolve(cfg["schema_path"]).read_text(encoding="utf-8"))

    split_path = _resolve(cfg["splits"][split])
    records = _load_jsonl(split_path)
    if limit is not None:
        records = records[:limit]

    backend = backend_override or build_backend(model_key, cfg)
    system_prompt = prompts["system"].strip()

    raw_predictions: list[str] = []
    latencies: list[float] = []
    reports_dir = _resolve(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"{model_key}_{split}_{'rag' if rag else 'norag'}"
    preds_path = reports_dir / f"{run_tag}_predictions.jsonl"
    metrics_path = reports_dir / f"{run_tag}_metrics.json"

    with preds_path.open("w", encoding="utf-8") as pred_fh:
        for rec in records:
            if rag and "rag_context" in rec:
                user_prompt = prompts["user_with_rag"].format(
                    framework=rec.get("framework", "unknown"),
                    package_manager=rec.get("package_manager", "unknown"),
                    rag_context=rec["rag_context"],
                    log_excerpt=rec["log_excerpt"],
                )
            else:
                user_prompt = prompts["user_no_rag"].format(
                    framework=rec.get("framework", "unknown"),
                    package_manager=rec.get("package_manager", "unknown"),
                    log_excerpt=rec["log_excerpt"],
                )

            raw_out, latency = backend.generate(system_prompt, user_prompt, json_schema=schema)
            raw_predictions.append(raw_out)
            latencies.append(latency)
            pred_fh.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "gold_category": rec["error_category"],
                        "raw_prediction": raw_out,
                        "latency_sec": round(latency, 4),
                    }
                )
                + "\n"
            )

    metrics = compute_metrics(records, raw_predictions, latencies)
    payload = {
        "model": model_key,
        "split": split,
        "rag": rag,
        "metrics": metrics,
        "predictions_path": str(preds_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        **payload,
        "metrics_path": str(metrics_path),
        "markdown_table": format_metrics_markdown(model_key, split, metrics),
    }


def get_cli_app() -> "typer.Typer":
    """Build and return the Typer CLI application."""
    import typer

    cli = typer.Typer(
        name="eval",
        help="Run zero-shot or fine-tuned evaluation on gold/test splits.",
        add_completion=False,
    )

    @cli.command("run")
    def run_cmd(
        config: Path = typer.Option(DEFAULT_EVAL_CONFIG, "--config", "-c"),
        model: str = typer.Option("qwen3_4b_instruct", "--model", "-m"),
        split: str = typer.Option("gold", "--split", "-s"),
        rag: bool = typer.Option(False, "--rag/--no-rag"),
        limit: int | None = typer.Option(None, "--limit", "-n"),
    ) -> None:
        """Run evaluation and print a markdown metrics table."""
        res = run_evaluation(
            config_path=config,
            model_key=model,
            split=split,
            rag=rag,
            limit=limit,
        )
        typer.echo(res["markdown_table"])
        typer.echo(f"\nSaved predictions: {res['predictions_path']}")
        typer.echo(f"Saved metrics    : {res['metrics_path']}")

    return cli


if __name__ == "__main__":
    get_cli_app()()
