"""Environment verification script for LaunchForge AI (WSL2 + RTX 5050 sm_120)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import psutil
import typer

app = typer.Typer(
    name="check-env",
    help="Verify Python, RAM, CUDA 12.8+, Blackwell sm_120 GPU, bitsandbytes 4-bit, and Unsloth.",
    add_completion=False,
)

MIN_PYTHON = (3, 11)
EXPECTED_COMPUTE_CAPABILITY = (12, 0)
BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single environment check."""

    name: str
    passed: bool
    detail: str
    fix_hint: str = ""


def _check_python() -> CheckResult:
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) < MIN_PYTHON:
        return CheckResult(
            name="Python",
            passed=False,
            detail=f"{version_str} (expected >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
            fix_hint=(
                "Create a Python 3.11+ virtual environment in WSL2: "
                "python3.11 -m venv .venv && source .venv/bin/activate"
            ),
        )
    return CheckResult(name="Python", passed=True, detail=version_str)


def _check_ram() -> CheckResult:
    vm = psutil.virtual_memory()
    total_gib = vm.total / BYTES_PER_GIB
    avail_gib = vm.available / BYTES_PER_GIB
    detail = f"{avail_gib:.2f} GiB free / {total_gib:.2f} GiB total"
    if total_gib < 8.0:
        return CheckResult(
            name="System RAM",
            passed=False,
            detail=detail,
            fix_hint=(
                "WSL2 has < 8 GiB RAM allocated. Set memory=12GB in "
                "%USERPROFILE%\\.wslconfig and restart WSL (`wsl --shutdown`)."
            ),
        )
    return CheckResult(name="System RAM", passed=True, detail=detail)


def _check_torch_and_cuda() -> tuple[list[CheckResult], object | None]:
    results: list[CheckResult] = []
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        results.append(
            CheckResult(
                name="PyTorch",
                passed=False,
                detail=f"Import failed: {exc}",
                fix_hint=(
                    "Install PyTorch CUDA 12.8+ build: "
                    "pip install torch torchvision torchaudio "
                    "--index-url https://download.pytorch.org/whl/cu128"
                ),
            )
        )
        return results, None

    cuda_ver = torch.version.cuda or "none"
    results.append(
        CheckResult(
            name="PyTorch",
            passed=True,
            detail=f"{torch.__version__} (CUDA build: {cuda_ver})",
        )
    )

    if not torch.cuda.is_available():
        results.append(
            CheckResult(
                name="CUDA Availability",
                passed=False,
                detail="torch.cuda.is_available() is False",
                fix_hint=(
                    "Ensure NVIDIA Windows driver supports CUDA 12.8+ (WSL2 passes through the "
                    "Windows GPU driver) and you installed the cu128 PyTorch wheel, not CPU-only."
                ),
            )
        )
        return results, torch

    results.append(CheckResult(name="CUDA Availability", passed=True, detail="available"))

    gpu_name = torch.cuda.get_device_name(0)
    results.append(CheckResult(name="GPU", passed=True, detail=gpu_name))

    cap = torch.cuda.get_device_capability(0)
    cap_str = f"{cap[0]}.{cap[1]}"
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    has_sm120 = not arch_list or any("120" in arch for arch in arch_list)

    if cap != EXPECTED_COMPUTE_CAPABILITY:
        results.append(
            CheckResult(
                name="Compute Capability",
                passed=False,
                detail=f"{cap_str} (expected {EXPECTED_COMPUTE_CAPABILITY[0]}.{EXPECTED_COMPUTE_CAPABILITY[1]} for RTX 5050 Blackwell)",
                fix_hint=(
                    "Verify the active GPU is the RTX 5050 (sm_120). If multiple GPUs exist, "
                    "set CUDA_VISIBLE_DEVICES to the RTX 5050 index."
                ),
            )
        )
    elif not has_sm120:
        results.append(
            CheckResult(
                name="Compute Capability",
                passed=False,
                detail=f"{cap_str}, but PyTorch arch list lacks sm_120 ({arch_list})",
                fix_hint=(
                    "Wrong CUDA wheel for Blackwell (sm_120). Reinstall PyTorch with CUDA 12.8+: "
                    "pip install --force-reinstall torch torchvision torchaudio "
                    "--index-url https://download.pytorch.org/whl/cu128"
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                name="Compute Capability",
                passed=True,
                detail=f"{cap_str} (sm_120 supported)",
            )
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / BYTES_PER_GIB
    total_gib = total_bytes / BYTES_PER_GIB
    vram_detail = f"{free_gib:.2f} GiB free / {total_gib:.2f} GiB total"
    if free_gib < 2.0:
        results.append(
            CheckResult(
                name="VRAM",
                passed=False,
                detail=vram_detail,
                fix_hint="Close GPU-heavy Windows/WSL processes to free at least 2 GiB VRAM.",
            )
        )
    else:
        results.append(CheckResult(name="VRAM", passed=True, detail=vram_detail))

    return results, torch


def _check_bitsandbytes(torch_mod: object | None) -> CheckResult:
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError as exc:
        return CheckResult(
            name="bitsandbytes (4-bit NF4)",
            passed=False,
            detail=f"Import failed: {exc}",
            fix_hint="Install bitsandbytes in your training env: pip install -e '.[train]'",
        )

    if torch_mod is None or not torch_mod.cuda.is_available():  # type: ignore[attr-defined]
        return CheckResult(
            name="bitsandbytes (4-bit NF4)",
            passed=False,
            detail=f"Imported v{getattr(bnb, '__version__', 'unknown')}, but CUDA is unavailable",
            fix_hint="Fix CUDA availability first so the 4-bit NF4 kernel test can run on GPU.",
        )

    try:
        layer = bnb.nn.Linear4bit(
            16,
            16,
            bias=False,
            compute_dtype=torch_mod.float16,  # type: ignore[attr-defined]
            quant_type="nf4",
        ).to("cuda")
        x = torch_mod.randn(1, 16, device="cuda", dtype=torch_mod.float16)  # type: ignore[attr-defined]
        y = layer(x)
        torch_mod.cuda.synchronize()  # type: ignore[attr-defined]
        del layer, x, y
        torch_mod.cuda.empty_cache()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="bitsandbytes (4-bit NF4)",
            passed=False,
            detail=f"4-bit CUDA load/forward test failed: {exc}",
            fix_hint=(
                "Upgrade bitsandbytes (>=0.45.0) built for CUDA 12.8+ / Blackwell sm_120, "
                "or verify LD_LIBRARY_PATH includes your CUDA/torch libs."
            ),
        )

    bnb_ver = getattr(bnb, "__version__", "unknown")
    return CheckResult(
        name="bitsandbytes (4-bit NF4)",
        passed=True,
        detail=f"v{bnb_ver} (4-bit NF4 CUDA forward pass OK)",
    )


def _check_unsloth() -> CheckResult:
    try:
        import unsloth  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="Unsloth",
            passed=False,
            detail=f"Import failed: {exc}",
            fix_hint=(
                "Install Unsloth after PyTorch cu128: "
                "pip install --no-deps unsloth && pip install -e '.[train]'"
            ),
        )

    ver = getattr(unsloth, "__version__", "installed")
    return CheckResult(name="Unsloth", passed=True, detail=str(ver))


@app.command()
def main() -> None:
    """Run all environment checks and exit non-zero if any check fails."""
    typer.echo("=== LaunchForge AI Environment Check ===")

    checks: list[CheckResult] = []
    simple_checks: list[Callable[[], CheckResult]] = [_check_python, _check_ram]
    for fn in simple_checks:
        checks.append(fn())

    torch_results, torch_mod = _check_torch_and_cuda()
    checks.extend(torch_results)
    checks.append(_check_bitsandbytes(torch_mod))
    checks.append(_check_unsloth())

    failed = False
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        typer.echo(f"[{status}] {result.name}: {result.detail}")
        if not result.passed:
            failed = True
            if result.fix_hint:
                typer.echo(f"       Fix: {result.fix_hint}")

    if failed:
        typer.echo("\nEnvironment check failed. Resolve the items marked [FAIL] above.")
        raise typer.Exit(code=1)

    typer.echo("\nAll environment checks passed.")


if __name__ == "__main__":
    app()
