# LaunchForge AI — QLoRA Training & OOM Guide

Target hardware: **RTX 5050 (8 GB VRAM, Blackwell `sm_120`)**, **12 GB WSL2 RAM cap**.

---

## 1. Dry-Run (Verify Memory Fits First)

Runs **10 steps on 20 examples** using `configs/train.yaml`:

```bash
python -m src.train.sft --dry-run
```

Inspect `models/adapters/qwen3_4b_qlora_v1_dryrun/run_summary.json` for `peak_vram_gb`, `mean_step_sec`, and `tokens_per_sec`.

---

## 2. Full QLoRA Training & Adapter Merge

```bash
# Full training run
python -m src.train.sft

# Resume from latest/specific checkpoint if interrupted
python -m src.train.sft --resume-from-checkpoint models/adapters/qwen3_4b_qlora_v1/checkpoints/checkpoint-50

# Monitor TensorBoard
tensorboard --logdir models/adapters/qwen3_4b_qlora_v1/tb_logs

# Merge best adapter into bf16 sharded weights (memory-safe)
python -m src.train.merge
```

---

## 3. OOM Mitigation Order (8 GB VRAM / 12 GB RAM)

If CUDA OOM or WSL2 host-RAM pressure occurs, apply reductions in this exact order in `configs/train.yaml`:

1. **Reduce sequence length (`training.max_seq_length`)**:
   - `3072` -> `2048` -> `1536` (largest activation-memory savings).
2. **Reduce LoRA rank (`lora.r` and `lora.lora_alpha`)**:
   - `r: 16, lora_alpha: 32` -> `r: 8, lora_alpha: 16`.
3. **Switch to the 3B model (`model_name`)**:
   - Change `model_name` from `Qwen/Qwen3-4B-Instruct-2507` to `Qwen/Qwen2.5-Coder-3B-Instruct`.

Never load full-precision (`fp32`) weights on CPU during training or merging.
