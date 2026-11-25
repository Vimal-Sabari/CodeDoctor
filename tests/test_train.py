"""Tests for Phase 8 QLoRA config and assistant-only token masking."""

from __future__ import annotations

from pathlib import Path

from src.train.masking import tokenize_and_mask_assistant
from src.train.sft import DEFAULT_TRAIN_CONFIG, load_train_config


class FakeChatTokenizer:
    """Deterministic character-level chat tokenizer for verifying label masking."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 3072,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        ids = [ord(ch) for ch in text]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_assistant_only_token_masking() -> None:
    """Verify system and user tokens are masked (-100) and only assistant tokens are trained."""
    tok = FakeChatTokenizer()
    assistant_json = '{"error_category":"dependency_missing"}'
    messages = [
        {"role": "system", "content": "You are LaunchForge AI."},
        {"role": "user", "content": "Error: Cannot find module 'react'"},
        {"role": "assistant", "content": assistant_json},
    ]

    encoded = tokenize_and_mask_assistant(messages, tok, max_seq_length=3072)
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]

    assert len(input_ids) == len(labels)

    prefix_text = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prefix_len = len(prefix_text)

    # All system + user + <assistant> header tokens must be masked with -100
    assert all(lbl == -100 for lbl in labels[:prefix_len])

    # Unmasked tokens must decode strictly to the assistant payload + closing tag
    unmasked_ids = [lbl for lbl in labels if lbl != -100]
    decoded_unmasked = "".join(chr(i) for i in unmasked_ids)
    assert decoded_unmasked == f"{assistant_json}</assistant>"


def test_train_config_hyperparameters() -> None:
    """Verify configs/train.yaml matches RTX 5050 8GB QLoRA constraints."""
    cfg = load_train_config(Path(DEFAULT_TRAIN_CONFIG))
    assert cfg["quantization"]["load_in_4bit"] is True
    assert cfg["quantization"]["quant_type"] == "nf4"
    assert cfg["quantization"]["use_double_quant"] is True
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["lora_alpha"] == 32
    assert cfg["lora"]["lora_dropout"] == 0.05
    assert len(cfg["lora"]["target_modules"]) == 7

    tr = cfg["training"]
    assert tr["max_seq_length"] == 3072
    assert tr["per_device_train_batch_size"] == 1
    assert tr["gradient_accumulation_steps"] == 16
    assert tr["optim"] == "paged_adamw_8bit"
    assert tr["packing"] is False
    assert tr["gradient_checkpointing"] is True
    assert tr["warmup_ratio"] == 0.03
