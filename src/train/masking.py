"""Assistant-only token masking utilities for chat SFT training."""

from __future__ import annotations

from typing import Any, Protocol


class ChatTokenizer(Protocol):
    """Minimal tokenizer protocol required for assistant-only label masking."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render chat messages into a single prompt string."""

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 3072,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        """Tokenize rendered text into `input_ids` and `attention_mask`."""


def tokenize_and_mask_assistant(
    messages: list[dict[str, str]],
    tokenizer: ChatTokenizer,
    max_seq_length: int = 3072,
) -> dict[str, list[int]]:
    """Tokenize chat messages and mask system/user tokens with -100 in labels."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Expected chat messages ending with an 'assistant' turn.")

    prefix_messages = messages[:-1]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    prefix_ids = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )["input_ids"]
    full_encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    input_ids = list(full_encoded["input_ids"])
    attention_mask = list(full_encoded.get("attention_mask", [1] * len(input_ids)))

    prefix_len = min(len(prefix_ids), len(input_ids))
    labels = [-100] * prefix_len + input_ids[prefix_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def pad_sft_batch(
    features: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    """Pad a batch of masked SFT examples for causal LM training."""
    import torch  # type: ignore[import-not-found]

    max_len = max(len(f["input_ids"]) for f in features)
    batch_input_ids: list[list[int]] = []
    batch_attention: list[list[int]] = []
    batch_labels: list[list[int]] = []

    for feat in features:
        pad_len = max_len - len(feat["input_ids"])
        batch_input_ids.append(feat["input_ids"] + [pad_token_id] * pad_len)
        batch_attention.append(feat["attention_mask"] + [0] * pad_len)
        batch_labels.append(feat["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }
