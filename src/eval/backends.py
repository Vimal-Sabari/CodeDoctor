"""Model-agnostic inference backends for evaluation (HF 4-bit, llama.cpp server, Mock)."""

from __future__ import annotations

import json
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable


class ModelBackend(ABC):
    """Abstract inference backend returning (raw_response_text, latency_seconds)."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        """Generate a single response (batch_size=1) and measure latency."""


class MockBackend(ModelBackend):
    """Deterministic mock backend for offline unit tests."""

    def __init__(
        self,
        responder: str | Callable[[str, str], str],
        fixed_latency_sec: float = 0.01,
    ) -> None:
        self._responder = responder
        self._fixed_latency_sec = fixed_latency_sec

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        if callable(self._responder):
            return self._responder(system_prompt, user_prompt), self._fixed_latency_sec
        return self._responder, self._fixed_latency_sec


class LlamaServerBackend(ModelBackend):
    """OpenAI-compatible HTTP backend targeting local llama.cpp llama-server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model_id: str = "local-gguf",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        timeout_sec: int = 120,
        use_json_schema: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout_sec = timeout_sec
        self.use_json_schema = use_json_schema

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        if self.use_json_schema and json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ModelOutput", "schema": json_schema},
            }

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency = time.perf_counter() - t0
        content = body["choices"][0]["message"]["content"]
        return content, latency


class HF4BitBackend(ModelBackend):
    """Memory-safe 4-bit NF4 Hugging Face Transformers backend (batch_size=1)."""

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self._torch = torch
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
            )
        latency = time.perf_counter() - t0
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, latency
