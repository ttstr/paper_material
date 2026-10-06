"""Switchable LLM backends: local transformers, vLLM, or an OpenAI-compatible HTTP API.

Selected by ``llm.backend`` in configs/llm/*.yaml. API keys are read ONLY from the
environment variable named by ``llm.api_key_env`` (never from config files).
Every call returns text + token usage + wall seconds for cost accounting.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLMBackend:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg["name"]
        self.max_tokens = int(cfg.get("max_tokens", 512))
        self.temperature = float(cfg.get("temperature", 0.0))

    def generate_batch(self, batch: list[list[dict]], max_tokens: int | None = None,
                       json_schema: dict | None = None) -> list[Generation]:
        raise NotImplementedError

    def decoding(self) -> dict:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "backend": self.name}


class TransformersBackend(LLMBackend):
    """Local HF causal LM, greedy decoding (temperature 0), optional left-padded batching."""

    name = "transformers"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if cfg.get("threads"):
            torch.set_num_threads(int(cfg["threads"]))
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            cfg.get("dtype", "float32")]
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self.model.eval()
        self.device = cfg.get("device", "cpu")
        self.model.to(self.device)

    def generate_batch(self, batch, max_tokens=None, json_schema=None):
        torch = self.torch
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(max_tokens or self.max_tokens), do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.pad_token_id)
        dt = time.time() - t0
        gens = []
        L = enc["input_ids"].shape[1]
        for i in range(len(batch)):
            new = out[i, L:]
            n_new = int((new != self.tok.pad_token_id).sum())
            gens.append(Generation(text=self.tok.decode(new, skip_special_tokens=True),
                                   prompt_tokens=int(enc["attention_mask"][i].sum()),
                                   completion_tokens=n_new, seconds=dt / len(batch)))
        return gens


class VLLMBackend(LLMBackend):
    """vLLM offline engine with JSON-schema guided decoding (GPU). Imported lazily."""

    name = "vllm"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from vllm import LLM  # noqa: WPS433  (optional dependency)

        self.llm = LLM(model=self.model_name, dtype=cfg.get("dtype", "auto"),
                       max_model_len=int(cfg.get("max_model_len", 4096)),
                       gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.9)))

    def generate_batch(self, batch, max_tokens=None, json_schema=None):
        from vllm import SamplingParams

        kw = dict(temperature=self.temperature, max_tokens=int(max_tokens or self.max_tokens))
        if json_schema is not None:
            try:  # vLLM >= 0.6
                from vllm.sampling_params import GuidedDecodingParams
                kw["guided_decoding"] = GuidedDecodingParams(json=json_schema)
            except Exception:  # pragma: no cover - version dependent
                pass
        t0 = time.time()
        outs = self.llm.chat(batch, SamplingParams(**kw), use_tqdm=False)
        dt = time.time() - t0
        return [Generation(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids), dt / len(batch))
                for o in outs]


class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible /chat/completions endpoint (vLLM server, hosted APIs)."""

    name = "openai"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        env = cfg.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(env, "")
        if not self.api_key and not cfg.get("allow_no_key", False):
            raise RuntimeError(f"set the API key in environment variable {env} (keys are never read from configs)")
        self.timeout = float(cfg.get("timeout", 120))

    def _one(self, messages, max_tokens, json_schema):
        body = {"model": self.model_name, "messages": messages, "temperature": self.temperature,
                "max_tokens": int(max_tokens or self.max_tokens)}
        if self.cfg.get("json_mode", True):
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read().decode())
        dt = time.time() - t0
        u = resp.get("usage", {}) or {}
        return Generation(resp["choices"][0]["message"]["content"] or "", int(u.get("prompt_tokens", 0)),
                          int(u.get("completion_tokens", 0)), dt)

    def generate_batch(self, batch, max_tokens=None, json_schema=None):
        return [self._one(m, max_tokens, json_schema) for m in batch]


class CachedOnlyBackend(LLMBackend):
    """Replays an existing call cache without loading any model (same name/decoding => same cache keys).
    Uncached jobs are skipped, never generated. Used to materialise partial runs and in tests."""

    cached_only = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.name = cfg.get("backend", "transformers")

    def generate_batch(self, batch, max_tokens=None, json_schema=None):
        raise RuntimeError("CachedOnlyBackend never generates")


BACKENDS = {"transformers": TransformersBackend, "vllm": VLLMBackend, "openai": OpenAICompatBackend}


def make_backend(cfg: dict) -> LLMBackend:
    return BACKENDS[cfg.get("backend", "transformers")](cfg)
