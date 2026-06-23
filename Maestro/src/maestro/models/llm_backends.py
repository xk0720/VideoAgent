"""Real LLM backends (v0.4) — OpenAI-compatible + Anthropic, raw `requests`.

These implement the SAME `BaseLLMClient.complete(prompt, **kwargs) -> str`
contract as the mock, so the rest of the pipeline is unchanged — flip
`models.llm.name` in the config to a real backend and set the matching key.

Pattern (cite): like UniVA's (2511.08521) `utils/*_api.py`, we hit the HTTP
endpoint with raw `requests` and a lazy import, rather than pinning a vendor SDK
(no new hard dep — `openai`/`anthropic` SDKs are NOT required). This mirrors
video_gen_backends.WaveSpeedClient exactly: lazy `import requests` inside the
call, a loud RuntimeError on a missing key, a registry + `build_real_llm`
dispatch, and config-dict ⊕ env-var key resolution.

Honesty: a real backend selected without its API key fails LOUDLY at call time
(never silently degrades to a stub) — the only correct behavior when the config
claims a real model is wired but it cannot run.

Backends:
  • OpenAICompatLLM — OpenAI, DeepSeek, Qwen (DashScope OpenAI-compat), vLLM,
    and any other OpenAI-compatible /chat/completions endpoint.
  • AnthropicLLM    — Anthropic Messages API.
"""
from __future__ import annotations

import os
from typing import Optional

from .llm import BaseLLMClient

# Per-provider defaults: name → (base_url, model, env-var for the key).
# A generic OpenAI-compatible endpoint (vllm / openai-compat) requires the
# caller to supply base_url + model; its env fallback is the generic LLM_API_KEY.
_OPENAI_COMPAT_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "openai": ("https://api.openai.com/v1", "gpt-4o", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", "QWEN_API_KEY"),
    "vllm": ("http://localhost:8000/v1", "", "LLM_API_KEY"),
    "openai-compat": ("", "", "LLM_API_KEY"),
}


class OpenAICompatLLM(BaseLLMClient):
    """Any OpenAI-compatible chat-completions endpoint via raw `requests`.

    Covers OpenAI, DeepSeek, Qwen (DashScope OpenAI-compat mode), a local vLLM
    server, or any other endpoint exposing POST {base_url}/chat/completions.

    config:
      models.llm:
        name: "deepseek"          # or openai / qwen / vllm / openai-compat
        model: "deepseek-chat"     # provider-specific id
        api_key: ...               # or the provider env var (see below)
        base_url: ...              # required for vllm / openai-compat
        temperature: 0.7
        max_tokens: 1024

    Key resolution (config.api_key first, then env): the provider env var when
    the name is known (OPENAI_API_KEY / DEEPSEEK_API_KEY / QWEN_API_KEY), else
    the generic LLM_API_KEY. vLLM servers usually need no real key; a placeholder
    is sent so the OpenAI-compat header shape is preserved.
    """

    def __init__(self, name: str = "openai", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        key = name.split("-")[0].lower() if name else ""
        d_base, d_model, env_var = _OPENAI_COMPAT_DEFAULTS.get(
            name.lower(), _OPENAI_COMPAT_DEFAULTS.get(key, ("", "", "LLM_API_KEY"))
        )
        self.base_url = (
            self.config.get("base_url") or os.getenv("LLM_BASE_URL") or d_base
        ).rstrip("/")
        self.model = self.config.get("model", d_model)
        # provider env var first, then generic LLM_API_KEY fallback
        self.api_key = (
            self.config.get("api_key") or os.getenv(env_var) or os.getenv("LLM_API_KEY")
        )
        # vLLM is keyless by convention — keep a placeholder so headers are valid.
        self._key_optional = key == "vllm"
        self.temperature = float(self.config.get("temperature", 0.7))
        self.max_tokens = int(self.config.get("max_tokens", 1024))

    def supports_function_calling(self) -> bool:
        return True

    def _resolved_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self._key_optional:
            return "EMPTY"  # vLLM accepts any non-empty token
        raise RuntimeError(
            f"OpenAICompatLLM('{self.name}') needs an API key: set "
            f"models.llm.api_key or the provider env var, or switch "
            f"models.llm.name back to 'mock-llm'."
        )

    def complete(self, prompt: str, **kwargs) -> str:
        import requests  # lazy — no hard dep (mirrors WaveSpeedClient)

        if not self.base_url:
            raise RuntimeError(
                f"OpenAICompatLLM('{self.name}') needs a base_url (set "
                f"models.llm.base_url or $LLM_BASE_URL for vllm/openai-compat)."
            )
        headers = {
            "Authorization": f"Bearer {self._resolved_key()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers,
            timeout=float(kwargs.get("timeout", 120)),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicLLM(BaseLLMClient):
    """Anthropic Messages API via raw `requests` (no `anthropic` SDK dep).

    config:
      models.llm:
        name: "anthropic"          # or "claude"
        model: "claude-sonnet-4-6"  # current id; claude-opus-4-8 also valid
        api_key: ...               # or $ANTHROPIC_API_KEY
        max_tokens: 1024
    """

    BASE = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"

    def __init__(self, name: str = "anthropic", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.model = self.config.get("model", "claude-sonnet-4-6")
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.base_url = self.config.get("base_url", self.BASE)

    def supports_function_calling(self) -> bool:
        return True

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                f"AnthropicLLM('{self.name}') needs an API key: set "
                f"$ANTHROPIC_API_KEY or models.llm.api_key (or switch "
                f"models.llm.name back to 'mock-llm')."
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.VERSION,
            "Content-Type": "application/json",
        }

    def complete(self, prompt: str, **kwargs) -> str:
        import requests  # lazy — no hard dep

        payload = {
            "model": self.model,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(
            self.base_url, json=payload, headers=self._headers(),
            timeout=float(kwargs.get("timeout", 120)),
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        # Messages API returns a list of content blocks; concat the text ones.
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


# name (or its provider prefix) → backend class
_REGISTRY = {
    "openai": OpenAICompatLLM,
    "gpt": OpenAICompatLLM,
    "deepseek": OpenAICompatLLM,
    "qwen": OpenAICompatLLM,
    "vllm": OpenAICompatLLM,
    "openai-compat": OpenAICompatLLM,
    "anthropic": AnthropicLLM,
    "claude": AnthropicLLM,
}


def build_real_llm(name: str, config: Optional[dict] = None) -> BaseLLMClient:
    """Dispatch a real LLM backend by config name. Unknown → ValueError."""
    key = name.split("-")[0].lower() if name else ""
    cls = _REGISTRY.get(name.lower()) or _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"unknown llm backend '{name}'. known: {sorted(_REGISTRY)} (+ 'mock-llm')"
        )
    return cls(name=name, config=config)
