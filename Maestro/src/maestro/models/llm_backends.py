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
    "openai": ("https://api.openai.com/v1", "gpt-5.6-sol", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", "QWEN_API_KEY"),
    "vllm": ("http://localhost:8000/v1", "", "LLM_API_KEY"),
    "openai-compat": ("", "", "LLM_API_KEY"),
}


def _is_reasoning_model(model: str) -> bool:
    """gpt-5.x / o-series reasoning models take DIFFERENT chat-completions
    params than gpt-4-era models: `max_completion_tokens` instead of
    `max_tokens` (which they REJECT with 400), and no custom `temperature`
    (only the default is supported). Detect by id prefix."""
    m = (model or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


# Transient transport failures (rate limits, gateway hiccups, dropped
# connections) must be retried HERE, at the transport layer — every agent
# shares this client, and a per-caller retry can't tell a garbage reply
# from a dead socket (incident 2026-08-06: two consecutive empty replies
# burned both camera-tree attempts and degraded the tree to flat).
# Mirrors WaveSpeedClient's poll/upload transient handling.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_TRANSIENT_BACKOFF_S = (2.0, 5.0)


def _post_with_transient_retry(url: str, payload: dict, headers: dict,
                               timeout: float, tag: str):
    import time as _time

    import requests

    last_exc: Exception | None = None
    for attempt in range(1 + len(_TRANSIENT_BACKOFF_S)):
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=timeout)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
            resp = None
        if resp is not None and resp.status_code not in _TRANSIENT_STATUS:
            return resp
        if attempt < len(_TRANSIENT_BACKOFF_S):
            import logging
            logging.getLogger("maestro.models.llm").warning(
                "LLM(%s) transient %s — retry %d/%d",
                tag,
                (f"HTTP {resp.status_code}" if resp is not None
                 else type(last_exc).__name__),
                attempt + 1, len(_TRANSIENT_BACKOFF_S))
            _time.sleep(_TRANSIENT_BACKOFF_S[attempt])
    if resp is not None:
        return resp                      # exhausted: surface the HTTP error
    raise RuntimeError(f"LLM({tag}) transport failed after "
                       f"{1 + len(_TRANSIENT_BACKOFF_S)} attempts: "
                       f"{last_exc}")


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
        # Reasoning models burn part of the completion budget on REASONING
        # tokens before any visible text — 1024 truncates the brain's JSON
        # mid-reply. Default higher for the gpt-5/o family.
        default_max = 4096 if _is_reasoning_model(self.model) else 1024
        self.max_tokens = int(self.config.get("max_tokens", default_max))

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

    def _payload(self, prompt: str, **kwargs) -> dict:
        """Chat-completions payload, per model family:
        • gpt-5.x / o-series → `max_completion_tokens`, NO temperature (they
          400 on `max_tokens` and on non-default temperature);
        • everything else    → classic `max_tokens` + `temperature`."""
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        limit = int(kwargs.get("max_tokens", self.max_tokens))
        if _is_reasoning_model(self.model):
            payload["max_completion_tokens"] = limit
        else:
            payload["max_tokens"] = limit
            payload["temperature"] = float(
                kwargs.get("temperature", self.temperature))
        return payload

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
        payload = self._payload(prompt, **kwargs)
        url = f"{self.base_url}/chat/completions"
        timeout = float(kwargs.get("timeout", 120))
        resp = _post_with_transient_retry(url, payload, headers, timeout,
                                          tag=f"{self.name}/{self.model}")
        if resp.status_code == 400:
            # Param-name mismatch safety net (proxies / older gateways):
            # if the server rejects the token-limit PARAM NAME, retry ONCE
            # with the other name (NEWTON does the same swap). Only for the
            # named-param complaint — any other 400 surfaces as-is.
            body = resp.text or ""
            swapped = None
            # Trigger on the body mentioning EITHER token-limit param name,
            # then swap based on WHICH ONE WE SENT ("max_tokens" is NOT a
            # substring of "max_completion_tokens" — direction can't be read
            # from the body alone).
            if "max_tokens" in body or "max_completion_tokens" in body:
                if "max_completion_tokens" in payload:
                    swapped = dict(payload)
                    swapped["max_tokens"] = swapped.pop("max_completion_tokens")
                elif "max_tokens" in payload:
                    swapped = dict(payload)
                    swapped["max_completion_tokens"] = swapped.pop("max_tokens")
                    swapped.pop("temperature", None)  # reasoning models reject it
            if swapped is not None:
                resp = _post_with_transient_retry(
                    url, swapped, headers, timeout,
                    tag=f"{self.name}/{self.model}")
        if resp.status_code >= 400:
            # Surface the API's own explanation — raise_for_status() drops the
            # body, which is where the server says WHICH field is wrong.
            raise RuntimeError(
                f"LLM('{self.name}' model={self.model}) HTTP "
                f"{resp.status_code}: {resp.text[:1000]}"
            )
        body = resp.json()
        choice = body["choices"][0]
        content = choice["message"]["content"]
        if not content:
            # Reasoning models share the completion budget between thinking
            # and visible text — a long think returns 200 with EMPTY content
            # (finish_reason=length). Name the cause so retries upstream
            # aren't misread as network flaps (incident 2026-08-06).
            import logging
            logging.getLogger("maestro.models.llm").warning(
                "LLM(%s/%s) returned EMPTY content (finish_reason=%s, "
                "usage=%s)", self.name, self.model,
                choice.get("finish_reason"), body.get("usage"))
        return content


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
class GeminiLLM(BaseLLMClient):
    """Gemini 原生 generateContent 的【纯文本】后端(2026-07-29 用户指示:
    可能把 brain 从 OpenAI 整体替换为 Gemini)。

    与 GeminiVLM 同一端点形状、同一 key、同一 $GEMINI_BASE_URL 覆盖 ——
    评审走得通的网络路径(含自定义中转),brain 同样走得通。切换只需:

      models.llm:
        name: "gemini"
        model: "gemini-3.5-pro"     # 或 $GEMINI_TEXT_MODEL 覆盖

    诚实约定:HTTP ≥400 直接抛(带响应体前 500 字,不静默);回复无
    text part 抛 —— 调用方(_brain_pick 等)自会记 usable=False。"""

    def __init__(self, name: str = "gemini", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.model = (self.config.get("model")
                      or os.getenv("GEMINI_TEXT_MODEL")
                      or "gemini-3.5-pro")
        self.base_url = (self.config.get("base_url")
                         or os.getenv("GEMINI_BASE_URL")
                         or "https://generativelanguage.googleapis.com"
                         ).rstrip("/")
        self.api_key = (self.config.get("api_key")
                        or os.getenv("GEMINI_API_KEY"))
        self.temperature = float(self.config.get("temperature", 0.7))
        self.max_tokens = int(self.config.get("max_tokens", 4096))

    def supports_function_calling(self) -> bool:
        return True

    def complete(self, prompt: str, **kwargs) -> str:
        import requests  # lazy — 与其余后端同款

        if not self.api_key:
            raise RuntimeError(
                f"GeminiLLM('{self.name}') needs an API key: set "
                f"models.llm.api_key or $GEMINI_API_KEY.")
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"role": "user",
                          "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(kwargs.get("temperature",
                                                self.temperature)),
                "maxOutputTokens": int(kwargs.get("max_tokens",
                                                  self.max_tokens)),
            },
        }
        headers = {"x-goog-api-key": self.api_key,
                   "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers,
                             timeout=float(kwargs.get("timeout", 120)))
        if resp.status_code >= 400:
            raise RuntimeError(
                f"GeminiLLM HTTP {resp.status_code}: {resp.text[:500]}")
        cands = resp.json().get("candidates") or []
        for part in reversed((cands[0].get("content") or {})
                             .get("parts", []) if cands else []):
            if part.get("text"):
                return str(part["text"])
        raise RuntimeError("GeminiLLM: no text part in response "
                           f"({str(resp.text)[:200]})")


_REGISTRY = {
    "openai": OpenAICompatLLM,
    "gpt": OpenAICompatLLM,
    "deepseek": OpenAICompatLLM,
    "qwen": OpenAICompatLLM,
    "vllm": OpenAICompatLLM,
    "openai-compat": OpenAICompatLLM,
    "gemini": GeminiLLM,
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
