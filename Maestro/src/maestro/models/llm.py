"""LLM wrapper. v0.1 MockLLMClient returns deterministic stub text."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        ...

    def supports_function_calling(self) -> bool:
        return False


class MockLLMClient(BaseLLMClient):
    """Echo-style stub. Agents in v0.1 compute structured outputs deterministically
    in Python; this client exists so every reasoning step is still logged and so
    v0.2 can swap in a real model that actually parses these prompts."""

    def __init__(self, name: str = "mock-llm"):
        self.name = name

    def complete(self, prompt: str, **kwargs) -> str:
        head = prompt.strip().splitlines()[0] if prompt.strip() else ""
        return f"[{self.name}] ack: {head[:80]}"


def build_llm(spec: str | dict | None) -> BaseLLMClient:
    """Factory: name None or 'mock*' → MockLLMClient; else a real backend.

    Real backends (openai / deepseek / qwen / vllm / openai-compat / anthropic)
    live in llm_backends and are imported lazily so the mock/smoke path needs no
    `requests` and no keys. Mock stays the default — every existing test and the
    `maestro smoke` path is untouched."""
    name = "mock-llm"
    config: dict | None = None
    if isinstance(spec, dict):
        name = spec.get("name", name)
        config = spec
    elif isinstance(spec, str):
        name = spec
    if name is None or name.lower().startswith("mock"):
        return MockLLMClient(name=name or "mock-llm")
    from .llm_backends import build_real_llm
    return build_real_llm(name, config)
