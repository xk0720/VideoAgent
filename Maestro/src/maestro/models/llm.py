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
    """Factory. v0.1 always returns a mock; v0.2 dispatches on spec['backend']."""
    name = "mock-llm"
    if isinstance(spec, dict):
        name = spec.get("name", name)
    elif isinstance(spec, str):
        name = spec
    # DESIGN_DECISION: real backends (deepseek/openai/anthropic/vllm) plug in here.
    return MockLLMClient(name=name)
