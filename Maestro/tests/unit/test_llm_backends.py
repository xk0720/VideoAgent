"""LLM-backend factory + graceful/loud degradation (v0.4).

CPU-only, NO network: construction is lazy (no `requests`, no key needed), and
the loud-on-missing-key guard fires at `complete()` time. We never hit a real
endpoint — the key check raises before any POST.
"""
from __future__ import annotations

import pytest

from maestro.models.llm import MockLLMClient, build_llm


# ── factory dispatch ──
def test_factory_returns_mock_by_default():
    assert isinstance(build_llm(None), MockLLMClient)
    assert isinstance(build_llm("mock-llm"), MockLLMClient)
    assert isinstance(build_llm({"name": "mock-llm"}), MockLLMClient)


def test_factory_dispatches_real_backends_without_io():
    """Constructing a real backend must NOT do network I/O (lazy requests)."""
    for name, cls in [
        ("openai", "OpenAICompatLLM"),
        ("gpt-4o", "OpenAICompatLLM"),
        ("deepseek", "OpenAICompatLLM"),
        ("qwen", "OpenAICompatLLM"),
        ("vllm", "OpenAICompatLLM"),
        ("openai-compat", "OpenAICompatLLM"),
        ("anthropic", "AnthropicLLM"),
        ("claude", "AnthropicLLM"),
    ]:
        client = build_llm({"name": name})
        assert client.__class__.__name__ == cls
        assert client.supports_function_calling() is True


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_llm({"name": "definitely-not-a-model"})


# ── base_url / model defaults per provider ──
def test_base_url_defaults_resolve(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert build_llm({"name": "deepseek"}).base_url == "https://api.deepseek.com/v1"
    assert build_llm({"name": "qwen"}).base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert build_llm({"name": "vllm"}).base_url == "http://localhost:8000/v1"
    assert build_llm({"name": "openai"}).base_url == "https://api.openai.com/v1"


def test_model_defaults_resolve():
    assert build_llm({"name": "deepseek"}).model == "deepseek-chat"
    assert build_llm({"name": "openai"}).model == "gpt-4o"
    assert build_llm({"name": "anthropic"}).model == "claude-sonnet-4-6"


def test_config_overrides_base_url_and_model():
    c = build_llm({"name": "openai-compat", "base_url": "http://h:9/v1", "model": "x"})
    assert c.base_url == "http://h:9/v1"
    assert c.model == "x"


# ── loud on missing key (no network reached) ──
def test_openai_compat_loud_without_key(monkeypatch):
    for var in ("OPENAI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    client = build_llm({"name": "openai"})
    with pytest.raises(RuntimeError, match="API key"):
        client.complete("hello")


def test_anthropic_loud_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = build_llm({"name": "anthropic"})
    with pytest.raises(RuntimeError, match="API key"):
        client.complete("hello")


def test_openai_compat_loud_without_base_url(monkeypatch):
    """openai-compat with a key but no base_url must fail loudly, not POST."""
    for var in ("LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    client = build_llm({"name": "openai-compat", "api_key": "k"})
    with pytest.raises(RuntimeError, match="base_url"):
        client.complete("hello")
