"""GeminiLLM(原生 generateContent 纯文本后端,2026-07-29)回归:
端点/认证/payload 形状与 GeminiVLM 同款;HTTP 错误与空回复必须响亮抛,
绝不静默;$GEMINI_BASE_URL 中转覆盖生效。全部离线(stub requests)。"""

import sys
import types

import pytest

from maestro.models.llm import build_llm
from maestro.models.llm_backends import GeminiLLM


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.text = text or str(body)

    def json(self):
        return self._body


def _stub_requests(monkeypatch, resp, calls):
    stub = types.SimpleNamespace(
        post=lambda url, json=None, headers=None, timeout=None:
            (calls.append({"url": url, "json": json, "headers": headers})
             or resp))
    monkeypatch.setitem(sys.modules, "requests", stub)


def test_registered_in_factory():
    llm = build_llm({"name": "gemini", "api_key": "k"})
    assert isinstance(llm, GeminiLLM)
    assert llm.model == "gemini-3.5-pro"          # 默认最强文本档


def test_complete_payload_shape_and_parse(monkeypatch):
    calls = []
    _stub_requests(monkeypatch, _Resp(body={
        "candidates": [{"content": {"parts": [{"text": "pong"}]}}]}), calls)
    llm = GeminiLLM(config={"api_key": "k", "model": "gemini-3.5-pro"})
    out = llm.complete("ping", max_tokens=99, temperature=0.1)
    assert out == "pong"
    c = calls[0]
    # 端点形状与 GeminiVLM 同款:/v1beta/models/<model>:generateContent
    assert c["url"].endswith("/v1beta/models/gemini-3.5-pro:generateContent")
    assert c["headers"]["x-goog-api-key"] == "k"
    assert c["json"]["contents"] == [
        {"role": "user", "parts": [{"text": "ping"}]}]
    gc = c["json"]["generationConfig"]
    assert gc == {"temperature": 0.1, "maxOutputTokens": 99}


def test_base_url_env_override(monkeypatch):
    # 自定义中转($GEMINI_BASE_URL)必须生效 —— 用户的网络路走这里
    monkeypatch.setenv("GEMINI_BASE_URL", "https://relay.example.com/")
    calls = []
    _stub_requests(monkeypatch, _Resp(body={
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}]}), calls)
    llm = GeminiLLM(config={"api_key": "k"})
    llm.complete("x")
    assert calls[0]["url"].startswith(
        "https://relay.example.com/v1beta/models/")


def test_http_error_raises_with_body(monkeypatch):
    _stub_requests(monkeypatch,
                   _Resp(status=404, text='{"error": "model not found"}'),
                   [])
    llm = GeminiLLM(config={"api_key": "k"})
    with pytest.raises(RuntimeError) as e:
        llm.complete("x")
    assert "HTTP 404" in str(e.value) and "model not found" in str(e.value)


def test_empty_reply_and_missing_key_raise(monkeypatch):
    _stub_requests(monkeypatch, _Resp(body={"candidates": []}), [])
    with pytest.raises(RuntimeError, match="no text part"):
        GeminiLLM(config={"api_key": "k"}).complete("x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        GeminiLLM(config={}).complete("x")
