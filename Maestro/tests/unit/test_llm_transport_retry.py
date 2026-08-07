"""LLM 传输层瞬时重试(2026-08-06 事故:两次空回复烧光机位树重试,
树平坠降级 —— 根修在客户端,惠及所有 agent)。"""
import requests

from maestro.models.llm_backends import (OpenAICompatLLM,
                                         _post_with_transient_retry)


class _Resp:
    def __init__(self, status, content="ok"):
        self.status_code = status
        self.text = content
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_transient_status_then_success(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        return _Resp(503) if len(calls) == 1 else _Resp(200)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = _post_with_transient_retry("http://x", {}, {}, 5, tag="t")
    assert r.status_code == 200 and len(calls) == 2


def test_connection_error_then_success(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("reset")
        return _Resp(200)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = _post_with_transient_retry("http://x", {}, {}, 5, tag="t")
    assert r.status_code == 200 and len(calls) == 2


def test_exhausted_transients_surface(monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(429, "rate limited"))
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = _post_with_transient_retry("http://x", {}, {}, 5, tag="t")
    assert r.status_code == 429          # 穷尽后如实交出 HTTP 错误


def test_complete_rides_the_retry(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        return _Resp(502) if len(calls) == 1 else _Resp(200, "hello")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = OpenAICompatLLM(name="openai-compat",
                          config={"base_url": "http://x", "model": "m",
                                  "api_key": "k"})
    assert llm.complete("hi") == "hello" and len(calls) == 2
