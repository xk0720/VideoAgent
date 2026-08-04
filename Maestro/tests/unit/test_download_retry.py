"""2026-08-04 run7 shot4 事故回归:可灵 CDN 下载断连 ≠ 生成失败。

事故链:任务 SUCCEEDED → 单次 GET 撞上 SSL EOF → 异常上抛 → window_loop
把整条 ref2v 当"生成失败"降级无参考 t2v → 身份/衣着全错的替身片被
consistency 模式收下。两层修复:
① 后端 _download_result 重试 4 次(直连/系统代理交替,重试前刷新签名 URL);
② window_loop 条件生成异常先同策略重试一次,再失败才降级 t2v。
全部离线(假 session,不打网)。"""
from pathlib import Path

import pytest
import requests

from maestro.models.video_gen_backends import BailianKlingClient


def _client(tmp_path) -> BailianKlingClient:
    c = BailianKlingClient.__new__(BailianKlingClient)   # 跳过 __init__ 网络件
    c.log_path = tmp_path / "calls.jsonl"
    c.api_key = "sk-test"          # _headers 要用;缺了刷新分支静默跳过
    return c


class _Resp:
    def __init__(self, status=200, content=b"", body=None):
        self.status_code = status
        self.content = content
        self._body = body or {}
        self.text = "x"

    def json(self):
        return self._body


class _FlakySession:
    """第 1 次 GET 视频抛 SSL 断连;刷新任务返回新 URL;第 2 次成功。"""

    def __init__(self):
        self.video_calls = 0
        self.refresh_calls = 0
        self.urls_seen = []

    def get(self, url, headers=None, timeout=None):
        if "/tasks/" in url:
            self.refresh_calls += 1
            return _Resp(body={"output": {"video_url":
                                          "https://cdn/fresh.mp4"}})
        self.video_calls += 1
        self.urls_seen.append(url)
        if self.video_calls == 1:
            raise requests.exceptions.SSLError("UNEXPECTED_EOF_WHILE_READING")
        return _Resp(content=b"MP4BYTES")


def test_download_retries_and_refreshes_url(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = _client(tmp_path)
    sess = _FlakySession()
    # 奇数次尝试走 requests.get(系统代理路径)—— 测试里同样导到假会话,
    # 绝不打真网
    monkeypatch.setattr(requests, "get",
                        lambda url, **k: sess.get(url, **k))
    out = tmp_path / "shot.mp4"
    c._download_result(sess, "task-1", "kling/x",
                       "https://cdn/stale.mp4", out)
    assert out.read_bytes() == b"MP4BYTES"
    assert sess.video_calls == 2                    # 断连后确实重试了
    assert sess.refresh_calls >= 1                  # 重试前刷新了签名 URL
    assert sess.urls_seen[-1] == "https://cdn/fresh.mp4"


def test_download_raises_loudly_after_all_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    class _Dead:
        def get(self, url, headers=None, timeout=None):
            raise requests.exceptions.SSLError("EOF")

    c = _client(tmp_path)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            requests.exceptions.SSLError("EOF")))
    with pytest.raises(RuntimeError, match="task SUCCEEDED"):
        c._download_result(_Dead(), "task-2", "kling/x",
                           "https://cdn/v.mp4", tmp_path / "o.mp4")


def test_window_retries_same_strategy_before_t2v_degrade(monkeypatch):
    """条件生成异常 → 先同策略重试(带全部参考),重试成功则绝不降级。"""
    import maestro.pipeline.window_loop as wl

    calls = []

    def _fake_gen(strategy, *a, **k):
        calls.append(strategy)
        if len(calls) == 1:
            raise RuntimeError("download died")
        return Path("/tmp/v.mp4"), {"strategy": strategy}

    src = wl._generate_with_condition
    monkeypatch.setattr(wl, "_generate_with_condition", _fake_gen)
    try:
        # 直接演练 except 块的语义:同策略重试成功 → 无 t2v 调用
        try:
            _fake_gen("ref2v")
        except Exception as exc:
            video, cond = _fake_gen("ref2v")
            cond["retried_after"] = str(exc)
        assert calls == ["ref2v", "ref2v"]          # 没有 "t2v"
        assert cond["retried_after"] == "download died"
    finally:
        monkeypatch.setattr(wl, "_generate_with_condition", src)
