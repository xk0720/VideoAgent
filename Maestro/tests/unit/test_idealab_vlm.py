"""IdealabGeminiVLM(2026-08-19):原生 video_url 装配、超限/假文件退
抽帧、4xx 自动降级。零 API(requests 打桩)。"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from maestro.models.mllm_backends import IdealabGeminiVLM  # noqa: E402


def _mp4(tmp_path, secs=1):
    p = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"color=c=red:s=160x90:d={secs}", str(p)],
                   check=True)
    return p


def _vlm():
    v = IdealabGeminiVLM(name="idealab-gemini",
                         config={"api_key": "k", "video_fps": 5})
    return v


def test_native_video_payload(tmp_path, monkeypatch):
    clip = SimpleNamespace(video_path=str(_mp4(tmp_path)))
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, payload=json)
        return SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "看到红色画面"}}]},
            text="")
    import maestro.models.mllm_backends as M
    monkeypatch.setattr("requests.post", fake_post)
    v = _vlm()
    frames = v._sample_frames(clip)
    assert isinstance(frames[0], v._VideoRef)      # 原生哨兵
    reply = v._chat(frames, "描述")
    assert reply == "看到红色画面"
    content = captured["payload"]["messages"][0]["content"]
    vid = [c for c in content if c.get("type") == "video_url"][0]
    assert vid["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert vid["video_metadata"] == {"fps": 5}     # 用户 curl 方言
    assert captured["url"].endswith("/chat/completions")


def test_fake_file_falls_to_honesty_gate(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"x")                          # 占位假文件(<1KB)
    clip = SimpleNamespace(video_path=str(fake))
    assert _vlm()._sample_frames(clip) is None      # 无像素不判(继承闸)


def test_native_reject_falls_back_to_frames(tmp_path, monkeypatch):
    clip = SimpleNamespace(video_path=str(_mp4(tmp_path)))
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["messages"][0]["content"])
        if any(c.get("type") == "video_url"
               for c in json["messages"][0]["content"]):
            return SimpleNamespace(status_code=400, text="no data uri",
                                   json=lambda: {})
        return SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "抽帧路成功"}}]}, text="")
    monkeypatch.setattr("requests.post", fake_post)
    v = _vlm()
    reply = v._chat(v._sample_frames(clip), "描述")
    assert reply == "抽帧路成功"
    assert len(calls) == 2                          # 原生被拒 → 帧路重试
    assert all(c.get("type") != "video_url" for c in calls[1])


def test_list_content_normalized(monkeypatch):
    """2026-08-19 事故回归:idealab 网关把 message.content 按分段列表
    返回([{"type":"text","text":"…"}]),曾直接把 list 交给
    _extract_json 崩 AttributeError。修后:边界归一化成 str。"""
    from maestro.models.llm_backends import content_to_text
    assert content_to_text('{"a":1}') == '{"a":1}'
    assert content_to_text(None) is None
    assert content_to_text(
        [{"type": "text", "text": '[{"question"'},
         {"type": "text", "text": ':"q","passed":true,"fix":""}]'},
         {"type": "image_url", "image_url": {"url": "x"}},  # 非 text 段忽略
         {"type": "text", "text": None},                     # 脏 None 忽略
         "tail"]) == '[{"question":"q","passed":true,"fix":""}]tail'

    # 三个客户端的 _chat/chat 出口都吃 list 形态(打桩 HTTP,端到端)
    import requests as _rq

    class _R:
        status_code = 200
        text = ""
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": [
                {"type": "text", "text": "PART1-"},
                {"type": "text", "text": "PART2"}]}}]}

    monkeypatch.setattr(_rq, "post",
                        lambda *a, **k: _R(), raising=True)

    from maestro.models.mllm_backends import OpenAICompatVLM
    v = OpenAICompatVLM("qwen-vl", {"api_key": "k"})
    import numpy as np
    fr = np.zeros((8, 8, 3), dtype=np.uint8)
    assert v._chat([fr], "hi") == "PART1-PART2"

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl"))
    from reward.judges import OpenAICompatChat
    c = OpenAICompatChat("http://x/v1", "m", "k")
    assert c.chat("hi") == "PART1-PART2"
