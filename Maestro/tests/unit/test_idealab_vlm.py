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
