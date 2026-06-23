"""Audio-gen backend factory + AudioGenTool delegation (v0.4: foley + TTS real
via WaveSpeed, music still a mock). CPU-only, NO network — every real client is
constructed and probed for its loud-on-missing-key / bad-input guards without
ever issuing an HTTP request."""
from pathlib import Path

import pytest

from maestro.models.audio_gen_backends import (
    BaseAudioGenClient,
    MockAudioGenClient,
    WaveSpeedAudioClient,
    build_audio_gen,
)
from maestro.tools.audio_gen import AudioGenTool


# ── factory dispatch ──
def test_factory_returns_mock_by_default():
    assert isinstance(build_audio_gen(None), MockAudioGenClient)
    assert isinstance(build_audio_gen("mock-audio-gen"), MockAudioGenClient)


def test_factory_dispatches_wavespeed():
    for name in ("wavespeed", "mmaudio", "minimax"):
        client = build_audio_gen({"name": name})
        assert isinstance(client, WaveSpeedAudioClient)
    # str spec works too
    assert isinstance(build_audio_gen("wavespeed"), WaveSpeedAudioClient)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_audio_gen({"name": "definitely-not-an-audio-model"})


def test_lazy_construction_no_network(monkeypatch):
    """Constructing the real client touches no network and needs no key."""
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    client = build_audio_gen({"name": "wavespeed"})
    assert isinstance(client, BaseAudioGenClient)


# ── loud-on-missing-key (no network reached) ──
def test_wavespeed_foley_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    client = build_audio_gen({"name": "wavespeed"})
    # Give it a real video file so we get past the input guard to the key check.
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="API key"):
        client.foley(vid, "wind in trees", tmp_path / "a.wav")


def test_wavespeed_speech_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    client = build_audio_gen({"name": "wavespeed"})
    with pytest.raises(RuntimeError, match="API key"):
        client.speech("hello world", tmp_path / "a.wav")


# ── foley input validation (raises before any network) ──
def test_foley_missing_path_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")  # past key check
    client = build_audio_gen({"name": "wavespeed"})
    with pytest.raises(FileNotFoundError):
        client.foley(tmp_path / "nope.mp4", "rain", tmp_path / "a.wav")


def test_foley_non_video_path_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    client = build_audio_gen({"name": "wavespeed"})
    txt = tmp_path / "notes.txt"
    txt.write_text("not a video")
    with pytest.raises(ValueError):
        client.foley(txt, "rain", tmp_path / "a.wav")


# ── supported_kinds per client ──
def test_supported_kinds():
    assert build_audio_gen("wavespeed").supported_kinds() == {"foley", "tts"}
    # mock fakes all three
    assert MockAudioGenClient().supported_kinds() == {"foley", "tts", "music"}


# ── AudioGenTool back-compat (default mock client) ──
def test_tool_default_is_mock():
    assert isinstance(AudioGenTool().client, MockAudioGenClient)


def test_tool_music_returns_path_with_mock(tmp_path: Path):
    out = AudioGenTool().run("a calm motif", tmp_path / "m.mp3", duration=4.0, kind="music")
    assert isinstance(out, Path)
    body = out.read_text()
    assert "kind=music" in body
    assert "prompt=a calm motif" in body


def test_tool_tts_returns_path_with_mock(tmp_path: Path):
    out = AudioGenTool().run("narration line", tmp_path / "n.mp3", kind="tts")
    assert isinstance(out, Path)
    assert "kind=tts" in out.read_text()


def test_tool_foley_requires_video_path(tmp_path: Path):
    with pytest.raises(ValueError, match="video_path"):
        AudioGenTool().run("ambient", tmp_path / "f.mp3", kind="foley")


def test_tool_foley_with_mock_client(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00")
    out = AudioGenTool().run("ambient", tmp_path / "f.mp3", kind="foley", video_path=vid)
    assert isinstance(out, Path)
    assert "kind=foley" in out.read_text()
