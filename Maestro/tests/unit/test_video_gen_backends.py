"""Video-gen backend factory + graceful degradation (v0.4: no control signal —
conditioning = first_frame + reference_images only)."""
from pathlib import Path

import pytest

from maestro.models.video_gen import MockVideoGenClient, build_video_gen


def test_factory_returns_mock_by_default():
    assert isinstance(build_video_gen("mock-video-gen"), MockVideoGenClient)
    assert isinstance(build_video_gen(None), MockVideoGenClient)


def test_factory_dispatches_real_backends():
    omni = build_video_gen({"name": "omniweaving"})
    assert omni.__class__.__name__ == "OmniWeavingClient"
    wave = build_video_gen({"name": "wavespeed"})
    assert wave.__class__.__name__ == "WaveSpeedClient"
    # v0.4 conditioning contract: keyframe anchor + identity refs, NO control
    assert "first_frame" in omni.supported_conditions()
    assert "control_signal" not in omni.supported_conditions()


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_video_gen({"name": "definitely-not-a-model"})


def test_real_backend_guards_when_unwired(tmp_path: Path):
    client = build_video_gen({"name": "omniweaving"})
    with pytest.raises((RuntimeError, NotImplementedError)):
        client.generate("a ball falls", 1.0, tmp_path / "o.mp4")


def test_wavespeed_loud_without_api_key(tmp_path: Path, monkeypatch):
    """The API backend must fail LOUDLY when configured without a key —
    never silently fall back to mock output."""
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    client = build_video_gen({"name": "wavespeed"})
    with pytest.raises(RuntimeError, match="API key"):
        client.generate("a ball falls", 1.0, tmp_path / "w.mp4")


def test_mock_writes_metadata_without_control(tmp_path: Path):
    out = MockVideoGenClient().generate("a ball falls", 1.0, tmp_path / "m.mp4")
    body = out.read_text()
    assert "prompt=a ball falls" in body
    assert "control_signal" not in body      # the dead line stays dead


# ── Phase-2 capability registry seed (capabilities() + optional methods) ──
def test_capabilities_per_client():
    # Default base = t2v/i2v only.
    assert MockVideoGenClient().capabilities() == {"t2v", "i2v"}
    assert build_video_gen({"name": "omniweaving"}).capabilities() == {"t2v", "i2v"}
    # WaveSpeed declares the extras backed by optional methods.
    wave = build_video_gen({"name": "wavespeed"})
    assert wave.capabilities() == {"t2v", "i2v", "flf2v", "edit"}
    # Extra capabilities are optional methods, NOT abstractmethods.
    assert hasattr(wave, "frame_to_frame") and hasattr(wave, "edit_video")
    assert not hasattr(MockVideoGenClient(), "frame_to_frame")


def test_wavespeed_flf2v_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    wave = build_video_gen({"name": "wavespeed"})
    first = tmp_path / "a.jpg"; first.write_bytes(b"\xff\xd8\xff")
    last = tmp_path / "b.jpg"; last.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(RuntimeError, match="API key"):
        wave.frame_to_frame("morph", first, last, tmp_path / "o.mp4")


def test_wavespeed_edit_video_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="API key"):
        wave.edit_video("make it rain", vid, tmp_path / "o.mp4", backend="runway")


def test_wavespeed_edit_video_unknown_backend(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00")
    with pytest.raises(ValueError):
        wave.edit_video("x", vid, tmp_path / "o.mp4", backend="nope")