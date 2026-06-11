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