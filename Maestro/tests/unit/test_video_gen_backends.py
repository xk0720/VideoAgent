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
    assert wave.capabilities() == {"t2v", "i2v", "flf2v", "edit", "extend"}
    # Extra capabilities are optional methods, NOT abstractmethods.
    assert hasattr(wave, "frame_to_frame") and hasattr(wave, "edit_video")
    assert hasattr(wave, "extend")
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


# ── video EXTEND (the capability UniVA has as video_extension; we add it) ──
def test_wavespeed_extend_loud_without_api_key(tmp_path: Path, monkeypatch):
    """extend() must fail LOUDLY (no key) BEFORE any decode or network POST.
    We fake the decoder so a missing real video can't mask the key check."""
    pytest.importorskip("numpy")
    import numpy as np

    import maestro.physics.track_extractor_backends as be

    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    # If the key check were skipped, this fake would let extend proceed — so a
    # RuntimeError here proves the loud guard fires first (no network either).
    monkeypatch.setattr(be, "_decode_frames",
                        lambda p: np.zeros((3, 8, 8, 3), dtype="uint8"))
    wave = build_video_gen({"name": "wavespeed"})
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    with pytest.raises(RuntimeError, match="API key"):
        wave.extend("continue the shot", vid, tmp_path / "o.mp4")


def test_wavespeed_extend_drives_i2v_on_last_frame(tmp_path: Path, monkeypatch):
    """With a key + a fake decoder, extend() saves the LAST frame and drives an
    i2v continuation via _run_task — recorded by a stub, NO network."""
    pytest.importorskip("numpy")
    import numpy as np

    import maestro.physics.track_extractor_backends as be

    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    frames = np.zeros((3, 8, 8, 3), dtype="uint8")
    frames[-1, 0, 0, 0] = 200  # mark the last frame
    monkeypatch.setattr(be, "_decode_frames", lambda p: frames)

    wave = build_video_gen(
        {"name": "wavespeed", "model_id": "bytedance/seedance-v1-pro-t2v-480p"}
    )
    calls = {}

    def _fake_run_task(model_id, payload, out_path):
        calls["model_id"] = model_id
        calls["payload"] = payload
        out = Path(out_path); out.write_bytes(b"OUT")
        return out

    monkeypatch.setattr(wave, "_run_task", _fake_run_task)
    vid = tmp_path / "in.mp4"; vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    out = wave.extend("continue the shot", vid, tmp_path / "o.mp4", duration=5)

    assert out.exists()
    assert "-i2v-" in calls["model_id"]                 # swapped to the i2v id
    assert calls["payload"]["image"].startswith("data:image/png;base64,")
    assert calls["payload"]["prompt"] == "continue the shot"