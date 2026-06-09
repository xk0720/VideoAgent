"""Track-extractor factory + real-backend graceful degradation (C6 oracle).

All tests here are CPU-only and DO NOT require torch / cotracker installed:
the real backend decodes frames BEFORE loading the model, so a mock (non-video)
clip returns None without ever touching torch — exactly the path the mock
pipeline takes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro.physics.oracle import (
    BaseTrackExtractor,
    MockTrackExtractor,
    TrajectoryOracle,
    build_track_extractor,
)
from maestro.physics.sketch import build_physics_sketch
from maestro.agents.generator import GeneratorAgent
from maestro.types import CandidateClip, ShotSpec


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def test_factory_defaults_to_mock():
    assert isinstance(build_track_extractor(None), MockTrackExtractor)
    assert isinstance(build_track_extractor("mock-track"), MockTrackExtractor)
    assert isinstance(build_track_extractor({"name": "mock-track"}), MockTrackExtractor)


def test_factory_dispatches_cotracker_without_loading_torch():
    """Constructing the real backend must NOT import torch (lazy). It only
    loads on first extract of a real video."""
    ex = build_track_extractor({"name": "cotracker", "device": "cpu"})
    assert ex.__class__.__name__ == "CoTrackerExtractor"
    assert isinstance(ex, BaseTrackExtractor)


def test_factory_dispatches_tapir():
    ex = build_track_extractor("tapir")
    assert ex.__class__.__name__ == "TAPIRExtractor"


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_track_extractor({"name": "definitely-not-a-tracker"})


# ─────────────────────────────────────────────────────────────────────────────
# Graceful degradation — real extractor on a non-video clip → None (silent)
# ─────────────────────────────────────────────────────────────────────────────
def test_cotracker_returns_none_on_mock_text_clip(tmp_path: Path):
    """A mock pipeline writes a TEXT placeholder with a .mp4 name. The real
    extractor must return None (oracle stays silent) — not crash, not load
    torch — because there are no pixels to track."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)

    ex = build_track_extractor({"name": "cotracker", "device": "cpu"})
    expected = TrajectoryOracle().expected_tracks(spec)
    assert expected is not None
    observed = ex.extract(clip, spec, expected, fps=8)
    assert observed is None


def test_cotracker_returns_none_on_missing_file(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "nope.mp4")
    ex = build_track_extractor({"name": "cotracker"})
    expected = TrajectoryOracle().expected_tracks(spec) or {}
    assert ex.extract(clip, spec, expected, fps=8) is None


def test_decode_frames_rejects_non_video(tmp_path: Path):
    from maestro.physics.track_extractor_backends import _decode_frames

    txt = tmp_path / "x.txt"
    txt.write_text("not a video" * 500)        # > 1KB but wrong suffix
    assert _decode_frames(txt) is None
    fake_mp4 = tmp_path / "y.mp4"
    fake_mp4.write_text("MOCK VIDEO metadata")   # right suffix, no pixels, < 1KB
    assert _decode_frames(fake_mp4) is None


# ─────────────────────────────────────────────────────────────────────────────
# Oracle still works end-to-end with a real-backend extractor configured
# (degrades to silent on mock clips — p2 not spuriously dinged)
# ─────────────────────────────────────────────────────────────────────────────
def test_oracle_silent_with_real_extractor_on_mock_clip(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    oracle = TrajectoryOracle(extractor=build_track_extractor({"name": "cotracker"}))
    # extraction fails on the text placeholder → compare() returns None (silent)
    assert oracle.compare(clip, spec, fps=8) is None


def test_cotracker_loud_failure_on_real_video_when_unwired(tmp_path: Path, monkeypatch):
    """The crucial honesty guarantee: a DECODABLE video + missing tracker model
    must raise LOUDLY (not silently emit a perfect, wrong p2). We fake a
    decodable clip and point at a checkpoint, forcing the cotracker import which
    is absent in this env → RuntimeError from _ensure_loaded."""
    pytest.importorskip("numpy")
    import numpy as np

    import maestro.physics.track_extractor_backends as be

    # Pretend the clip decodes to 3 RGB frames.
    monkeypatch.setattr(be, "_decode_frames",
                        lambda p: np.zeros((3, 16, 16, 3), dtype="uint8"))
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "real.mp4")
    expected = TrajectoryOracle().expected_tracks(spec)
    # checkpoint path forces the `import cotracker` branch → ImportError → loud.
    ex = build_track_extractor({"name": "cotracker", "checkpoint": "/no/such.pth"})
    with pytest.raises(RuntimeError):
        ex.extract(clip, spec, expected, fps=8)
