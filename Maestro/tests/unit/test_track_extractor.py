"""Track-extractor factory + real-backend graceful degradation (C6 v0.4).

All tests here are CPU-only and DO NOT require torch / cotracker installed:
the real backend decodes frames BEFORE loading the model, so a mock (non-video)
clip returns None without ever touching torch — exactly the path the mock
pipeline takes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro.agents.generator import GeneratorAgent
from maestro.physics.annotate import annotate_physics
from maestro.physics.tracks import (
    BaseTrackExtractor,
    MockTrackExtractor,
    build_track_extractor,
)
from maestro.physics.verifier import PhysicsFromPixelsVerifier
from maestro.types import CandidateClip, ShotSpec


def _spec(prompt="a ball falls") -> ShotSpec:
    spec = ShotSpec(shot_idx=0, duration=2.0, prompt=prompt)
    spec.physics_annotation = annotate_physics(spec)
    return spec


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
# Mock extractor contract
# ─────────────────────────────────────────────────────────────────────────────
def test_mock_extractor_tracks_every_entity(tmp_path: Path):
    spec = _spec("a ball falls while a person runs")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    tracks = MockTrackExtractor().extract(
        clip, spec, spec.physics_annotation.entities, fps=8
    )
    assert set(tracks) == {e.name for e in spec.physics_annotation.entities}
    n = max(8, int(round(spec.duration * 8)))
    assert all(len(t) == n for t in tracks.values())


def test_mock_extractor_none_on_missing_clip(tmp_path: Path):
    spec = _spec()
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "nope.mp4")
    assert MockTrackExtractor().extract(
        clip, spec, spec.physics_annotation.entities, fps=8
    ) is None


# ─────────────────────────────────────────────────────────────────────────────
# Graceful degradation — real extractor on a non-video clip → None (silent)
# ─────────────────────────────────────────────────────────────────────────────
def test_cotracker_returns_none_on_mock_text_clip(tmp_path: Path):
    """A mock pipeline writes a TEXT placeholder with a .mp4 name. The real
    extractor must return None (verifier stays silent) — not crash, not load
    torch — because there are no pixels to track."""
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    ex = build_track_extractor({"name": "cotracker", "device": "cpu"})
    assert ex.extract(clip, spec, spec.physics_annotation.entities, fps=8) is None


def test_cotracker_returns_none_on_missing_file(tmp_path: Path):
    spec = _spec()
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "nope.mp4")
    ex = build_track_extractor({"name": "cotracker"})
    assert ex.extract(clip, spec, spec.physics_annotation.entities, fps=8) is None


def test_decode_frames_rejects_non_video(tmp_path: Path):
    from maestro.physics.track_extractor_backends import _decode_frames

    txt = tmp_path / "x.txt"
    txt.write_text("not a video" * 500)        # > 1KB but wrong suffix
    assert _decode_frames(txt) is None
    fake_mp4 = tmp_path / "y.mp4"
    fake_mp4.write_text("MOCK VIDEO metadata")   # right suffix, no pixels, < 1KB
    assert _decode_frames(fake_mp4) is None


# ─────────────────────────────────────────────────────────────────────────────
# Verifier still works end-to-end with a real-backend extractor configured
# (degrades to silent on mock clips — p2 not spuriously dinged)
# ─────────────────────────────────────────────────────────────────────────────
def test_verifier_silent_with_real_extractor_on_mock_clip(tmp_path: Path):
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    verifier = PhysicsFromPixelsVerifier(
        extractor=build_track_extractor({"name": "cotracker"})
    )
    # extraction fails on the text placeholder → verify() returns None (silent)
    assert verifier.verify(clip, spec, fps=8) is None


def test_cotracker_loud_failure_on_real_video_when_unwired(tmp_path: Path, monkeypatch):
    """The crucial honesty guarantee: a DECODABLE video + missing tracker model
    must raise LOUDLY (not silently emit a perfect, wrong verdict). We fake a
    decodable clip, forcing the torch/cotracker import which is absent in this
    env → RuntimeError from _ensure_loaded."""
    pytest.importorskip("numpy")
    import numpy as np

    import maestro.physics.track_extractor_backends as be

    monkeypatch.setattr(be, "_decode_frames",
                        lambda p: np.zeros((3, 16, 16, 3), dtype="uint8"))
    spec = _spec()
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "real.mp4")
    ex = build_track_extractor({"name": "cotracker", "checkpoint": "/no/such.pth",
                                "device": "cpu"})
    with pytest.raises(RuntimeError):
        ex.extract(clip, spec, spec.physics_annotation.entities, fps=8)
