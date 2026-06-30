"""propagate_repair — the localized forward-cascade algorithm.

CPU-only, NO ffmpeg/torch/network. A STUB video_gen records every generate /
frame_to_frame call; _decode_frames + _write_frame + _splice + frame_similarity
are monkeypatched so we drive the cascade deterministically and assert call
counts (repair, cascade, early-stop, degrade).
"""
from pathlib import Path

import numpy as np

from maestro.agents.defect_report import Defect
from maestro.pipeline import timeline as tl
from maestro.types import CandidateClip


class StubVideoGen:
    def __init__(self, caps=None):
        self._caps = caps or {"t2v", "i2v"}
        self.generate_calls = []
        self.flf_calls = []

    def capabilities(self):
        return set(self._caps)

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("seg", encoding="utf-8")
        self.generate_calls.append({"out": str(out_path),
                                    "first_frame": str(first_frame)})
        return out_path

    def frame_to_frame(self, prompt, first_frame, last_frame, out_path,
                       duration=5, seed=0):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("flf", encoding="utf-8")
        self.flf_calls.append({"first": str(first_frame), "last": str(last_frame)})
        return out_path


def _fake_frames(T=12, H=4, W=4):
    return np.stack([np.full((H, W, 3), i, dtype=np.uint8) for i in range(T)])


def _patch_decode_and_write(monkeypatch, frames):
    import maestro.physics.track_extractor_backends as teb
    monkeypatch.setattr(teb, "_decode_frames", lambda p: frames)

    def fake_write(frame, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("f", encoding="utf-8")
        return out_path
    monkeypatch.setattr(tl, "_write_frame", fake_write)


def _stub_splice(monkeypatch):
    """Record the spliced segment order; write a real mp4-ish file."""
    calls = {}

    def fake_splice(paths, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("spliced", encoding="utf-8")
        calls["paths"] = [str(p) for p in paths]
        return out_path
    monkeypatch.setattr(tl, "_splice", fake_splice)
    return calls


def _build_timeline(tmp_path, monkeypatch, n_segments=4):
    _patch_decode_and_write(monkeypatch, _fake_frames(T=12))
    # extract_frame should return a stub path (the "new last frame").
    def fake_extract(video_path, idx, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("newlast", encoding="utf-8")
        return out_path
    monkeypatch.setattr(tl, "extract_frame", fake_extract)
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "clip.mp4")
    return tl.ClipTimeline.from_clip(clip, tmp_path / "tl", n_segments=n_segments)


def test_degraded_timeline_returns_none(tmp_path, monkeypatch):
    import maestro.physics.track_extractor_backends as teb
    monkeypatch.setattr(teb, "_decode_frames", lambda p: None)
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "mock.mp4")
    t = tl.ClipTimeline.from_clip(clip, tmp_path / "tl", n_segments=3)
    vg = StubVideoGen()
    d = Defect("physics", "ball", (1, 2), 0.8, "motion")
    assert tl.propagate_repair(t, d, video_gen=vg, hint="fix",
                               cache_dir=tmp_path / "p") is None
    assert not vg.generate_calls


def test_repairs_defect_segment_and_cascades_no_early_stop(tmp_path, monkeypatch):
    # similarity always BELOW threshold → cascade runs to max_cascade / clip end.
    monkeypatch.setattr(tl, "frame_similarity", lambda a, b: 0.0)
    splice = _stub_splice(monkeypatch)
    t = _build_timeline(tmp_path, monkeypatch, n_segments=4)
    vg = StubVideoGen(caps={"t2v", "i2v"})           # no flf2v → i2v repair
    # defect overlaps segment 1 (frames 3-6 → seg1 is 3..6 for T=12,n=4)
    d = Defect("physics", "ball", (3, 5), 0.8, "motion")
    out = tl.propagate_repair(t, d, video_gen=vg, hint="fix arc",
                              cache_dir=tmp_path / "p", max_cascade=4)
    assert out is not None and Path(out).exists()
    # S_1 repaired via i2v generate + S_2, S_3 cascaded = 3 generate calls
    assert len(vg.generate_calls) == 3
    assert not vg.flf_calls
    # cascade segment 2 is anchored on segment 1's NEW last frame
    assert "seg1_new_last" in vg.generate_calls[1]["first_frame"]
    # splice keeps the untouched head (seg0 original) + repaired/cascaded tail
    assert "seg0_first" not in splice["paths"][0]  # head is the original clip seg


def test_similarity_above_threshold_stops_cascade_early(tmp_path, monkeypatch):
    # FIRST cascaded boundary already matches the old one → STOP after one cascade.
    monkeypatch.setattr(tl, "frame_similarity", lambda a, b: 0.99)
    _stub_splice(monkeypatch)
    t = _build_timeline(tmp_path, monkeypatch, n_segments=4)
    vg = StubVideoGen(caps={"t2v", "i2v"})
    d = Defect("physics", "ball", (3, 5), 0.8, "motion")   # seg 1
    tl.propagate_repair(t, d, video_gen=vg, hint="fix",
                        cache_dir=tmp_path / "p", max_cascade=4)
    # S_1 repair (1) + exactly ONE cascade (S_2) then early-stop = 2 calls
    assert len(vg.generate_calls) == 2


def test_flf2v_double_anchor_used_for_motion_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "frame_similarity", lambda a, b: 0.99)
    _stub_splice(monkeypatch)
    t = _build_timeline(tmp_path, monkeypatch, n_segments=4)
    vg = StubVideoGen(caps={"t2v", "i2v", "flf2v"})
    d = Defect("physics", "ball", (3, 5), 0.8, "motion")   # seg 1
    tl.propagate_repair(t, d, video_gen=vg, hint="fix",
                        cache_dir=tmp_path / "p", max_cascade=4)
    # repair uses flf2v (double-anchor): prev.last → next.first
    assert len(vg.flf_calls) == 1
    assert "seg0_last" in vg.flf_calls[0]["first"]
    assert "seg2_first" in vg.flf_calls[0]["last"]


def test_no_ffmpeg_splice_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "frame_similarity", lambda a, b: 0.0)
    # _splice returns None (ffmpeg missing) → whole propagate degrades to None.
    monkeypatch.setattr(tl, "_splice", lambda paths, out: None)
    t = _build_timeline(tmp_path, monkeypatch, n_segments=3)
    vg = StubVideoGen()
    d = Defect("physics", "ball", (0, 2), 0.8, "motion")
    assert tl.propagate_repair(t, d, video_gen=vg, hint="x",
                               cache_dir=tmp_path / "p") is None
