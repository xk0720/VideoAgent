"""ClipTimeline + frame_similarity. CPU-only, no torch/ffmpeg/network.

_decode_frames is monkeypatched to a fake (T,H,W,3) ndarray; _write_frame is
monkeypatched so boundary-frame writing never needs imageio/PIL.
"""
from pathlib import Path

import numpy as np
import pytest

from maestro.pipeline import timeline as tl
from maestro.types import CandidateClip


def _fake_frames(T=12, H=4, W=4):
    # Each frame a distinct constant value so boundaries are identifiable.
    return np.stack([np.full((H, W, 3), i, dtype=np.uint8) for i in range(T)])


def _stub_write_frame(monkeypatch):
    """Make _write_frame write a tiny text stub recording the frame's [0,0,0]
    value, so we never depend on a real PNG encoder."""
    def fake(frame, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(str(int(frame[0, 0, 0])), encoding="utf-8")
        return out_path
    monkeypatch.setattr(tl, "_write_frame", fake)


def test_from_clip_splits_into_n_segments_with_boundaries(tmp_path, monkeypatch):
    frames = _fake_frames(T=12)
    monkeypatch.setattr(tl, "_decode_frames", lambda p: frames, raising=False)
    # extract_frame / from_clip both import _decode_frames lazily from the
    # track-extractor module, so patch THERE too.
    import maestro.physics.track_extractor_backends as teb
    monkeypatch.setattr(teb, "_decode_frames", lambda p: frames)
    _stub_write_frame(monkeypatch)

    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "clip.mp4")
    t = tl.ClipTimeline.from_clip(clip, tmp_path / "tl", n_segments=3)
    assert not t.degraded
    assert len(t.segments) == 3
    assert t.n_frames == 12
    # equal spans over [0,12): 0-4, 4-8, 8-12
    assert [(s.start_frame, s.end_frame) for s in t.segments] == [(0, 4), (4, 8), (8, 12)]
    for s in t.segments:
        assert s.first_frame_path and s.first_frame_path.exists()
        assert s.last_frame_path and s.last_frame_path.exists()
    # segment 0's last frame is frame index 3
    assert t.segments[0].last_frame_path.read_text() == "3"


def test_from_clip_mock_text_clip_is_degraded(tmp_path, monkeypatch):
    import maestro.physics.track_extractor_backends as teb
    monkeypatch.setattr(teb, "_decode_frames", lambda p: None)

    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "mock.mp4")
    t = tl.ClipTimeline.from_clip(clip, tmp_path / "tl", n_segments=3)
    assert t.degraded
    assert len(t.segments) == 1
    assert t.segments[0].first_frame_path is None


def test_segment_for_frame_range_picks_max_overlap(tmp_path, monkeypatch):
    frames = _fake_frames(T=12)
    import maestro.physics.track_extractor_backends as teb
    monkeypatch.setattr(teb, "_decode_frames", lambda p: frames)
    _stub_write_frame(monkeypatch)
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "clip.mp4")
    t = tl.ClipTimeline.from_clip(clip, tmp_path / "tl", n_segments=3)
    assert t.segment_for_frame_range((9, 11)).idx == 2
    assert t.segment_for_frame_range((4, 6)).idx == 1
    assert t.segment_for_frame_range((0, 1)).idx == 0


def test_frame_similarity_identical_is_high(tmp_path):
    img = tmp_path / "a.png"
    try:
        from PIL import Image
    except Exception:
        pytest.skip("PIL not available")
    Image.fromarray(np.full((4, 4, 3), 100, dtype=np.uint8)).save(img)
    assert tl.frame_similarity(img, img) > 0.99


def test_frame_similarity_different_is_low(tmp_path):
    try:
        from PIL import Image
    except Exception:
        pytest.skip("PIL not available")
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(a)
    Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)).save(b)
    assert tl.frame_similarity(a, b) < 0.2


def test_frame_similarity_missing_returns_zero(tmp_path):
    # Missing file → 0.0 (conservative: "different", never early-stop).
    assert tl.frame_similarity(tmp_path / "nope.png", tmp_path / "nope2.png") == 0.0
