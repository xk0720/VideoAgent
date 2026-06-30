"""Segment timeline + the LOCALIZED, PROPAGATED repair algorithm (v0.4).

A whole-clip reroll throws away every good frame to fix one bad span. This
module splits a clip into time SEGMENTS, repairs only the segment a defect
overlaps, then PROPAGATES that repair forward by re-anchoring each downstream
segment on its predecessor's NEW last frame — stopping as soon as continuity
reconverges (the new boundary matches the old one). The untouched head and the
post-convergence tail are spliced back in unchanged.

Why a forward cascade with an early-stop:

  Editing segment S_i changes its last frame. S_{i+1} was generated to continue
  the OLD S_i, so it no longer joins cleanly. We re-generate S_{i+1} anchored
  (i2v) on S_i's NEW last frame, then S_{i+2} on S_{i+1}'s new last frame, and
  so on. But the edit's influence decays: once a regenerated boundary matches
  the segment's OLD boundary (frame_similarity >= sim_threshold), everything
  downstream of it is still valid as-is, so we STOP. This is the
  "edit one segment → re-anchor downstream until continuity converges" rule.

Everything degrades honestly:
  • non-video mock clip (no decodable frames) → a single degenerate segment with
    `degraded=True`; `propagate_repair` returns None and the caller falls back to
    a whole-clip action.
  • no imageio/PIL to write boundary frames, or no numpy/PIL for similarity, or
    no ffmpeg to splice → degraded / None, never a crash.

Training-free; reuses `_decode_frames` (the same decoder the track extractor and
`extend()` use) and `VideoConcatTool` (ffmpeg concat) for splicing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Lazy image IO — write/read a single frame as PNG without forcing imageio/PIL
# into the mock pipeline's import graph.
# ─────────────────────────────────────────────────────────────────────────────
def _write_frame(frame, out_path: Path) -> Optional[Path]:
    """Write an (H,W,3) uint8 ndarray to PNG. None if no writer is available."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio  # type: ignore

        iio.imwrite(str(out_path), frame)
        return out_path
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(frame).save(str(out_path))
        return out_path
    except Exception:
        return None


def extract_frame(video_path: Path, idx: int, out_path: Path) -> Optional[Path]:
    """Decode `video_path` and write frame `idx` (clamped) to `out_path` as PNG.

    Returns the path, or None if the clip is not decodable or no image writer is
    available. Reuses `_decode_frames` (the shared decoder)."""
    from ..physics.track_extractor_backends import _decode_frames

    frames = _decode_frames(Path(video_path))
    if frames is None or len(frames) < 1:
        return None
    i = max(0, min(int(idx), len(frames) - 1))
    return _write_frame(frames[i], Path(out_path))


def frame_similarity(img_a_path, img_b_path) -> float:
    """Cheap pixel similarity in [0,1]: 1 - clamp(normalized MSE).

    Identical frames → ~1.0; very different frames → low. Lazy numpy/PIL; if
    either image is missing/unreadable or the libs are unavailable, returns 0.0
    — treated as "different" so the cascade NEVER early-stops on missing
    evidence (the conservative choice: keep re-anchoring rather than wrongly
    declaring continuity)."""
    try:
        import numpy as np
        from PIL import Image  # type: ignore
    except Exception:
        return 0.0
    try:
        a = np.asarray(Image.open(str(img_a_path)).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(str(img_b_path)).convert("RGB"), dtype=np.float64)
    except Exception:
        return 0.0
    if a.shape != b.shape:
        # Resize b to a's shape via PIL (cheap nearest) so MSE is defined.
        try:
            b_img = Image.open(str(img_b_path)).convert("RGB").resize(
                (a.shape[1], a.shape[0])
            )
            b = np.asarray(b_img, dtype=np.float64)
        except Exception:
            return 0.0
    mse = float(np.mean((a - b) ** 2)) / (255.0 ** 2)   # normalize to [0,1]
    return float(max(0.0, 1.0 - min(1.0, mse)))


@dataclass
class Segment:
    idx: int
    start_frame: int
    end_frame: int                       # exclusive
    video_path: Path
    first_frame_path: Optional[Path] = None
    last_frame_path: Optional[Path] = None


@dataclass
class ClipTimeline:
    clip_path: Path
    n_frames: int
    segments: list[Segment] = field(default_factory=list)
    degraded: bool = False               # no real frames / no image writer
    cache_dir: Optional[Path] = None

    @classmethod
    def from_clip(
        cls, clip, cache_dir, n_segments: int = 3
    ) -> "ClipTimeline":
        """Split a clip into `n_segments` equal time spans, writing each
        segment's first/last boundary frame to `cache_dir` as PNG.

        Non-decodable (mock) clip → a single degenerate segment with no boundary
        images and `degraded=True`, so callers no-op gracefully."""
        from ..physics.track_extractor_backends import _decode_frames

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip_path = Path(getattr(clip, "video_path", clip))

        frames = _decode_frames(clip_path)
        if frames is None or len(frames) < 2:
            seg = Segment(idx=0, start_frame=0, end_frame=0, video_path=clip_path)
            return cls(clip_path=clip_path, n_frames=0, segments=[seg],
                       degraded=True, cache_dir=cache_dir)

        n = len(frames)
        n_segments = max(1, min(int(n_segments), n))
        # Equal spans over [0, n); last span absorbs the remainder.
        bounds = [round(i * n / n_segments) for i in range(n_segments)] + [n]
        segments: list[Segment] = []
        degraded = False
        for i in range(n_segments):
            s, e = bounds[i], bounds[i + 1]
            if e <= s:                       # degenerate span (n < n_segments)
                e = s + 1
            first = _write_frame(
                frames[s], cache_dir / f"seg{i}_first.png"
            )
            last = _write_frame(
                frames[min(e - 1, n - 1)], cache_dir / f"seg{i}_last.png"
            )
            if first is None or last is None:
                degraded = True              # no image writer available
            segments.append(Segment(
                idx=i, start_frame=s, end_frame=e, video_path=clip_path,
                first_frame_path=first, last_frame_path=last,
            ))
        return cls(clip_path=clip_path, n_frames=n, segments=segments,
                   degraded=degraded, cache_dir=cache_dir)

    def segment_for_frame_range(self, frame_range) -> Optional[Segment]:
        """The segment whose span overlaps `frame_range` the most."""
        if not self.segments:
            return None
        lo, hi = int(frame_range[0]), int(frame_range[1])
        if hi <= lo:
            hi = lo + 1
        best, best_ov = None, -1
        for seg in self.segments:
            ov = max(0, min(hi, seg.end_frame) - max(lo, seg.start_frame))
            if ov > best_ov:
                best, best_ov = seg, ov
        return best or self.segments[0]


def _splice(segment_paths: list[Path], out_path: Path) -> Optional[Path]:
    """ffmpeg-concat the (ordered) segment clips into `out_path`.

    Returns None when ffmpeg is unavailable so the caller can degrade to a
    whole-clip action rather than ship a manifest placeholder as if it were a
    real spliced clip."""
    import shutil

    if not shutil.which("ffmpeg"):
        return None
    from ..tools.video_concat import VideoConcatTool

    try:
        return VideoConcatTool().run([str(p) for p in segment_paths], out_path)
    except Exception:
        return None


def propagate_repair(
    timeline: ClipTimeline,
    defect,
    *,
    video_gen,
    image_edit=None,
    hint: str = "",
    cache_dir,
    sim_threshold: float = 0.92,
    max_cascade: int = 4,
) -> Optional[Path]:
    """Repair the defect's segment, propagate the edit forward until continuity
    reconverges, then splice. Returns the spliced clip path, or None (degrade).

    Algorithm (the crux):
      1. degraded timeline → return None (caller falls back to a whole-clip tool).
      2. find S_i overlapping defect.frame_range; repair it:
           • fix_modality=="motion" and the backend has "flf2v" and S_i has both
             boundaries → flf2v double-anchor (prev.last/S_i.first → next.first/
             S_i.last): strongest continuity for the edited span;
           • else i2v generate(first_frame = S_i.first [or an image_edit'd
             corrected keyframe]) + extra_prompt=hint.
      3. FORWARD CASCADE (continuity lock): for j = i+1 .. i+max_cascade,
         re-generate S_j i2v-anchored on S_{j-1}'s NEW last frame; extract S_j's
         new last frame; if it matches S_j's OLD last frame
         (frame_similarity >= sim_threshold) STOP — downstream is still valid.
      4. splice [untouched head | repaired+cascaded | untouched tail] via ffmpeg.
    """
    if timeline is None or timeline.degraded or not timeline.segments:
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    caps = video_gen.capabilities() if video_gen is not None else set()
    fr = getattr(defect, "frame_range", (0, timeline.n_frames))
    modality = getattr(defect, "fix_modality", "motion")

    seg = timeline.segment_for_frame_range(fr)
    if seg is None:
        return None
    i = seg.idx
    segs = timeline.segments
    # Mutable per-segment output paths: start as the originals, overwrite the
    # repaired + cascaded ones. Track each segment's NEW last frame as we go.
    new_paths: list[Path] = [s.video_path for s in segs]
    new_last: list[Optional[Path]] = [s.last_frame_path for s in segs]

    dur = max(1, seg.end_frame - seg.start_frame)

    # ── 2. repair S_i ────────────────────────────────────────────────────────
    repaired: Optional[Path] = None
    out_i = cache_dir / f"seg{i}_repaired.mp4"
    if (modality == "motion" and "flf2v" in caps
            and seg.first_frame_path and seg.last_frame_path
            and hasattr(video_gen, "frame_to_frame")):
        first_anchor = (segs[i - 1].last_frame_path if i > 0
                        and segs[i - 1].last_frame_path else seg.first_frame_path)
        last_anchor = (segs[i + 1].first_frame_path if i + 1 < len(segs)
                       and segs[i + 1].first_frame_path else seg.last_frame_path)
        repaired = video_gen.frame_to_frame(
            prompt=hint or "one continuous passive trajectory",
            first_frame=first_anchor, last_frame=last_anchor,
            out_path=out_i, duration=dur,
        )
    else:
        anchor = seg.first_frame_path
        if image_edit is not None and anchor is not None and modality in (
            "motion", "presence", "content"
        ) and hint:
            edited = image_edit.edit(
                anchor, hint, cache_dir / f"seg{i}_anchor_edit.txt"
            )
            anchor = edited or anchor
        repaired = video_gen.generate(
            prompt=hint or "regenerate this span faithfully",
            duration=dur, out_path=out_i, first_frame=anchor,
        )
    if repaired is None:
        return None
    new_paths[i] = Path(repaired)
    # New last frame of the repaired segment (drives the cascade anchor).
    nl = extract_frame(repaired, 10**9, cache_dir / f"seg{i}_new_last.png")
    new_last[i] = nl or seg.last_frame_path

    # ── 3. forward cascade with similarity early-stop ────────────────────────
    cascade_depth = 0
    end = min(len(segs) - 1, i + max_cascade)
    for j in range(i + 1, end + 1):
        anchor = new_last[j - 1]
        if anchor is None:                   # lost the anchor → cannot continue
            break
        out_j = cache_dir / f"seg{j}_cascade.mp4"
        dur_j = max(1, segs[j].end_frame - segs[j].start_frame)
        regen = video_gen.generate(
            prompt=hint or "continue the shot; keep one continuous trajectory",
            duration=dur_j, out_path=out_j, first_frame=anchor,
        )
        if regen is None:
            break
        new_paths[j] = Path(regen)
        cascade_depth += 1
        new_j_last = extract_frame(
            regen, 10**9, cache_dir / f"seg{j}_new_last.png"
        )
        new_last[j] = new_j_last or segs[j].last_frame_path
        # Continuity reconverged? If the regenerated boundary already matches the
        # OLD boundary, everything downstream of j is still valid — STOP.
        if new_j_last is not None and segs[j].last_frame_path is not None:
            sim = frame_similarity(new_j_last, segs[j].last_frame_path)
            if sim >= sim_threshold:
                break

    # ── 4. splice everything back together ───────────────────────────────────
    out = cache_dir / "spliced_repair.mp4"
    spliced = _splice(new_paths, out)
    return spliced
