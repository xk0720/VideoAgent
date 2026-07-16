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

from ..logging_utils import get_logger

log = get_logger(__name__)


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


def _same_shot(img_a_path, img_b_path, threshold: float = 27.0) -> bool:
    """Pre-generation condition check for flf2v double-anchoring: are the two
    anchor frames plausibly the SAME shot?

    FLF2V models respond to overly-dissimilar start/end frames by inserting a
    LENS SWITCH (a cut) instead of blending — documented in Kling's start/end
    docs — which would ruin a repair segment. Before spending a generation we
    gate on the only battle-tested "content changed" threshold in the field:
    PySceneDetect's ContentDetector — mean per-pixel |ΔHSV| > 27 (0-255 scale)
    = a shot cut. (NEWTON, arXiv:2605.18396, applies the same philosophy:
    judge the CONDITIONING before paying for a generation.)

    Unknown (missing libs / unreadable stub images, e.g. mock mode) → True:
    the check only exists to catch a REAL, measurable scene jump; when it
    cannot measure, flf2v keeps its normal priority."""
    try:
        import numpy as np
        from PIL import Image  # type: ignore

        a = Image.open(str(img_a_path)).convert("HSV")
        b = Image.open(str(img_b_path)).convert("HSV")
        if a.size != b.size:
            b = b.resize(a.size)
        da = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        return float(da.mean()) <= float(threshold)
    except Exception:
        return True


def _probe_fps(path: Path) -> float:
    """Container-reported fps of a video, or 0.0 if unknowable (mock clip /
    no decoder). Needed to convert segment FRAME spans into SECONDS for the
    generation APIs — WaveSpeed's `duration` is seconds, and its models output
    their own fps (seedance: 24), not whatever fps we requested."""
    p = Path(path)
    try:
        import decord  # type: ignore

        fps = float(decord.VideoReader(str(p)).get_avg_fps())
        if fps > 0:
            return fps
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def _fit_to_seconds(video_path: Path, target_s: float, out_path: Path) -> Path:
    """Retime a generated segment to `target_s` seconds.

    Fixed-duration APIs (seedance: 5 or 10 s) return MORE video than a short
    segment span needs. Compress the WHOLE returned motion into the span
    (setpts) instead of cutting it off mid-arc, so the spliced timeline keeps
    its original length and pacing. Already-matching lengths pass through.
    Any failure (no ffmpeg/ffprobe, undecodable input) returns the ORIGINAL
    path — a too-long segment still splices, it is just long."""
    import shutil
    import subprocess

    if target_s <= 0 or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return Path(video_path)
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        actual = float(probe.stdout.strip())
    except Exception:
        return Path(video_path)
    if actual <= 0 or abs(actual - target_s) / target_s < 0.1:
        return Path(video_path)
    factor = target_s / actual
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"setpts={factor:.6f}*PTS", "-an",
             "-t", f"{target_s:.3f}", str(out_path)],
            capture_output=True, timeout=600,
        )
    except Exception:
        return Path(video_path)
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        return Path(video_path)
    return out_path


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
    fps: float = 0.0                     # real container fps; 0.0 = unknown

    @classmethod
    def from_clip(
        cls, clip, cache_dir, n_segments: int = 3,
        duration_s: Optional[float] = None,
    ) -> "ClipTimeline":
        """Split a clip into `n_segments` equal time spans, writing each
        segment's first/last boundary frame to `cache_dir` as PNG.

        `duration_s` (the shot's known length in seconds, e.g. spec.duration)
        is the fps fallback when the container does not report one: fps =
        n_frames / duration_s. Frame spans MUST convert to seconds before any
        generation call — the APIs take seconds, not frames.

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
        fps = _probe_fps(clip_path)
        if fps <= 0 and duration_s and float(duration_s) > 0:
            fps = len(frames) / float(duration_s)

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
                   degraded=degraded, cache_dir=cache_dir, fps=fps)

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
    head_anchor=None,        # head 跨度的左锚(该镜条件里的首帧图;可 None)
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

    # Segment spans are FRAMES; the generation APIs take SECONDS. Convert with
    # the real container fps (24.0 = the common API output rate, used only when
    # the container reports nothing and the caller gave no duration_s).
    fps = timeline.fps if timeline.fps > 0 else 24.0
    dur = max(1e-3, (seg.end_frame - seg.start_frame) / fps)

    # ── 2. repair S_i(2026-07-16 重做:flf2v 双锚,边界 = 相邻段的【原始】
    # 边界帧 —— 尾锚就是下游的原开头,下游天然连续,前向级联整条删除。
    # 上一版每次修复 = 段修复 + 最多 4 段级联 i2v(attempt2 实测 12 笔调用);
    # 新版 = 1 笔 flf2v。三种跨度:
    #   interior(左右邻都在)→ flf2v(左邻尾帧, 右邻首帧),免级联;
    #   tail(无右邻)      → i2v(左邻尾帧) 重生到尾,后面没东西,免级联;
    #   head(无左邻)      → 左锚 = head_anchor(该镜条件里的首帧图,调用方
    #                         传入);没有 → None(诚实降级,brain 改选整镜工具)。
    # 传统 i2v+级联只保留为"后端无 flf2v 能力/_same_shot 否决"的兼容兜底。──
    repaired: Optional[Path] = None
    out_i = cache_dir / f"seg{i}_repaired.mp4"
    left_anchor = (segs[i - 1].last_frame_path
                   if i > 0 and segs[i - 1].last_frame_path else
                   (Path(head_anchor) if i == 0 and head_anchor else None))
    right_anchor = (segs[i + 1].first_frame_path
                    if i + 1 < len(segs) and segs[i + 1].first_frame_path
                    else None)
    has_flf = "flf2v" in caps and hasattr(video_gen, "frame_to_frame")
    cascade_needed = False

    if left_anchor is None and i == 0:
        # head 跨度且无条件首帧可锚 —— 无法在不动第 0 帧的前提下双锚重生。
        log.info("propagate_repair: head span with no usable left anchor "
                 "(no first-frame condition image) — degrading to whole-clip "
                 "tools")
        return None
    if right_anchor is not None and has_flf             and _same_shot(left_anchor, right_anchor):
        # interior:双锚重生,右锚 = 下游原开头 ⇒ 免级联
        repaired = video_gen.frame_to_frame(
            prompt=hint or "one continuous passive trajectory",
            first_frame=left_anchor, last_frame=right_anchor,
            out_path=out_i, duration=dur,
        )
    elif right_anchor is None:
        # tail:从左锚重生到尾;后面没有内容,同样免级联
        repaired = video_gen.generate(
            prompt=hint or "regenerate this span faithfully",
            duration=dur, out_path=out_i, first_frame=left_anchor,
        )
    else:
        # 兼容兜底(无 flf2v 能力 / 锚不同镜):i2v 段修复 + 前向级联(旧路)
        anchor = left_anchor or seg.first_frame_path
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
        cascade_needed = True
    if repaired is None:
        return None
    # Fixed-duration APIs return ≥ the span (seedance min 5 s) — retime the
    # result back to the span length so the spliced timeline keeps its length.
    repaired = _fit_to_seconds(Path(repaired), dur,
                               cache_dir / f"seg{i}_repaired_fit.mp4")
    new_paths[i] = Path(repaired)
    # New last frame of the repaired segment (drives the cascade anchor).
    nl = extract_frame(repaired, 10**9, cache_dir / f"seg{i}_new_last.png")
    new_last[i] = nl or seg.last_frame_path

    # ── 3. forward cascade(仅兼容兜底路径需要;flf2v/tail 免级联)──────────
    if cascade_needed:
        end = min(len(segs) - 1, i + max_cascade)
        for j in range(i + 1, end + 1):
            anchor = new_last[j - 1]
            if anchor is None:               # lost the anchor → cannot continue
                break
            out_j = cache_dir / f"seg{j}_cascade.mp4"
            dur_j = max(1e-3, (segs[j].end_frame - segs[j].start_frame) / fps)
            regen = video_gen.generate(
                prompt=hint or "continue the shot; keep one continuous trajectory",
                duration=dur_j, out_path=out_j, first_frame=anchor,
            )
            if regen is None:
                break
            regen = _fit_to_seconds(Path(regen), dur_j,
                                    cache_dir / f"seg{j}_cascade_fit.mp4")
            new_paths[j] = Path(regen)
            new_j_last = extract_frame(
                regen, 10**9, cache_dir / f"seg{j}_new_last.png"
            )
            new_last[j] = new_j_last or segs[j].last_frame_path
            # Continuity reconverged? If the regenerated boundary matches the
            # OLD boundary, everything downstream of j is still valid — STOP.
            if new_j_last is not None and segs[j].last_frame_path is not None:
                sim = frame_similarity(new_j_last, segs[j].last_frame_path)
                if sim >= sim_threshold:
                    break

    # ── 4. splice everything back together ───────────────────────────────────
    out = cache_dir / "spliced_repair.mp4"
    spliced = _splice(new_paths, out)
    return spliced
