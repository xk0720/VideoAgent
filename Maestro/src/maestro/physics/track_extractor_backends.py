"""Real point-tracking track extractors (v0.3) — CoTracker / TAPIR.

These implement the SAME `BaseTrackExtractor.extract(...)` contract as the mock
in `physics/oracle.py`, so the oracle math and the whole pipeline are unchanged
— you only fill the model call and flip `models.track_extractor.name` in config.

────────────────────────────────────────────────────────────────────────────
What the extractor must do (oracle ground-truth recovery)
────────────────────────────────────────────────────────────────────────────
The oracle compares the simulator's EXPECTED per-entity screen tracks against
the motion OBSERVED in the generated clip. A real extractor recovers the
observed tracks by point tracking:

  1. decode the generated clip into frames (H×W×3);
  2. for each entity, seed ONE query point at its expected start position
     (normalized [0,1] → pixel), at t=0;
  3. run CoTracker / TAPIR → per-frame pixel tracks (+ visibility);
  4. normalize pixel tracks back to [0,1] screen space (÷W, ÷H) so they are
     directly comparable to `expected` (same convention as control_render's
     `_project_to_screen`, image y grows downward).

Borrowed (cite): CoTracker (Karaev et al., Meta), TAPIR/TAP-Vid (DeepMind).
OUR use: tracking is a *verification* instrument feeding the physics oracle —
not a controller. The tracker's quality bounds p2's sharpness, nothing else.

Graceful degradation (matches video_gen_backends / world_reward):
  • The generated clip must be a REAL decodable video. In the mock pipeline the
    "clip" is a text placeholder with no pixels → `extract()` returns None and
    the oracle stays silent (honest: you cannot track points in a non-video).
  • If torch / the tracker package / weights are missing while a real backend is
    explicitly configured, `_ensure_loaded()` raises LOUDLY rather than silently
    emitting a perfect (=wrong) p2.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .oracle import BaseTrackExtractor, Track


# ─────────────────────────────────────────────────────────────────────────────
# Frame decoding — tries decord → opencv → imageio; None if none available or
# the path is not a real video (e.g. a mock text placeholder).
# ─────────────────────────────────────────────────────────────────────────────
def _looks_like_video(p: Path) -> bool:
    """Cheap magic-byte sniff so we never hand a mock TEXT placeholder (which
    carries a .mp4 name) to a real decoder — that only spews ffmpeg/opencv
    'moov atom not found' noise before failing. Real containers are recognizable
    from their first bytes."""
    try:
        with open(p, "rb") as f:
            head = f.read(16)
    except Exception:
        return False
    if len(head) < 12:
        return False
    if head[4:8] == b"ftyp":                       # mp4 / mov / m4v
        return True
    if head[:4] == b"\x1aE\xdf\xa3":               # mkv / webm (EBML)
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":  # avi
        return True
    return False


def _decode_frames(path: Path):
    """Return an (T, H, W, 3) uint8 numpy array, or None if undecodable."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1024:
        return None
    if p.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        return None
    if not _looks_like_video(p):                   # text placeholder / corrupt
        return None
    # decord (fast, preferred)
    try:
        import decord  # type: ignore
        import numpy as np

        vr = decord.VideoReader(str(p))
        return vr[:].asnumpy()
    except Exception:
        pass
    # opencv
    try:
        import cv2  # type: ignore
        import numpy as np

        cap = cv2.VideoCapture(str(p))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames) if frames else None
    except Exception:
        pass
    # imageio
    try:
        import imageio.v3 as iio  # type: ignore

        return iio.imread(str(p), plugin="pyav")
    except Exception:
        return None


def _seed_queries(expected: dict[str, Track], width: int, height: int):
    """Build CoTracker queries [t, x_px, y_px] from each entity's start point.
    Returns (queries_list, ordered_entity_names)."""
    queries = []
    names = []
    for name, track in expected.items():
        if not track:
            continue
        x0, y0 = track[0]
        queries.append([0.0, x0 * width, y0 * height])
        names.append(name)
    return queries, names


class CoTrackerExtractor(BaseTrackExtractor):
    """CoTracker point tracking. One query point per entity (its start centroid).

    config:
      models.track_extractor:
        name: "cotracker"
        checkpoint: "/path/cotracker3.pth"   # or omit to use torch.hub
        device: "cuda"
    """

    def __init__(self, name: str = "cotracker", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.checkpoint = self.config.get("checkpoint") or os.getenv("COTRACKER_CKPT")
        self.device = self.config.get("device", "cuda")
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
        except Exception as exc:  # loud — configured but unusable
            raise RuntimeError(
                "CoTrackerExtractor needs torch. `pip install torch` and the "
                "co-tracker package, or set models.track_extractor.name back to "
                "'mock-track'."
            ) from exc
        try:
            import torch

            if self.checkpoint:
                from cotracker.predictor import CoTrackerPredictor  # type: ignore
                model = CoTrackerPredictor(checkpoint=self.checkpoint)
            else:
                # torch.hub fallback (downloads weights once)
                model = torch.hub.load("facebookresearch/co-tracker",
                                       "cotracker3_offline")
            self._model = model.to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"failed to load CoTracker (checkpoint={self.checkpoint}, "
                f"device={self.device}): {exc}"
            ) from exc

    def extract(self, clip, spec, expected, fps):
        frames = _decode_frames(Path(clip.video_path))
        if frames is None or len(frames) < 2:
            return None  # not a real/decodable video → oracle stays silent
        self._ensure_loaded()
        try:
            import torch

            T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
            queries, names = _seed_queries(expected, W, H)
            if not queries:
                return None
            video = (
                torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
            ).to(self.device)  # B T C H W
            q = torch.tensor(queries, dtype=torch.float32, device=self.device)[None]
            with torch.no_grad():
                pred_tracks, _vis = self._model(video, queries=q)
            # pred_tracks: B T N 2 (pixels) → normalize to [0,1] per entity.
            pt = pred_tracks[0].detach().cpu().numpy()  # T N 2
            observed: dict[str, Track] = {}
            for i, name in enumerate(names):
                observed[name] = [
                    (float(pt[t, i, 0] / W), float(pt[t, i, 1] / H))
                    for t in range(pt.shape[0])
                ]
            # entities with no query (empty expected track) → empty observed
            for name in expected:
                observed.setdefault(name, [])
            return observed
        except Exception:
            # A tracking failure is non-fatal: stay silent rather than crash.
            return None


class TAPIRExtractor(BaseTrackExtractor):
    """TAPIR / TAP-Vid point tracking (DeepMind). Same contract as CoTracker;
    thinner skeleton — fill `_ensure_loaded` + the inference call.

    config: models.track_extractor.name: "tapir"
    """

    def __init__(self, name: str = "tapir", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.checkpoint = self.config.get("checkpoint") or os.getenv("TAPIR_CKPT")
        self.device = self.config.get("device", "cuda")
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        raise RuntimeError(
            "TAPIRExtractor not wired yet. Implement _ensure_loaded()/extract() "
            "with tapnet, or use models.track_extractor.name = 'cotracker' / "
            "'mock-track'."
        )

    def extract(self, clip, spec, expected, fps):
        frames = _decode_frames(Path(clip.video_path))
        if frames is None or len(frames) < 2:
            return None
        self._ensure_loaded()  # raises until wired
        return None


_REGISTRY = {
    "cotracker": CoTrackerExtractor,
    "tapir": TAPIRExtractor,
}


def build_real_track_extractor(name: str, config: Optional[dict] = None) -> BaseTrackExtractor:
    key = name.split("-")[0].lower() if name else ""
    cls = _REGISTRY.get(name.lower()) or _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"unknown track_extractor backend '{name}'. known: {list(_REGISTRY)} "
            "(+ 'mock-track')"
        )
    return cls(name=name, config=config)
