"""Real point-tracking track extractors (v0.4) — CoTracker / TAPIR.

These implement the SAME `BaseTrackExtractor.extract(...)` contract as the
mock in `physics/tracks.py`, so the law checks and the whole pipeline are
unchanged — you only fill the model call and flip
`models.track_extractor.name` in config.

────────────────────────────────────────────────────────────────────────────
What the extractor must do (observed-track recovery, reference-free)
────────────────────────────────────────────────────────────────────────────
There is no expected trajectory anywhere (the sketch line is dead). The
extractor's only job is to recover what actually moves in the clip:

  1. decode the generated clip into frames (H×W×3);
  2. seed ONE query point per annotated entity at t=0 by DETECTION: each
     entity is grounded in frame 0 with an open-vocabulary detector
     (GroundingDINO, models/detection_backends.py) and CoTracker is seeded at
     the detected centroid — so it tracks the ACTUAL named entity, not an
     arbitrary pixel. If an entity can't be grounded (or the default
     MockDetector path, which ignores pixels) the seed falls back to an
     evenly-spaced heuristic and a WARNING marks that entity's verdict
     unreliable on real video;
  3. run CoTracker / TAPIR → per-frame pixel tracks (+ visibility);
  4. normalize pixel tracks to [0,1] screen space (÷W, ÷H; y grows downward)
     for physics/laws.py.

Borrowed (cite): CoTracker (Karaev et al., Meta), TAPIR/TAP-Vid (DeepMind).
OUR use: tracking is a *verification* instrument feeding reference-free law
checks — never a controller. Reliability gating (physics/reliability.py)
decides whether each recovered track can be trusted at all.

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

from ..logging_utils import get_logger
from .laws import Track
from .tracks import BaseTrackExtractor

log = get_logger(__name__)


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


def _heuristic_seed_xy(i: int, k: int, width: int, height: int):
    """Heuristic seed for entity i of k: evenly spaced across the upper-center
    band. Used ONLY when detection can't ground the entity in frame 0 — a blind
    fallback, so the per-entity physics verdict is unreliable on real video."""
    return (i + 1) / (k + 1) * width, 0.35 * height


def _heuristic_seed(entities, width: int, height: int):
    """Build CoTracker queries [t, x_px, y_px] without detection — one evenly
    spaced seed per entity. Returns (queries, names). (Detection-grounded
    seeding is _grounded_seed; this is the no-detector / fallback path.)"""
    names = [e.name for e in entities]
    k = len(names)
    queries = [
        [0.0, *_heuristic_seed_xy(i, k, width, height)] for i in range(k)
    ]
    return queries, names


def _grounded_seed(frames, entities, detector, width: int, height: int):
    """Detection-grounded CoTracker seeding (the closed break).

    For each entity, run `detector.detect(frames[0], entity.name)`; if a box is
    found seed at its centroid (in PIXELS); otherwise fall back to the heuristic
    position for THAT entity and WARN (its physics verdict is unreliable). Order
    is preserved. Pure function → unit-testable without torch.

    Returns (queries [[0.0, x_px, y_px], ...], names).
    """
    names = [e.name for e in entities]
    k = len(names)
    queries = []
    for i, ent in enumerate(entities):
        box = None
        try:
            dets = detector.detect(frames[0], ent.name, max_results=1)
            if dets:
                box = dets[0].get("bbox")
        except Exception as exc:  # detector blew up → heuristic + warn
            log.warning(
                "detector raised on entity '%s': %r; seeding heuristically "
                "(physics verdict for it is unreliable)", ent.name, exc,
            )
        if box is not None:
            cx, cy = detector.centroid(box)        # normalized
            x_px, y_px = cx * width, cy * height
        else:
            x_px, y_px = _heuristic_seed_xy(i, k, width, height)
            log.warning(
                "could not ground entity '%s' in frame 0; seeding heuristically "
                "(physics verdict for it is unreliable)", ent.name,
            )
        queries.append([0.0, x_px, y_px])
    return queries, names


class CoTrackerExtractor(BaseTrackExtractor):
    """CoTracker point tracking. One query point per entity (its start centroid).

    Query points are SEEDED by detection: each entity is detected in frame 0
    (config `detector` sub-block → models/detection_backends.build_detector) and
    CoTracker is seeded at the detected centroid, so it tracks the ACTUAL named
    entity, not an arbitrary pixel. If the detector can't ground an entity (or
    is the default MockDetector, which ignores pixels) the seed falls back to an
    evenly-spaced heuristic and a WARNING marks that entity's verdict unreliable.

    config:
      models.track_extractor:
        name: "cotracker"
        checkpoint: "/path/cotracker3.pth"   # or omit to use torch.hub
        device: "cuda"
        detector:                            # → real grounding (else heuristic)
          name: "groundingdino"
          model: "IDEA-Research/grounding-dino-tiny"
          device: "cuda"
    """

    def __init__(self, name: str = "cotracker", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.checkpoint = self.config.get("checkpoint") or os.getenv("COTRACKER_CKPT")
        self.device = self.config.get("device", "cuda")
        self._model = None
        # Detector that grounds each entity in frame 0 (default → MockDetector,
        # which is text-deterministic → heuristic seeding on real video).
        from ..models.detection_backends import build_detector
        self.detector = build_detector(self.config.get("detector"))

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

    def extract(self, clip, spec, entities, fps):
        """Recover one normalized track per entity, or None on failure.

        Inference failures are NON-FATAL (None → the verifier stays silent)
        but are logged at WARNING level: a persistent inference bug would
        otherwise disable physics verification forever with no trace in the
        server logs. Metric semantics of the silent path: p2 stays NEUTRAL —
        'no measured violation' is not 'verified', and the coverage report
        makes the gap explicit."""
        frames = _decode_frames(Path(clip.video_path))
        if frames is None or len(frames) < 2:
            return None  # not a real/decodable video → verifier stays silent
        self._ensure_loaded()
        try:
            import torch

            T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
            # Detection-grounded seeding: track WHERE the named entity is, not a
            # fixed pixel (falls back to the heuristic + warns per ungrounded
            # entity). This is the closed physics-from-pixels break.
            queries, names = _grounded_seed(frames, entities, self.detector, W, H)
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
            return observed
        except Exception as exc:
            # Non-fatal (verifier stays silent, p2 stays neutral) but VISIBLE:
            # a silent `return None` here would mask a persistent inference
            # bug as "nothing to verify" forever.
            log.warning(
                "CoTracker inference failed on %s (%d frames, %d entities): %r "
                "— physics verification skipped for this clip",
                clip.video_path, len(frames), len(entities), exc,
            )
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

    def extract(self, clip, spec, entities, fps):
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
