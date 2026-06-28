"""DetectionTool — tracking category. Object detection + lightweight tracking.

UniVA exposes detection/tracking as first-class so a Director can ask "where is
the protagonist" and condition generation on the bbox. Maestro uses it for two
things:
  • Identity grounding: lock a face/object bbox across the source video so
    the Generator's reference_images feed stays consistent (E1).
  • Physics verification seeding (C6, v0.4): detected entity centroids are
    what seed the track extractor's query points
    (physics/track_extractor_backends.py) — the recovered tracks then feed
    the reference-free law checks. There is no sim trajectory to compare to.

v0.4: delegates to a configured `BaseDetector` (models/detection_backends.py):
  • MockDetector (default) — deterministic bboxes from prompt nouns; ignores
    pixels, so it's byte-for-byte the v0.2.2 mock and keeps every test stable.
  • GroundingDINODetector — REAL zero-shot detection (needs torch+transformers).

The default client is MockDetector, so `default_registry()` and existing tests
stay unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.detection_backends import BaseDetector, MockDetector
from .base import BaseTool


class DetectionTool(BaseTool):
    name = "detect_objects"
    category = "tracking"
    description = "Detect objects (and optionally track them) in an image/video; returns bboxes."

    def __init__(self, client: Optional[BaseDetector] = None):
        # Default mock keeps the key-free CPU path identical to v0.2.2.
        self.client: BaseDetector = client or MockDetector()

    def run(
        self,
        media: str | Path,
        query: str = "subject",
        max_results: int = 3,
    ) -> list[dict]:
        # Mock path: no pixels needed — the detector emits one deterministic bbox
        # per query noun. Byte-identical to v0.2.2 (tests depend on it).
        if isinstance(self.client, MockDetector):
            out = self.client.detect(None, query, max_results=max_results)
            for d in out:
                d["source"] = str(media)
            return out

        # Real path: `media` is a path. Load frame 0 (PIL for an image, the video
        # decoder for a clip) and hand the actual pixels to the detector.
        frame = self._load_frame0(media)
        out = self.client.detect(frame, query, max_results=max_results)
        for d in out:
            d["source"] = str(media)
        return out

    @staticmethod
    def _load_frame0(media: str | Path):
        """Return an (H,W,3) uint8 RGB ndarray for the first frame of `media`."""
        p = Path(media)
        suffix = p.suffix.lower()
        if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            from ..physics.track_extractor_backends import _decode_frames

            frames = _decode_frames(p)
            if frames is None or len(frames) == 0:
                raise ValueError(f"could not decode a frame from video: {media}")
            return frames[0]
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(p).convert("RGB"))
