"""Open-vocabulary detection backends (v0.4) — GroundingDINO / mock.

WHY THIS EXISTS (the closed break in physics-from-pixels)
─────────────────────────────────────────────────────────
The physics verifier recovers a per-entity track with CoTracker and asks the
law layer whether that track is physically explicable. CoTracker needs ONE
query point per named entity at t=0. Until v0.4 that seed was a hardcoded
evenly-spaced screen position — so on a REAL video CoTracker tracked an
arbitrary pixel, NOT "the ball", and the per-entity law verdict was
meaningless. These detectors close that link: detect the named entity in
frame 0, take its bbox centroid, and seed CoTracker THERE.

  prompt entity name  →  detect(frame0, name)  →  bbox centroid  →  CoTracker seed

Borrowed (cite): GroundingDINO (Liu et al., IDEA-Research) — open-vocabulary
zero-shot detection, here via the HuggingFace transformers route
(AutoModelForZeroShotObjectDetection) so it is hub-loadable with no separate
weights download step. STRICTLY inference-only (training-free).

Honesty boundary: GroundingDINO is trained on REAL images; detection QUALITY
on synthetically generated video is the remaining honest limit (and it needs a
GPU + weights). When the entity is NOT grounded the extractor falls back to the
heuristic seed and WARNS that the per-entity verdict is unreliable. The
MockDetector is the cold-start / CPU path (text-deterministic, ignores pixels).

House style mirrors models/audio_gen_backends.py: ABC + factory, lazy heavy
imports inside methods, loud RuntimeError when a configured real backend's deps
are missing, non-fatal WARNING on a per-call inference failure.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from ..logging_utils import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# ABC — the contract the extractor / DetectionTool delegate to
# ─────────────────────────────────────────────────────────────
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame_rgb, query: str, max_results: int = 1) -> list[dict]:
        """Detect `query` in an (H,W,3) uint8 RGB frame.

        Returns a list of dicts sorted by score desc:
          {"label": str, "bbox": [x0,y0,x1,y1] normalized [0,1], "score": float}
        Empty list ⇒ not found (the caller seeds heuristically and warns).
        """
        ...

    @staticmethod
    def centroid(box) -> tuple[float, float]:
        """Normalized bbox [x0,y0,x1,y1] → (cx, cy) center, normalized."""
        x0, y0, x1, y1 = box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# ─────────────────────────────────────────────────────────────
# Mock — deterministic bboxes from the query nouns (ignores pixels)
# ─────────────────────────────────────────────────────────────
class MockDetector(BaseDetector):
    """Text-deterministic detector mirroring the v0.2.2 DetectionTool mock so
    its byte-shape stays back-compatible. Splits the query into nouns and spreads
    one bbox per noun across the frame. Ignores `frame_rgb` entirely — it has no
    pixels grounding, which is exactly why the per-entity physics verdict is
    unreliable on this path (the real GroundingDINO backend closes that gap)."""

    def __init__(self, name: str = "mock-detect"):
        self.name = name

    def detect(self, frame_rgb, query: str, max_results: int = 1) -> list[dict]:
        terms = [t for t in (query or "").replace(",", " ").split() if len(t) > 2][
            :max_results
        ]
        if not terms:
            terms = ["subject"]
        out = []
        for i, t in enumerate(terms):
            x0 = round(0.1 + 0.25 * i, 3)
            out.append({
                "label": t,
                "bbox": [x0, 0.3, round(x0 + 0.2, 3), 0.7],   # [x0,y0,x1,y1]
                "score": round(0.9 - 0.1 * i, 2),
            })
        return out


# ─────────────────────────────────────────────────────────────
# GroundingDINO — REAL zero-shot detection (HuggingFace route)
# ─────────────────────────────────────────────────────────────
class GroundingDINODetector(BaseDetector):
    """Open-vocabulary detection via transformers' zero-shot OD head.

    config (models.track_extractor.detector):
      name: "groundingdino"
      model: "IDEA-Research/grounding-dino-tiny"   # HF hub id
      device: "cuda"
      box_threshold:  0.3
      text_threshold: 0.25
    """

    def __init__(self, name: str = "groundingdino", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.model_id = (
            self.config.get("model")
            or os.getenv("GROUNDINGDINO_MODEL")
            or "IDEA-Research/grounding-dino-tiny"
        )
        self.device = self.config.get("device", "cuda")
        self.box_threshold = float(self.config.get("box_threshold", 0.3))
        self.text_threshold = float(self.config.get("text_threshold", 0.25))
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except Exception as exc:  # loud — configured but unusable
            raise RuntimeError(
                "GroundingDINODetector needs torch+transformers; "
                "`pip install torch transformers` or set "
                "models.track_extractor.detector.name back to 'mock'."
            ) from exc
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"failed to load GroundingDINO (model={self.model_id}, "
                f"device={self.device}): {exc}"
            ) from exc

    def detect(self, frame_rgb, query: str, max_results: int = 1) -> list[dict]:
        self._ensure_loaded()
        try:
            import torch
            from PIL import Image

            image = Image.fromarray(frame_rgb)
            W, H = image.size  # PIL: (width, height)
            # GroundingDINO expects a lowercase, period-terminated text prompt.
            text = f"{query.strip().lower()}."
            inputs = self._processor(images=image, text=text, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(H, W)],
            )[0]
            boxes = results["boxes"].detach().cpu().numpy()    # x0,y0,x1,y1 px
            scores = results["scores"].detach().cpu().numpy()
            labels = results.get("labels") or results.get("text_labels") or []
            dets = []
            for i in range(len(boxes)):
                x0, y0, x1, y1 = (float(v) for v in boxes[i])
                dets.append({
                    "label": (labels[i] if i < len(labels) else query),
                    "bbox": [x0 / W, y0 / H, x1 / W, y1 / H],   # normalize [0,1]
                    "score": float(scores[i]),
                })
            dets.sort(key=lambda d: d["score"], reverse=True)
            return dets[:max_results]
        except Exception as exc:
            # Non-fatal (caller falls back to heuristic seed) but VISIBLE.
            log.warning(
                "GroundingDINO detection failed for query %r: %r — entity will "
                "be seeded heuristically (physics verdict for it is unreliable)",
                query, exc,
            )
            return []


# ─────────────────────────────────────────────────────────────
# Factory — mirrors models/audio_gen_backends.build_audio_gen
# ─────────────────────────────────────────────────────────────
def build_detector(spec: str | dict | None) -> BaseDetector:
    """None / "mock*" → MockDetector; "groundingdino"/"dino"/"grounding-dino" →
    GroundingDINODetector; anything else → ValueError."""
    name = "mock-detect"
    config: dict = {}
    if isinstance(spec, dict):
        name = spec.get("name", name)
        config = spec
    elif isinstance(spec, str):
        name = spec
    key = (name or "").lower()
    if key.startswith("mock"):
        return MockDetector(name=name)
    if key in ("groundingdino", "dino", "grounding-dino"):
        return GroundingDINODetector(name=name, config=config)
    raise ValueError(
        f"Unknown detector backend '{name}'. Known: mock*, groundingdino, "
        "dino, grounding-dino"
    )
