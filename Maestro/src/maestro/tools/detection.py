"""DetectionTool — tracking category. Object detection + lightweight tracking.

UniVA exposes detection/tracking as first-class so a Director can ask "where is
the protagonist" and condition generation on the bbox. Maestro uses it for two
things:
  • Identity grounding: lock a face/object bbox across the source video so
    the Generator's reference_images feed stays consistent (E1).
  • Sketch verification (C6): future hook — point trajectories estimated here
    are what PhysicsConsistencyCritic compares to the sim trajectory.

v0.2.2: mock returns deterministic bboxes based on prompt keywords. v0.3 wires
Grounding-DINO / SAM / a tracker.
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseTool


class DetectionTool(BaseTool):
    name = "detect_objects"
    category = "tracking"
    description = "Detect objects (and optionally track them) in an image/video; returns bboxes."

    def run(
        self,
        media: str | Path,
        query: str = "subject",
        max_results: int = 3,
    ) -> list[dict]:
        # Mock: split the query into nouns and emit one bbox per noun. The bbox
        # placement is deterministic from the noun text so tests are stable.
        terms = [t for t in query.replace(",", " ").split() if len(t) > 2][:max_results]
        if not terms:
            terms = ["subject"]
        out = []
        for i, t in enumerate(terms):
            # Spread bboxes across the frame; coordinates in [0,1].
            x0 = round(0.1 + 0.25 * i, 3)
            out.append({
                "label": t,
                "bbox": [x0, 0.3, round(x0 + 0.2, 3), 0.7],   # [x0,y0,x1,y1]
                "score": round(0.9 - 0.1 * i, 2),
                "source": str(media),
            })
        return out
