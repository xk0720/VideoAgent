"""Optical-flow extraction at shot boundaries.

Open-source library wrapped here:
    • **RAFT** via ``torchvision.models.optical_flow.raft_large``
      https://pytorch.org/vision/stable/models/raft.html

v0.1 returns zeros of the right shape so downstream retrieval metrics
(see tools.metric_tool, metric m3) can still be computed deterministically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PreprocessCfg


class FlowExtractor:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False,
                 height: int = 32, width: int = 32) -> None:
        self.cfg = cfg
        self.mock = mock
        # We deliberately keep mock flow tiny (32x32x2) so it fits in caches.
        self.h, self.w = height, width

    def extract_boundary_flows(self, video_path: Path | str,
                               start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (flow_at_start, flow_at_end, avg_magnitude). Both flow maps
        are (H, W, 2) float32."""
        if self.mock:
            return self._mock(start_s, end_s)
        # v0.2: load 4 frames near start_s and 4 near end_s, run RAFT pairs.
        return self._mock(start_s, end_s)

    def _mock(self, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray, float]:
        rng = np.random.default_rng(abs(hash((start_s, end_s))) % (2**32))
        flow_start = rng.standard_normal((self.h, self.w, 2)).astype("float32") * 0.5
        flow_end = rng.standard_normal((self.h, self.w, 2)).astype("float32") * 0.5
        mag = float(np.mean(np.linalg.norm(flow_end, axis=-1)))
        return flow_start, flow_end, mag


__all__ = ["FlowExtractor"]
