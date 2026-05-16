"""Saliency-map extraction (boundary frames).

Open-source library wrapped here:
    • **U²-Net**  https://github.com/xuebinqin/U-2-Net
      (No first-class PyPI package; we vendor weights in v0.2.)

v0.1 mock returns a unimodal Gaussian-blob saliency map so retrieval
metric m4 (Wasserstein distance between saliency maps at the cut) is
deterministic and non-trivial.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PreprocessCfg


class SaliencyExtractor:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False,
                 height: int = 32, width: int = 32) -> None:
        self.cfg = cfg
        self.mock = mock
        self.h, self.w = height, width

    def extract_boundary_saliency(self, video_path: Path | str,
                                  start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
        if self.mock:
            return self._gaussian_blob(start_s), self._gaussian_blob(end_s)
        return self._gaussian_blob(start_s), self._gaussian_blob(end_s)

    def _gaussian_blob(self, seed: float) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
        cy = rng.integers(self.h // 4, 3 * self.h // 4)
        cx = rng.integers(self.w // 4, 3 * self.w // 4)
        sigma = max(2.0, min(self.h, self.w) / 5.0)
        ys, xs = np.mgrid[0:self.h, 0:self.w]
        m = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
        return (m / m.sum()).astype("float32")


__all__ = ["SaliencyExtractor"]
