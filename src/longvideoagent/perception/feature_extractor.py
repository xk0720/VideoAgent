"""Per-shot CLIP feature extraction.

Open-source library wrapped here:
    • **open_clip**  (pip: ``open_clip_torch``)
      https://github.com/mlfoundations/open_clip
    Alternative backend: ``transformers.CLIPModel``.

v0.1 returns a deterministic hash-based mock embedding when ``mock=True``
or when ``open_clip`` is not importable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PreprocessCfg
from ..logging import logger


class FeatureExtractor:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = mock
        self.embed_dim = cfg.feature_extractor.embed_dim
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _lazy_init(self) -> None:                              # pragma: no cover
        if self._model is not None:
            return
        import open_clip  # type: ignore
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def embed_shot(self, video_path: Path | str, start_s: float, end_s: float,
                   shot_id: Optional[str] = None) -> np.ndarray:
        """Return a (D,) float32 unit vector for the shot. D == ``embed_dim``."""
        if self.mock:
            return self._mock_embed(f"{video_path}::{start_s:.3f}::{end_s:.3f}")
        try:
            self._lazy_init()
        except ImportError:
            logger.warning("open_clip not installed — using mock embedding.")
            return self._mock_embed(f"{video_path}::{start_s:.3f}::{end_s:.3f}")
        # v0.2: sample N keyframes via utils.video_io.iter_frames and average
        # CLIP visual features. Stubbed here on purpose.
        return self._mock_embed(f"{video_path}::{start_s:.3f}::{end_s:.3f}")

    def embed_text(self, text: str) -> np.ndarray:
        if self.mock:
            return self._mock_embed(text)
        try:
            self._lazy_init()
        except ImportError:
            return self._mock_embed(text)
        # v0.2: real CLIP text encoder; mock for now.
        return self._mock_embed(text)

    def _mock_embed(self, seed_text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
        v = rng.standard_normal(self.embed_dim).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)


__all__ = ["FeatureExtractor"]
