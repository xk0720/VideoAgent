"""Character identification across shots.

Open-source libraries wrapped here:
    • **InsightFace**  (pip: ``insightface``)
      https://github.com/deepinsight/insightface
    Fallback / re-ID for full-body crops:
    • **SOLIDER**      https://github.com/tinyvision/SOLIDER (no pypi; vendor)

v0.1 mock assigns one synthetic character id ("char_0") so the rest of
the pipeline never sees a missing field. v0.2 will:
    1. Run InsightFace face detection + embedding on sampled frames.
    2. Cluster across all shots (e.g. agglomerative w/ cosine threshold).
    3. Propagate IDs along tracker trajectories.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PreprocessCfg
from ..types import Character


class CharacterIdentifier:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = mock

    def identify_shot(self, video_path: Path | str, start_s: float, end_s: float) -> list[str]:
        """Return list of character_ids visible in the shot."""
        if self.mock:
            return ["char_0"]
        return ["char_0"]                                       # v0.2: real impl

    def build_character_bank(self) -> list[Character]:
        """Return a list of Character entities after all shots are processed."""
        if self.mock:
            return [Character(
                character_id="char_0",
                name="Protagonist",
                face_embedding=np.zeros(512, dtype="float32"),
                appearance_shot_ids=[],
                profile_summary="(mock) primary on-screen character.",
            )]
        return []


__all__ = ["CharacterIdentifier"]
