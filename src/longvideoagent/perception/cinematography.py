"""Cinematography tagging (shot scale / movement / angle / framing).

Open-source library wrapped here:
    • **ShotVL / ShotBench**  (HF: ``Vchitect/ShotBench-3B``)
      Loaded via ``transformers`` AutoModel.

v0.1 mock returns a deterministic, varied set of tags based on shot
duration so the rest of the pipeline sees realistic-looking metadata.
"""
from __future__ import annotations

from pathlib import Path

from ..config import PreprocessCfg
from ..types import CinematographyTags


_SCALES = ["close_up", "medium", "long", "extreme_long", "extreme_close_up"]
_MOVES = ["static", "pan", "tilt", "zoom", "tracking", "handheld"]
_ANGLES = ["eye_level", "low", "high", "dutch", "overhead"]
_FRAMES = ["single", "two_shot", "group", "ows", "pov"]


class CinematographyTagger:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = mock

    def tag(self, video_path: Path | str, start_s: float, end_s: float) -> CinematographyTags:
        if self.mock:
            seed = abs(hash((str(video_path), start_s, end_s))) % 1000
            return CinematographyTags(
                shot_scale=_SCALES[seed % len(_SCALES)],        # type: ignore[arg-type]
                shot_movement=_MOVES[(seed // 7) % len(_MOVES)],
                shot_angle=_ANGLES[(seed // 13) % len(_ANGLES)],
                framing=_FRAMES[(seed // 17) % len(_FRAMES)],
            )
        return CinematographyTags()


__all__ = ["CinematographyTagger"]
