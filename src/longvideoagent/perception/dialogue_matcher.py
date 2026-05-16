"""Dialogue extraction & speaker matching.

Open-source libraries wrapped here (v0.2):
    • **EasyOCR**     (pip: ``easyocr``)            https://github.com/JaidedAI/EasyOCR
      (PaddleOCR is a drop-in alternative.)
    • **WeSpeaker**   (pip: ``wespeaker``)          https://github.com/wenet-e2e/wespeaker
      Used for voiceprint clustering → speaker → character mapping.

v0.1 mock returns ``None`` (no dialogue extracted) and is stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import PreprocessCfg


class DialogueMatcher:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = mock

    def extract(self, video_path: Path | str, start_s: float, end_s: float) -> Optional[str]:
        if self.mock:
            return None
        return None                                             # v0.2: real OCR + voiceprint


__all__ = ["DialogueMatcher"]
