"""Audio I/O helpers.

Open-source dependencies (only imported when the optional ``music`` extra is
installed):
    • librosa   — https://librosa.org   (beats, tempo, RMS)
    • soundfile — https://pysoundfile.readthedocs.io

v0.1 ships only stubs so the rest of the pipeline can run on CPU without
pulling librosa. Real music analysis lives in
``perception.music_analyzer`` and uses **allin1**.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_audio_stub(path: Path | str, sr: int = 22050, duration: float = 30.0) -> Tuple[np.ndarray, int]:
    """Mock-load: returns deterministic silence so tests don't depend on a real wav."""
    n = int(sr * duration)
    return np.zeros(n, dtype=np.float32), sr


def load_audio(path: Path | str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Real loader. Imports librosa lazily to keep core install slim."""
    try:
        import librosa  # type: ignore
    except ImportError as e:
        raise ImportError(
            "load_audio() requires librosa. Install with: pip install 'longvideoagent[music]'"
        ) from e
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y, int(sr_out)


__all__ = ["load_audio_stub", "load_audio"]
