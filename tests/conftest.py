"""Pytest fixtures.

Open-source dependencies (test-only):
    • pytest          https://docs.pytest.org

We auto-generate ``tests/fixtures/tiny_clip.mp4`` on first run via
ffmpeg (or a cv2 fallback) so the repo can stay binary-free in git.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).parent / "fixtures"
FIX.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def tiny_clip_path() -> Path:
    p = FIX / "tiny_clip.mp4"
    if not p.exists():
        from longvideoagent.utils.video_io import write_silent_color_clip
        write_silent_color_clip(p, duration_s=4.0, fps=24, width=160, height=120,
                                color=(50, 120, 200))
    return p


@pytest.fixture(scope="session")
def tmp_cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture
def deterministic_rng() -> np.random.Generator:
    return np.random.default_rng(42)
