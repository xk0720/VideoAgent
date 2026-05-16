"""CineBench adapter (v0.1 stub).

Reference: https://github.com/MikeWangWZHL/CineBench (cinematic-style benchmark).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


class CineBenchAdapter:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def load_cases(self) -> Iterable[dict]:
        raise NotImplementedError(
            "CineBenchAdapter.load_cases is a v0.2 task. Clone CineBench and "
            "implement manifest parsing here."
        )
