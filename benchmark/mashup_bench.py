"""Mashup-Bench / DIRECT-Bench adapter (v0.1 stub).

Reference dataset: https://github.com/AK-DREAM/DIRECT
v0.2 will populate ``load_cases`` from the upstream JSON manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


class MashupBenchAdapter:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def load_cases(self) -> Iterable[dict]:
        raise NotImplementedError(
            "MashupBenchAdapter.load_cases is a v0.2 task. Clone "
            "https://github.com/AK-DREAM/DIRECT and point root at the manifest."
        )
