"""Binary tournament candidate selection with bidirectional de-biasing.

Borrowed (cite): VISTA — A Test-Time Self-Improving Video Generation Agent
(arXiv:2510.15831). VISTA selects among candidate videos via a pairwise
"Binary Tournament" and compares each pair in BOTH orders to cancel the MLLM
judge's position/token bias.

Why this over ViMax's "generate N, let a VLM pick one": bidirectional pairwise
comparison is more robust to judge bias than a single multi-way pick. We reuse it
as the candidate selector that seeds Maestro's self-improvement loop (E3).
"""
from __future__ import annotations

from typing import Optional

from ..models.mllm import BaseMLLMClient, MockMLLMClient
from ..types import CandidateClip, ShotSpec


class Tournament:
    def __init__(self, judge: Optional[BaseMLLMClient] = None):
        self.judge = judge or MockMLLMClient()

    def _bidirectional(self, a: CandidateClip, b: CandidateClip, spec: ShotSpec) -> int:
        """+1 if a wins, -1 if b wins, 0 tie. Compares (a,b) and (b,a)."""
        fwd = self.judge.compare(a, b, spec)          # +1 -> a
        rev = self.judge.compare(b, a, spec)          # +1 -> b
        net = fwd - rev                                # de-biased aggregate
        if net > 0:
            return 1
        if net < 0:
            return -1
        return 0

    def select(self, candidates: list[CandidateClip], spec: ShotSpec) -> CandidateClip:
        if not candidates:
            raise ValueError("no candidates to select from")
        champion = candidates[0]
        for challenger in candidates[1:]:
            if self._bidirectional(challenger, champion, spec) > 0:
                champion = challenger
        return champion
