"""VerifierAgent — enforces MONOTONIC IMPROVEMENT (M3 idea, hard rule).

Only accept a revised candidate if it is strictly better than the current best:
higher weighted metric total, or equal total with fewer failed checklist items.
This prevents 'fix one defect, break another' regressions.
"""
from __future__ import annotations

from ..types import CandidateClip
from .base import BaseAgent


class VerifierAgent(BaseAgent):
    def run(self, candidate: CandidateClip, best: CandidateClip | None) -> bool:
        return self.is_better(candidate, best)

    def is_better(
        self, candidate: CandidateClip, best: CandidateClip | None, eps: float = 1e-4
    ) -> bool:
        if best is None:
            return True
        cand_total = candidate.metric_scores.get("weighted_total", 0.0)
        best_total = best.metric_scores.get("weighted_total", 0.0)
        cand_failed = len(candidate.checklist.failed_items) + len(candidate.physics_verdicts)
        best_failed = len(best.checklist.failed_items) + len(best.physics_verdicts)

        better = (cand_total > best_total + eps) or (
            abs(cand_total - best_total) <= eps and cand_failed < best_failed
        )
        self._log(
            "verify",
            {"cand_total": round(cand_total, 4), "best_total": round(best_total, 4),
             "cand_failed": cand_failed, "best_failed": best_failed},
            {"accepted": better},
        )
        return better
