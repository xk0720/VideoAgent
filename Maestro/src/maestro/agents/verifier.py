"""VerifierAgent — enforces MONOTONIC IMPROVEMENT (M3 idea, hard rule).

Only accept a revised candidate if it is strictly better than the current best:
higher weighted metric total, or equal total with fewer failed checklist items.
This prevents 'fix one defect, break another' regressions.
"""
from __future__ import annotations

from ..types import CandidateClip
from .base import BaseAgent


def _defect_count(clip: CandidateClip) -> int:
    """De-duplicated defect count: failed non-physics checklist items plus
    physics verdicts (physics-kind items mirror verdicts 1:1)."""
    failed_non_physics = sum(
        1 for i in clip.checklist.failed_items if i.kind != "physics"
    )
    return failed_non_physics + len(clip.physics_verdicts)


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
        # Defect count (tie-break only): every physics verdict also has a
        # MIRRORED failed checklist item appended by the critics, so counting
        # failed items + verdicts double-counted each physics defect. Count
        # failed NON-physics items + physics verdicts instead.
        cand_failed = _defect_count(candidate)
        best_failed = _defect_count(best)

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
