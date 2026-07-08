"""VerifierAgent — enforces MONOTONIC IMPROVEMENT (M3 idea, hard rule).

Only accept a revised candidate if it is strictly better than the current best:
higher weighted metric total, or equal total with fewer failed checklist items.
This prevents 'fix one defect, break another' regressions.

Optional BLIND PAIRWISE CONFIRMATION (borrowed: NEWTON, arXiv:2605.18396 —
its only artifact verdict is a blind A/B comparison, never an absolute score):
when a `judge` MLLM is wired and the metric win is MARGINAL (inside the noise
band of VLM absolute scoring), the candidate must also survive a bidirectional
pairwise comparison against the current best on actual pixels. The judge can
only VETO a marginal win, never rescue a metric loss — monotonicity stays a
hard rule. A mock judge ties on same-content clips, so mock mode is unchanged.
"""
from __future__ import annotations

from typing import Optional

from ..types import CandidateClip, ShotSpec
from .base import BaseAgent


def _defect_count(clip: CandidateClip) -> int:
    """De-duplicated defect count: failed non-physics checklist items plus
    physics verdicts (physics-kind items mirror verdicts 1:1)."""
    failed_non_physics = sum(
        1 for i in clip.checklist.failed_items if i.kind != "physics"
    )
    return failed_non_physics + len(clip.physics_verdicts)


class VerifierAgent(BaseAgent):
    def __init__(self, judge=None, margin: float = 0.02, **kwargs):
        """`judge` — optional MLLM whose bidirectional compare() confirms
        MARGINAL metric wins on real pixels; `margin` — the metric band that
        counts as marginal (the VLM absolute-score noise scale)."""
        super().__init__(**kwargs)
        self.judge = judge
        self.margin = float(margin)

    def run(self, candidate: CandidateClip, best: CandidateClip | None) -> bool:
        return self.is_better(candidate, best)

    def is_better(
        self, candidate: CandidateClip, best: CandidateClip | None,
        eps: float = 1e-4, spec: Optional[ShotSpec] = None,
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
        judge_net = None
        if (better and self.judge is not None and spec is not None
                and (cand_total - best_total) <= self.margin):
            # Marginal metric win = inside VLM-scoring noise. Confirm on the
            # PIXELS with a bidirectional (order-debiased) pairwise compare; a
            # clear pixel-level preference for the OLD best vetoes the win.
            # Ties (mock judge / undecodable clips) keep the metric verdict.
            fwd = self.judge.compare(candidate, best, spec)   # +1 → candidate
            rev = self.judge.compare(best, candidate, spec)   # +1 → best
            judge_net = fwd - rev
            if judge_net < 0:
                better = False
        self._log(
            "verify",
            {"cand_total": round(cand_total, 4), "best_total": round(best_total, 4),
             "cand_failed": cand_failed, "best_failed": best_failed},
            {"accepted": better,
             **({"judge_net": judge_net} if judge_net is not None else {})},
        )
        return better
