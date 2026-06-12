"""PhysicsConsistencyCritic — physics-from-pixels verification (C6, v0.4).

WHAT CHANGED vs v0.3:
  • v0.3 compared observed motion against a SIMULATED expectation. That still
    presumed the simulator knew masses/restitution/scale from one prompt — a
    reviewer-fatal assumption ("your simulator is wrong, not the video").
  • v0.4 is REFERENCE-FREE: a track extractor recovers observed motion, a
    reliability gate certifies the tracks (trackers lie on generated video),
    and the law layer asks the parameter-free question "is there ANY
    physically consistent explanation for this track?" (best-law residual +
    anomaly localization). Entities the measurement tier cannot check are
    routed to world_model / vlm tiers and reported as explicit deferrals.

Precedents (cite): Morpheus 2504.02918 (conservation residuals, benchmarking
only), PISA 2503.09595 (trajectory residual as reward, sim ground truth),
equation-discovery forecasting 2507.06830 (parametric fit to observed tracks).
OUR increment: measured, interpretable, per-entity localized verdicts that
drive best-of-N selection AND targeted repair (HSI), training-free, with
tracker-reliability gating nobody else does (survey_physics_2026_06.md §4.1).
"""
from __future__ import annotations

from typing import Optional

from ..physics.tracks import BaseTrackExtractor
from ..physics.verifier import PhysicsFromPixelsVerifier
from ..types import (
    AssetMemory,
    CandidateClip,
    ChecklistItem,
    PhysFailureMode,
    PhysicsVerdict,
    ShotSpec,
)
from .base import BaseCritic


class PhysicsConsistencyCritic(BaseCritic):
    """Reference-free physical-consistency verdicts from observed pixels."""

    kind = "physics"

    def __init__(
        self,
        violation_threshold: float = 0.4,
        extractor: Optional[BaseTrackExtractor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = violation_threshold
        self.verifier = PhysicsFromPixelsVerifier(extractor=extractor)

    def review(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> None:
        result = self.verifier.verify(clip, spec, fps)
        if result is None:
            # No annotation / clip unreadable -> stay silent (nothing was
            # verified; avoids emitting confident verdicts from no evidence).
            self._log({"shot_idx": spec.shot_idx, "revision": clip.revision},
                      {"verifier": "silent"})
            return

        # HSI tier-1 replans tighten the bar (strictness > 1.0)
        strictness = (
            spec.physics_annotation.strictness if spec.physics_annotation else 1.0
        )
        threshold = self.threshold / max(strictness, 1e-6)
        n_frames = max(1, int(round(spec.duration * fps)))

        for report in result.measured_reports:
            if report.violation < threshold:
                continue
            if report.anomalies:
                worst = max(report.anomalies, key=lambda a: a.severity)
                mode, frame_range, why = worst.mode, worst.frame_range, worst.note
            else:
                # Residual-only violation: nothing localized to blame, just
                # "no passive law fits" — labeling that gravity/inertia would
                # mislead the lesson/skill indexes. UNEXPLAINED is honest.
                mode = PhysFailureMode.UNEXPLAINED
                frame_range = (0, n_frames)
                why = (f"no consistent motion law (best '{report.fit.law}' "
                       f"residual {report.fit.residual:.2f})")
            verdict = PhysicsVerdict(
                mode=mode,
                frame_range=frame_range,
                severity=report.violation,
                source="law_verifier",
                suggested_intervention=(
                    f"observed motion of '{report.entity}' has no physical "
                    f"explanation ({why}); regenerate this span with the "
                    "entity following one continuous passive trajectory; "
                    "consider best-of-N reseed"
                ),
            )
            clip.physics_verdicts.append(verdict)
            clip.checklist.items.append(
                ChecklistItem(
                    question=(f"Is the observed motion of '{report.entity}' "
                              "physically explainable?"),
                    kind="physics",
                    passed=False,
                    fix_instruction=verdict.suggested_intervention,
                )
            )

        # MetricTool owns metric_scores and derives p2 from the verdicts above
        # (single ownership, no stash-then-wipe).
        self._log(
            {"shot_idx": spec.shot_idx, "revision": clip.revision},
            {
                "worst_violation": result.worst_violation,
                "per_entity": {r.entity: r.violation
                               for r in result.measured_reports},
                "coverage": result.coverage,           # tier -> entities (S3)
                "uncertified": result.uncertified,     # gated out (S2)
                "threshold": round(threshold, 3),
                "flagged": result.worst_violation >= threshold,
            },
        )
