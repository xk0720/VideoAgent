"""PhysicsConsistencyCritic — sim-as-ORACLE verification (C6, repositioned v0.3).

WHAT CHANGED vs v0.2 (see PHYSICS_LITERATURE_REVIEW.md):
  • OLD framing: "did the generator obey the control signal we conditioned it
    on?" — implemented by reading a `control_signal=...` metadata line. That
    presumed sketch-as-conditioning, which the literature does not support
    (trajectory conditioning on frozen models is trained-only and trajectories
    under-determine physics; review §2-3).
  • NEW framing: the simulator is a VERIFICATION ORACLE. We compare the motion
    the sim says SHOULD happen against the motion OBSERVED in the generated
    clip via a track extractor (mock now; CoTracker/RAFT in v0.3), PISA-style
    normalized Trajectory-L2 (arXiv:2503.09595). Conditioning, if any, is
    irrelevant to the verdict — only observed pixels matter.

Direct precedents (cite): PISA 2503.09595 (trajectory-L2 vs sim truth),
Morpheus 2504.02918 (tracked quantities vs conservation laws), PhyCoBench
2502.05503 (flow deviation), Physics-IQ 2501.09038 (predicted vs real frames).
OUR increment: the oracle's verdict is localized (worst entity, severity) and
feeds an agentic repair loop (Refiner/HSI), not just a benchmark score.
"""
from __future__ import annotations

from typing import Optional

from ..physics.oracle import BaseTrackExtractor, TrajectoryOracle
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
    """Verify observed motion matches the sim-predicted motion (oracle check)."""

    kind = "physics"

    def __init__(
        self,
        divergence_threshold: float = 0.4,
        extractor: Optional[BaseTrackExtractor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = divergence_threshold
        self.oracle = TrajectoryOracle(extractor=extractor)

    def review(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> None:
        cmp = self.oracle.compare(clip, spec, fps)
        if cmp is None:
            # No sketch / no predicted motion / extraction failed -> stay silent
            # (nothing to verify; avoids spurious CONSERVATION noise).
            self._log({"shot_idx": spec.shot_idx, "revision": clip.revision},
                      {"oracle": "silent"})
            return

        if cmp.deviation >= self.threshold:
            n_frames = max(1, int(round(spec.duration * fps)))
            verdict = PhysicsVerdict(
                mode=PhysFailureMode.CONSERVATION,
                frame_range=(0, n_frames),
                severity=cmp.deviation,
                suggested_intervention=(
                    f"observed motion of '{cmp.worst_entity}' diverges from the "
                    "simulated expectation; re-anchor keyframes to physically "
                    "plausible poses (apex/contact) and regenerate; consider "
                    "best-of-N reseed"
                ),
            )
            clip.physics_verdicts.append(verdict)
            clip.checklist.items.append(
                ChecklistItem(
                    question="Does the observed motion match the physics oracle's prediction?",
                    kind="physics",
                    passed=False,
                    fix_instruction=verdict.suggested_intervention,
                )
            )
        # MetricTool owns metric_scores and derives p2 from the CONSERVATION
        # verdict above (single ownership, no stash-then-wipe).
        self._log(
            {"shot_idx": spec.shot_idx, "revision": clip.revision},
            {"deviation": cmp.deviation, "per_entity": cmp.per_entity,
             "worst_entity": cmp.worst_entity, "flagged": cmp.deviation >= self.threshold},
        )
