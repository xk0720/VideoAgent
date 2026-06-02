"""PhysicsConsistencyCritic — close the loop on the C1 sketch layer.

WHY THIS IS NEW (vs. existing methods):
  • VISTA (arXiv:2510.15831) judges physics with a VLM scoring "physical commonsense"
    over the rendered pixels alone — no mid-level physical ground truth to compare to.
  • Event-Graph (arXiv:2604.10383) does enforce physics, but via a 3D engine rendering
    the final pixels — it gives up photorealism in exchange.
  • Maestro v0.1/v0.2 ships the sketch's `control_signal` *forward* into the generator
    but never verifies that the rendered video actually *followed* the predicted
    trajectory. That made physics grounding one-way: sim → generator, no closed loop.

WHAT THIS CRITIC ADDS:
  A second physics critic that *cross-checks* the generated clip against the same
  physics-sketch the generator was conditioned on. If the implied motion of the
  rendered clip diverges from the simulated trajectory, we flag it as a CONSERVATION
  failure (a generator that ignored the control signal). This makes the sketch layer
  bidirectional — sim is now BOTH a control input AND a verification reference.

MOCK SEMANTICS (v0.2, CPU-only):
  In the mock pipeline the generator writes a metadata file whose body records
  whether it received the `control_signal=...` argument. We parse that as a proxy
  for "did the generator follow the sketch?". v0.3: swap for an optical-flow or
  point-tracking estimator that recovers each entity's trajectory from the actual
  video frames and compares it to the simulated tracks.
"""
from __future__ import annotations

from typing import Optional

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
    """Verify the rendered clip actually followed the physics sketch (C6, new)."""

    kind = "physics"

    def __init__(self, divergence_threshold: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.threshold = divergence_threshold

    def _divergence(self, clip: CandidateClip, spec: ShotSpec) -> float:
        """Mock divergence in [0,1]; 0 = follows sketch perfectly, 1 = ignored it.

        Heuristic: the mock generator writes `control_signal=<path>` into its
        metadata file iff it was actually conditioned on the sketch. We read that
        line — when the value is `None`, divergence is high; otherwise low. We
        also taper divergence with the revision count (refinements should
        gradually align the clip to the sketch, mirroring real test-time
        improvement).
        """
        # No sketch was supplied at all -> nothing to be consistent with.
        if not spec.physics_sketch or not spec.physics_sketch.control_signal:
            return 0.0
        try:
            body = clip.video_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return 0.0
        # The mock backend records `control_signal=<path-or-None>`; real backends
        # would instead read estimated trajectories from the frames.
        followed = "control_signal=None" not in body
        base = 0.15 if followed else 0.7
        # Refinements drive the implied motion closer to the sketch.
        decayed = max(0.0, base - 0.15 * clip.revision)
        return round(decayed, 3)

    def review(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> None:
        div = self._divergence(clip, spec)
        if div >= self.threshold:
            # Surface as a CONSERVATION verdict — the generator didn't conserve
            # the sketch's predicted momentum/trajectory.
            n_frames = max(1, int(round(spec.duration * fps)))
            verdict = PhysicsVerdict(
                mode=PhysFailureMode.CONSERVATION,
                frame_range=(0, n_frames),
                severity=div,
                suggested_intervention=(
                    "tighten conditioning on the sketch's trajectory control signal; "
                    "feed first-frame anchor derived from sim"
                ),
            )
            clip.physics_verdicts.append(verdict)
            clip.checklist.items.append(
                ChecklistItem(
                    question="Does the rendered motion follow the physics sketch?",
                    kind="physics",
                    passed=False,
                    fix_instruction=verdict.suggested_intervention,
                )
            )
        # We do NOT write to clip.metric_scores here: ReviewBoard's MetricTool
        # owns that dict and derives p2_sketch_consistency from the CONSERVATION
        # verdict above. Keeping ownership single avoids "stash then get wiped".
        self._log(
            {"shot_idx": spec.shot_idx, "revision": clip.revision},
            {"divergence": div, "flagged": div >= self.threshold},
        )
