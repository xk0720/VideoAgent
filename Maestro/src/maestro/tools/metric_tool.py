"""MetricTool — the quantitative basis of the self-improvement loop (C3).

Reuses the old repo's metric ideas (m1/m2/m5/m6) and adds physics (p1) and
identity (id1). Scores are in [0,1], higher is better. v0.1 computes them
deterministically from the candidate's checklist / physics verdicts / clip
body / spec so the loop is observable and testable; v0.2 swaps in real
CLIP/flow/saliency/VLM signals.

SIGNAL HONESTY (see models/mock_signals.py): a mock may simulate the WORLD —
a generator that responds to repair instructions. But critics and metrics
must read the ARTIFACT (the clip's content), never the revision counter. If
a metric is `f(revision)`, the whole self-improve loop is a clock, not a
feedback system: regenerating WITHOUT applying the fix would still
"improve" — exactly the failure mode documented in ../docs/CRITICAL_REVIEW.md
(parent repo) §meta-error-1. Every mock proxy below is derived from the clip
body (applied fixes, first-frame anchoring, reference conditioning) or from
review state — never from clip.revision.

Exposed as a tool so agents can *query specific scores* (not a black box).
"""
from __future__ import annotations

from typing import Optional

from .base import BaseTool
from ..models.mock_signals import (
    applied_fixes,
    first_frame_anchored,
    reference_images_present,
)
from ..types import AssetMemory, CandidateClip, ShotSpec

# Mock proxy constants (content-derived; real backends replace the whole
# computation, not the constants).
M2_BASE = 0.6           # unanchored, unrepaired temporal coherence
M2_ANCHOR_BONUS = 0.15  # clip was conditioned on a first-frame keyframe anchor
M2_PER_FIX_BONUS = 0.05  # each applied repair instruction steadies the motion
ID1_CONDITIONED = 0.9   # identity refs requested AND reference images applied
ID1_UNCONDITIONED = 0.6  # identity refs requested but NOT conditioned on
ID1_NEUTRAL = 0.8       # nothing to keep consistent -> neutral-high
AESTHETIC_FLAT = 0.7    # see comment at use site


class MetricTool(BaseTool):
    name = "compute_metrics"
    category = "metric"
    description = "Score a CandidateClip on m1/m2/p1/p2/id1/m5/aesthetic and weighted_total."

    def __init__(self, weights: Optional[dict[str, float]] = None, world_reward=None):
        # p2_law_consistency (C6 v0.4) is the MEASURED check: does the
        # observed motion have any physically consistent explanation (law
        # residual + anomalies)? Kept distinct from p1 (VLM-judged) so a
        # measured violation is debuggable separately from a judged one.
        self.weights = weights or {
            "m1_semantic": 0.22,
            "m2_temporal": 0.13,
            "p1_physics": 0.22,
            "p2_law_consistency": 0.10,
            "id1_identity": 0.13,
            "m5_rhythm": 0.10,
            "aesthetic": 0.10,
        }
        # Optional BaseWorldReward (models/world_reward.py). When set, adds a
        # `wm_reward` dimension — turning the existing best-of-N + monotonic
        # Verifier into WMReward-style test-time search (arXiv:2601.10553).
        # None (default) keeps output keys identical to v0.2.2 (back-compat).
        self.world_reward = world_reward

    def run(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> dict[str, float]:
        # Content-derived repair depth: fixes actually recorded in the clip
        # body, NOT the revision counter (see module docstring).
        n_fixes = len(applied_fixes(clip))

        # m1 semantic: checklist pass rate
        m1 = clip.checklist.pass_rate

        # Split verdicts by evidence source: judged (VLM critic) vs measured
        # (reference-free law verifier).
        judged = [v for v in clip.physics_verdicts if v.source != "law_verifier"]
        measured = [v for v in clip.physics_verdicts if v.source == "law_verifier"]
        # p1 physics: 1 - worst judged severity
        p1 = max(0.0, 1.0 - max((v.severity for v in judged), default=0.0))
        # p2 law consistency (C6): 1 - worst measured severity
        p2 = max(0.0, 1.0 - max((v.severity for v in measured), default=0.0))

        # id1 identity consistency: when refs are requested, the honest mock
        # signal is whether the clip was ACTUALLY conditioned on reference
        # images (recorded in the body) — not how many times we regenerated.
        if spec.identity_refs:
            id1 = ID1_CONDITIONED if reference_images_present(clip) else ID1_UNCONDITIONED
        else:
            id1 = ID1_NEUTRAL  # nothing to keep consistent -> neutral-high

        # m2 temporal consistency: keyframe anchoring + applied repairs are
        # the honest mock proxies for temporal coherence (both read from the
        # clip body).
        m2 = min(1.0, M2_BASE
                 + (M2_ANCHOR_BONUS if first_frame_anchored(clip) else 0.0)
                 + M2_PER_FIX_BONUS * n_fixes)

        # m5 rhythm/beat sync: reads the PLAN (spec.rhythmic_pacing vs the
        # music profile), not the clip — plan-derived by design, no revision
        # signal involved.
        has_music = bool(asset_memory and asset_memory.music_profile)
        if spec.rhythmic_pacing and has_music:
            m5 = 0.9
        elif spec.rhythmic_pacing:
            m5 = 0.7
        else:
            m5 = 0.5

        # No mock signal exists for aesthetics; flat constant, NOT a fake
        # ramp — the real backend wires an aesthetic predictor here.
        aesthetic = AESTHETIC_FLAT

        scores = {
            "m1_semantic": round(m1, 3),
            "m2_temporal": round(m2, 3),
            "p1_physics": round(p1, 3),
            "p2_law_consistency": round(p2, 3),
            "id1_identity": round(id1, 3),
            "m5_rhythm": round(m5, 3),
            "aesthetic": round(aesthetic, 3),
        }
        if self.world_reward is not None:
            scores["wm_reward"] = round(
                self.world_reward.score(clip, spec, fps), 3
            )
        scores["weighted_total"] = round(
            sum(self.weights.get(k, 0.0) * v for k, v in scores.items()), 4
        )
        return scores
