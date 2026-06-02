"""MetricTool — the quantitative basis of the self-improvement loop (C3).

Reuses the old repo's metric ideas (m1/m2/m5/m6) and adds physics (p1) and
identity (id1). Scores are in [0,1], higher is better. v0.1 computes them
deterministically from the candidate's checklist / physics verdicts / spec so the
loop is observable and testable; v0.2 swaps in real CLIP/flow/saliency/VLM signals.

Exposed as a tool so agents can *query specific scores* (not a black box).
"""
from __future__ import annotations

from typing import Optional

from .base import BaseTool
from ..types import AssetMemory, CandidateClip, PhysFailureMode, ShotSpec


class MetricTool(BaseTool):
    name = "compute_metrics"

    def __init__(self, weights: Optional[dict[str, float]] = None):
        # p2_sketch_consistency (C6) is the closed-loop check: did the generator
        # follow the physics sketch? Kept distinct from p1 so a failure to track
        # the sketch is debuggable separately from a native physics violation.
        self.weights = weights or {
            "m1_semantic": 0.22,
            "m2_temporal": 0.13,
            "p1_physics": 0.22,
            "p2_sketch_consistency": 0.10,
            "id1_identity": 0.13,
            "m5_rhythm": 0.10,
            "aesthetic": 0.10,
        }

    def run(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> dict[str, float]:
        rev = clip.revision

        # m1 semantic: checklist pass rate
        m1 = clip.checklist.pass_rate

        # Split verdicts: native physics failures vs. closed-loop sketch divergence
        # (CONSERVATION verdicts come from PhysicsConsistencyCritic only).
        native = [v for v in clip.physics_verdicts if v.mode != PhysFailureMode.CONSERVATION]
        conserv = [v for v in clip.physics_verdicts if v.mode == PhysFailureMode.CONSERVATION]
        # p1 physics: 1 - worst native severity
        p1 = max(0.0, 1.0 - max((v.severity for v in native), default=0.0))
        # p2 sketch consistency (C6): 1 - worst CONSERVATION severity
        p2 = max(0.0, 1.0 - max((v.severity for v in conserv), default=0.0))

        # id1 identity consistency: better when refs exist and across revisions
        if spec.identity_refs:
            id1 = min(1.0, 0.6 + 0.15 * rev)
        else:
            id1 = 0.8  # nothing to keep consistent -> neutral-high

        # m2 temporal consistency: improves with revision (proxy)
        m2 = min(1.0, 0.6 + 0.1 * rev)

        # m5 rhythm/beat sync: reward having a pacing plan aligned to music
        has_music = bool(asset_memory and asset_memory.music_profile)
        if spec.rhythmic_pacing and has_music:
            m5 = 0.9
        elif spec.rhythmic_pacing:
            m5 = 0.7
        else:
            m5 = 0.5

        aesthetic = min(1.0, 0.65 + 0.08 * rev)

        scores = {
            "m1_semantic": round(m1, 3),
            "m2_temporal": round(m2, 3),
            "p1_physics": round(p1, 3),
            "p2_sketch_consistency": round(p2, 3),
            "id1_identity": round(id1, 3),
            "m5_rhythm": round(m5, 3),
            "aesthetic": round(aesthetic, 3),
        }
        scores["weighted_total"] = round(
            sum(self.weights.get(k, 0.0) * v for k, v in scores.items()), 4
        )
        return scores
