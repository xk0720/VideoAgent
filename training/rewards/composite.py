"""CompositeReward — α·RM + β·mean(m1..m6) + γ·beat_alignment.

This is the reward we feed to GRPO. The proposal doc says:
    r = α * RM_score + β * mean(m1..m6) + γ * beat_alignment_bonus

We keep this small and pure-numpy so it can be JIT-evaluated inside
verl / OpenRLHF rollout loops without pulling extra deps.

Reward-hacking defenses (per AGENTIC_RL_PROPOSAL §4):
    • ODIN-style disentangle: caller can pass weights=0 to mask any term.
    • Ensemble RM consumer: pass an EnsembleRewardModel as ``rm`` and the
      EnsembleResult.disagreement signal flows out via ``last_disagreement``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from longvideoagent.models.reward.base import BaseRewardModel, RewardResult
from longvideoagent.models.reward.ensemble import EnsembleResult
from longvideoagent.types import EditingSegment, MusicProfile, SegmentGuidance


@dataclass
class CompositeRewardResult:
    total: float
    rm_score: float
    metric_mean: float
    beat_bonus: float
    disagreement: float = 0.0
    reasons: list[str] = field(default_factory=list)


class CompositeReward:
    def __init__(
        self,
        rm: BaseRewardModel,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        beat_sigma: float = 0.15,
    ) -> None:
        self.rm = rm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.beat_sigma = beat_sigma
        self.last_disagreement = 0.0

    def __call__(
        self,
        candidate: EditingSegment,
        guidance: SegmentGuidance,
        music: Optional[MusicProfile] = None,
    ) -> CompositeRewardResult:
        rm_res: RewardResult = self.rm.score(candidate, guidance)
        rm_score = rm_res.score / 10.0                # → [0, 1]
        m = candidate.metric_scores or {}
        metric_mean = sum(float(m.get(f"m{i}", 0.0)) for i in range(1, 7)) / 6.0
        beat_bonus = self._beat_bonus(candidate, music)
        total = (self.alpha * rm_score
                 + self.beta * metric_mean
                 + self.gamma * beat_bonus)
        disagreement = (rm_res.disagreement if isinstance(rm_res, EnsembleResult)
                        else 0.0)
        self.last_disagreement = disagreement
        return CompositeRewardResult(
            total=total, rm_score=rm_score, metric_mean=metric_mean,
            beat_bonus=beat_bonus, disagreement=disagreement,
            reasons=list(rm_res.reasons),
        )

    def _beat_bonus(self, cand: EditingSegment, music: Optional[MusicProfile]) -> float:
        if music is None or not music.beats or not cand.shot_trims:
            return 0.0
        beats = music.beats
        starts = [t[0] for t in cand.shot_trims]
        deltas = [min(abs(s - b) for b in beats) for s in starts]
        # Gaussian decay around nearest beat.
        from math import exp
        decayed = [exp(-(d ** 2) / (2 * self.beat_sigma ** 2)) for d in deltas]
        return sum(decayed) / len(decayed)


__all__ = ["CompositeReward", "CompositeRewardResult"]
