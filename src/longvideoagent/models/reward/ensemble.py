"""EnsembleRewardModel — multi-judge aggregation with disagreement signal.

Background (older + 2024–2025 evolution):
    • **Multi-Agent Debate** (Du et al., 2023)   — N agents argue, then vote.
      2024 successor: **DyLAN** (Liu et al., 2024) — dynamic LLM agent network
      that adapts topology by task.
    • **LLM-as-judge bias** (Zheng et al., NeurIPS 2023) — single-judge ranking
      is biased by position, length, self-preference. 2024: **MJ-Bench**
      quantifies these biases for multimodal judges.
    • **JudgeLM** (Zhu et al., 2024) — purpose-trained judge models that
      outperform GPT-4 on judgement tasks while being smaller.
    • **PandaLM** (Wang et al., 2024) — open judge for reproducible eval.
    • Constitutional AI (Bai et al., 2022) — multiple critiques per response;
      2024 successor: **Tülu-3** RLAIF pipeline with explicit multi-judge.

The idea: a single ValidatorAgent is one judge with one prompt; if it has a
systematic bias (over-praising long candidates, under-penalising motion
discontinuity, etc.), the whole pipeline inherits that bias. We mitigate by
running K reward models — each with a slightly different judge prompt or
backbone — and aggregating their scores. Disagreement between judges is *itself*
useful signal: high-variance candidates are excellent active-learning seeds.

For v0.1 the K models are typically:
    1. MockRewardModel (closed-form weighted sum of m1..m6)
    2. MLLMJudge with the default reward_judge.txt prompt
    3. (optional) MLLMJudge with a "harsh critic" prompt variant

When ``mocks.reward = True`` we can still build an ensemble of K
MockRewardModels with different weight vectors (one per heuristic preset),
so the disagreement signal is non-zero even hermetically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ...types import EditingSegment, SegmentGuidance
from .base import BaseRewardModel, RewardResult


@dataclass
class EnsembleResult(RewardResult):
    """RewardResult plus disagreement metadata."""
    per_judge_scores: list[float] = field(default_factory=list)
    judge_names: list[str] = field(default_factory=list)
    disagreement: float = 0.0      # population std of per_judge_scores

    def is_active_learning_candidate(self, threshold: float = 1.5) -> bool:
        """High variance ⇒ this candidate is worth flagging for human review."""
        return self.disagreement > threshold


class EnsembleRewardModel(BaseRewardModel):
    """Aggregate K reward models with mean-of-means + variance signal.

    The acceptance rule is *quorum-aware*: a candidate is accepted only if
    at least ``min_accept_quorum`` of the judges accepted it. This catches
    cases where one judge gives a high score for the wrong reasons but
    the others disagree — exactly the multi-agent-debate insight.
    """

    def __init__(
        self,
        judges: list[BaseRewardModel],
        accept_threshold: float = 6.0,
        min_accept_quorum: int | None = None,
    ) -> None:
        if not judges:
            raise ValueError("EnsembleRewardModel needs at least one judge")
        self.judges = judges
        self.accept_threshold = accept_threshold
        # default quorum = ceil(K/2)
        self.min_accept_quorum = min_accept_quorum or math.ceil(len(judges) / 2)

    def score(
        self,
        candidate: EditingSegment,
        guidance: SegmentGuidance,
        context: dict[str, Any] | None = None,
    ) -> EnsembleResult:
        per_scores: list[float] = []
        per_names: list[str] = []
        all_reasons: list[str] = []
        n_accepted = 0
        for j in self.judges:
            r = j.score(candidate, guidance, context)
            per_scores.append(r.score)
            per_names.append(type(j).__name__)
            all_reasons.extend(r.reasons)
            if r.accepted:
                n_accepted += 1
        mean = sum(per_scores) / len(per_scores)
        if len(per_scores) > 1:
            var = sum((s - mean) ** 2 for s in per_scores) / len(per_scores)
            stdev = math.sqrt(var)
        else:
            stdev = 0.0
        # quorum-aware acceptance
        accepted = (mean >= self.accept_threshold) and (n_accepted >= self.min_accept_quorum)
        if not accepted and n_accepted < self.min_accept_quorum:
            all_reasons.append(
                f"ensemble: only {n_accepted}/{len(self.judges)} judges accepted "
                f"(need {self.min_accept_quorum})."
            )
        return EnsembleResult(
            score=mean, accepted=accepted, reasons=all_reasons,
            per_judge_scores=per_scores, judge_names=per_names,
            disagreement=stdev,
        )


__all__ = ["EnsembleRewardModel", "EnsembleResult"]
