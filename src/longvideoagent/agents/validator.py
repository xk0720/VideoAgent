"""ValidatorAgent — thin wrapper around a BaseRewardModel.

In v0.1 the reward model is MockRewardModel / MLLMJudge (mocked); in v0.2
it becomes a fine-tuned EditingQualityRM (design doc §12 path B).
"""
from __future__ import annotations

from typing import Any

from ..models.reward.base import BaseRewardModel, RewardResult
from ..types import EditingSegment, SegmentGuidance
from .base import BaseAgent


class ValidatorAgent(BaseAgent):
    name = "validator"

    def __init__(self, reward_model: BaseRewardModel, trajectory_logger=None) -> None:
        # We use a *no-op LLM* dependency by passing a MockLLMClient — the
        # judge itself is the reward_model. This keeps the BaseAgent signature.
        from ..models.llm.base import MockLLMClient
        super().__init__(
            llm_client=MockLLMClient(alias="validator"),
            prompt_template="reward_judge",
            trajectory_logger=trajectory_logger,
        )
        self.reward_model = reward_model

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        # Convenience batch interface (the per-candidate call is .score()).
        results = []
        for cand, guidance in state.get("pairs", []):
            results.append(self.score(cand, guidance))
        return {"validator_results": results}

    def score(self, candidate: EditingSegment, guidance: SegmentGuidance) -> RewardResult:
        result = self.reward_model.score(candidate, guidance)
        self.log_step(
            action="judge",
            action_input={"segment_idx": candidate.segment_idx,
                          "semantic_query": guidance.semantic_query},
            observation={"score": result.score, "accepted": result.accepted},
            reward=result.score,
        )
        return result


__all__ = ["ValidatorAgent"]
