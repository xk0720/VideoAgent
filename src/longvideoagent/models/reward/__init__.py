"""Reward / quality-judge models.

v0.1: ``MLLMJudge`` is a thin wrapper around an MLLM (mocked).
v0.2: replace with a fine-tuned EditingQualityRM / NarrativeCoherenceRM.
"""
from .base import BaseRewardModel, MockRewardModel, RewardResult
from .mllm_judge import MLLMJudge
from .ensemble import EnsembleRewardModel, EnsembleResult

__all__ = [
    "BaseRewardModel", "RewardResult", "MockRewardModel", "MLLMJudge",
    "EnsembleRewardModel", "EnsembleResult",
]
