"""Reward / quality-judge models.

v0.1: ``MLLMJudge`` is a thin wrapper around an MLLM (mocked).
v0.2: replace with a fine-tuned EditingQualityRM / NarrativeCoherenceRM.
"""
from .base import BaseRewardModel, MockRewardModel
from .mllm_judge import MLLMJudge

__all__ = ["BaseRewardModel", "MockRewardModel", "MLLMJudge"]
