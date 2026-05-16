"""Reward-model ABC.

v0.1 uses ``MockRewardModel`` which returns a stable score derived from the
candidate's pre-computed metric_scores dict, so EditorAgent's accept/reject
behaviour is deterministic in tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...types import EditingSegment, SegmentGuidance


@dataclass
class RewardResult:
    score: float                                # 1..10
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    raw: Any = None


class BaseRewardModel(ABC):
    @abstractmethod
    def score(
        self,
        candidate: EditingSegment,
        guidance: SegmentGuidance,
        context: dict[str, Any] | None = None,
    ) -> RewardResult: ...


class MockRewardModel(BaseRewardModel):
    """Linear combination of provided metric_scores. Deterministic."""

    def __init__(self, accept_threshold: float = 6.0) -> None:
        self.accept_threshold = accept_threshold

    def score(self, candidate, guidance, context=None) -> RewardResult:
        m = candidate.metric_scores or {}
        # Default weights mirror configs/heuristics/presets.yaml::default.
        weights = {"m1": 0.20, "m2": 0.15, "m3": 0.20, "m4": 0.15, "m5": 0.15, "m6": 0.15}
        s = sum(weights[k] * float(m.get(k, 0.5)) for k in weights) * 10.0
        s = max(1.0, min(10.0, s))
        return RewardResult(score=s, accepted=s >= self.accept_threshold)
