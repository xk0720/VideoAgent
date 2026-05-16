"""Reward-model ABC.

v0.1 uses ``MockRewardModel`` which returns a stable score derived from the
candidate's pre-computed metric_scores dict, so EditorAgent's accept/reject
behaviour is deterministic in tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

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
    """Linear combination of provided metric_scores. Deterministic.

    The default weights mirror ``configs/heuristics/presets.yaml::default``,
    but callers can pass any weight dict to simulate a different "judge" —
    handy when building an EnsembleRewardModel of K diverse mock judges
    (e.g. one weighted toward m1, another toward m3, another toward m5).
    """

    _DEFAULT_WEIGHTS = {"m1": 0.20, "m2": 0.15, "m3": 0.20, "m4": 0.15, "m5": 0.15, "m6": 0.15}

    def __init__(
        self,
        accept_threshold: float = 6.0,
        weights: Optional[dict[str, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.accept_threshold = accept_threshold
        self.weights = weights or dict(self._DEFAULT_WEIGHTS)
        if name:
            self.__class__ = type(name, (MockRewardModel,), {})  # nicer repr in ensemble

    def score(self, candidate, guidance, context=None) -> RewardResult:
        m = candidate.metric_scores or {}
        s = sum(self.weights[k] * float(m.get(k, 0.5)) for k in self.weights) * 10.0
        s = max(1.0, min(10.0, s))
        return RewardResult(score=s, accepted=s >= self.accept_threshold)
