"""BaseCritic ABC. A critic reviews a CandidateClip and appends checklist items
(and possibly physics verdicts) in place."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..trajectory import TrajectoryLogger
from ..types import AssetMemory, CandidateClip, ShotSpec


class BaseCritic(ABC):
    kind: str = "base"

    def __init__(self, logger: Optional[TrajectoryLogger] = None, config: Optional[dict] = None):
        self.logger = logger
        self.config = config or {}

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _log(self, action_input: dict, observation: dict) -> None:
        if self.logger:
            self.logger.append(self.name, "review", action_input, observation)

    @abstractmethod
    def review(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> None:
        ...
