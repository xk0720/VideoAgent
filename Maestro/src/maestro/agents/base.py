"""BaseAgent ABC. Each agent: run(...) -> output, logging every step."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..models.llm import BaseLLMClient, MockLLMClient
from ..trajectory import TrajectoryLogger


def load_prompt(path: Optional[Path]) -> str:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return ""


class BaseAgent(ABC):
    def __init__(
        self,
        llm: Optional[BaseLLMClient] = None,
        prompt_path: Optional[Path] = None,
        config: Optional[dict] = None,
        logger: Optional[TrajectoryLogger] = None,
    ):
        self.llm = llm or MockLLMClient()
        self.prompt_template = load_prompt(prompt_path)
        self.config = config or {}
        self.logger = logger

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _log(self, action: str, action_input: dict, observation: dict) -> None:
        if self.logger:
            self.logger.append(self.name, action, action_input, observation)

    @abstractmethod
    def run(self, *args, **kwargs):
        ...
