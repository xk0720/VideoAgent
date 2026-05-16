"""BaseAgent ABC.

All five agents subclass BaseAgent. The LLM client is injected (so tests
can pass MockLLMClient); the prompt template is loaded from a .txt file
on disk (never hardcoded — design doc §15 first anti-pattern).

This module only depends on:
    • Python stdlib
    • longvideoagent.config (pydantic-loaded)
    • longvideoagent.models.llm.BaseLLMClient (ABC, importable without `openai`)
"""
from __future__ import annotations

import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..config import load_prompt
from ..logging import logger
from ..models.llm.base import BaseLLMClient
from ..utils.trajectory import TrajectoryLogger


def extract_placeholders(template: str) -> set[str]:
    """Return the set of named placeholders ``{foo}`` referenced by ``template``.

    Built on top of stdlib ``string.Formatter.parse``; ignores ``{{``/``}}``
    escapes and positional ``{0}`` references.
    """
    fmt = string.Formatter()
    out: set[str] = set()
    for _literal, field_name, _spec, _conv in fmt.parse(template):
        if field_name and not field_name.isdigit():
            # Strip attribute / index suffixes (".x", "[0]") to keep just the root name.
            root = field_name.split(".")[0].split("[")[0]
            if root:
                out.add(root)
    return out


class BaseAgent(ABC):
    name: str = "base"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: str | Path,
        config: Optional[dict[str, Any]] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None,
    ) -> None:
        self.llm = llm_client
        self.prompt_template = load_prompt(str(prompt_template))
        self.required_placeholders: set[str] = extract_placeholders(self.prompt_template)
        self.config = config or {}
        self.trajectory_logger = trajectory_logger

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Take a state snapshot, return partial state update."""

    # ─── helpers ───

    def render_prompt(self, **kwargs: Any) -> str:
        missing = self.required_placeholders - set(kwargs)
        if missing:
            msg = (f"[{self.name}] render_prompt missing placeholder(s): "
                   f"{sorted(missing)}; template needs {sorted(self.required_placeholders)}")
            logger.error(msg)
            raise KeyError(msg)
        try:
            return self.prompt_template.format(**kwargs)
        except (KeyError, IndexError) as e:                     # pragma: no cover
            logger.error(f"[{self.name}] prompt format failed: {e!r}")
            raise

    def log_step(
        self,
        action: str,
        action_input: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        state_snapshot: dict[str, Any] | None = None,
        reward: float | None = None,
    ) -> None:
        if self.trajectory_logger is None:
            return
        self.trajectory_logger.log(
            agent_name=self.name,
            action=action,
            action_input=action_input or {},
            observation=observation or {},
            state_snapshot=state_snapshot or {},
            reward=reward,
        )


__all__ = ["BaseAgent", "extract_placeholders"]
