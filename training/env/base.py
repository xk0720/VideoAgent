"""AgentEnvBase — gymnasium-style ABC for RL envs that wrap our agents.

Conformance:
    • signature mirrors **OpenAI Gym 0.26+ / gymnasium**:
        reset(seed) → obs, info
        step(action) → obs, reward, terminated, truncated, info
    • action / observation spaces are intentionally loose (any JSON-able dict)
      to accommodate the structured outputs LLM agents produce; this matches
      the Open Reward Standard (ORS) "tool call as action" pattern, where the
      action is itself a structured dict not a gym.Box.
    • all methods are synchronous; the async wrapping for vLLM batched
      rollouts is done at the **runner** layer (training/runners/rollout.py),
      not here — same pattern as verl 0.7 AgentLoop.

References:
    https://gymnasium.farama.org/
    https://openrewardstandard.io
    https://verl.readthedocs.io/en/latest/start/agentic_rl.html
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EnvObservation:
    """Structured observation returned by env.reset/step.

    Keep it dataclass-based so the RL trainer can pickle / serialize cleanly;
    LLM agents will see this as a JSON dump.
    """
    state: dict[str, Any]
    available_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvStepResult:
    """One env.step() result. Mirrors gymnasium's 5-tuple but as a dataclass."""
    observation: EnvObservation
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class AgentEnvBase(ABC):
    """ABC for any RL environment wrapping a LongVideoEditAgent agent.

    Subclasses must implement ``reset`` and ``step``; everything else has
    a sensible default.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        """Start a new episode. Return the initial observation."""

    @abstractmethod
    def step(self, action: dict[str, Any]) -> EnvStepResult:
        """Apply one action; return next observation + reward + done flags."""

    def render(self) -> Optional[str]:
        """Optional textual render of current state (handy for trajectory log)."""
        return None

    def close(self) -> None:
        """Release any resources (no-op in v0.1)."""

    # ─── ORS-style helpers ────────────────────────────────────────────

    def tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-style tool schemas the agent may call.

        Mirrors ORS' "agent interacts with env only via tool calls" principle.
        Default: an empty tool set; subclasses should override.
        """
        return []
