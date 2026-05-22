"""AgentPolicyBase — RL policy ABC.

OpenRLHF (https://openrlhf.readthedocs.io/en/latest/async_rl.html) defines:

    class AgentInstanceBase:
        async def initialize(self, ...): ...
        async def reset(self, ...): ...
        async def step(self, prompt, ...): ...

We use a synchronous version because our env is synchronous; an async
wrapper (``runners/rollout.py``) wraps multiple sync policies in asyncio for
high-throughput vLLM inference, matching verl 0.7 AgentLoop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PolicyOutput:
    """One action proposal from the policy.

    ``log_prob`` is optional in v0.1 (mock policy doesn't return one); real
    RL trainers will require it for importance sampling in GRPO/PPO.
    """
    action: dict[str, Any]
    log_prob: Optional[float] = None
    raw_text: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class AgentPolicyBase(ABC):
    """Synchronous policy interface."""

    @abstractmethod
    def reset(self) -> None:
        """Called once per episode start."""

    @abstractmethod
    def act(self, observation, context_messages: list[dict[str, str]]) -> PolicyOutput:
        """Produce an action conditioned on the current observation + history."""
