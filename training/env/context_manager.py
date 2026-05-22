"""ContextManager — keeps the rollout's running observation history bounded.

RAGEN's "Context Manager" component (see RAGEN paper §3.2) maintains the
text history fed to the LLM at each turn. We adopt the same name and the
same job, but our observations are structured dicts (not raw text) so we
serialize on the fly.

Reference: https://github.com/RAGEN-AI/RAGEN — README §"Architecture"
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

from .base import EnvObservation


@dataclass
class ContextManager:
    """Bounded ring-buffer of (action, observation) turns.

    The LLM agent sees the last ``max_turns`` turns when deciding its next
    action — same pattern as CineAgents rolling caption buffer and ReAct.
    """
    max_turns: int = 8
    history: Deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self):
        # Force the deque to honour max_turns even if it was set by hand.
        self.history = deque(self.history, maxlen=self.max_turns)

    def reset(self) -> None:
        self.history.clear()

    def push(self, action: dict[str, Any] | None, observation: EnvObservation,
             reward: float | None = None) -> None:
        self.history.append({
            "action": action,
            "observation": {
                "state": observation.state,
                "available_actions": observation.available_actions,
            },
            "reward": reward,
        })

    def to_prompt(self, system_prefix: str = "") -> str:
        """Render the history as a prompt string the LLM can read.

        Layout mirrors ReAct: Thought / Action / Observation per turn.
        """
        lines: list[str] = [system_prefix.strip()] if system_prefix else []
        for i, turn in enumerate(self.history):
            lines.append(f"--- turn {i} ---")
            if turn["action"] is not None:
                lines.append(f"Action: {json.dumps(turn['action'], default=str)}")
            lines.append(f"Observation: {json.dumps(turn['observation'], default=str)[:600]}")
            if turn["reward"] is not None:
                lines.append(f"Reward: {turn['reward']:.3f}")
        return "\n".join(lines)

    def to_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        """Render the history as OpenAI-format messages (more useful than to_prompt
        when the policy uses the chat API)."""
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for turn in self.history:
            if turn["action"] is not None:
                msgs.append({"role": "assistant",
                             "content": json.dumps(turn["action"], default=str)})
            msgs.append({"role": "user",
                         "content": "Observation: " + json.dumps(
                             turn["observation"], default=str)[:600]})
        return msgs

    def __len__(self) -> int:
        return len(self.history)


__all__ = ["ContextManager"]
