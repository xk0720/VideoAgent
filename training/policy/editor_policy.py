"""EditorAgentPolicy — adapts our EditorAgent's action-picking step into the RL policy interface.

For RL training we don't need EditorAgent's full ``run()`` (it owns the whole
multi-segment loop). We only need its single-step LLM action decision —
which is exactly what ``EditorAgent._llm_pick_action`` already does.
"""
from __future__ import annotations

import json
from typing import Any

from longvideoagent.models.llm.base import BaseLLMClient

from .base import AgentPolicyBase, PolicyOutput


_DEFAULT_SYSTEM_PROMPT = (
    "You are the EditorAgent in a long-video editing pipeline. "
    "For each observation, output ONE JSON action of the form "
    "{\"action\": \"retrieve\"|\"generate\"|\"fallback\", \"rationale\": \"...\"}."
)


class EditorAgentPolicy(AgentPolicyBase):
    """Stateless wrapper: re-asks the LLM at every env.step() call."""

    def __init__(self, llm: BaseLLMClient, temperature: float = 0.3,
                 system_prompt: str = _DEFAULT_SYSTEM_PROMPT) -> None:
        self.llm = llm
        self.temperature = temperature
        self.system_prompt = system_prompt

    def reset(self) -> None:
        # Policy holds no per-episode state; ContextManager does.
        pass

    def act(self, observation, context_messages: list[dict[str, str]]) -> PolicyOutput:
        # The last item in context_messages is already the current observation
        # (the env wrapper push it before calling .act). If the caller didn't,
        # we append the observation now.
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(context_messages or [])
        msgs.append({"role": "user",
                     "content": "Current observation:\n" + json.dumps(
                         {"state": observation.state,
                          "available_actions": observation.available_actions},
                         default=str)})

        resp = self.llm.chat(msgs, temperature=self.temperature, max_tokens=300)
        action = self._parse_action(resp.text)
        return PolicyOutput(action=action, raw_text=resp.text or "")

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any]:
        try:
            data = json.loads((text or "").strip().strip("`").lstrip("json").strip())
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        action = data.get("action", "retrieve")
        return {"action": action, "rationale": data.get("rationale", "")}


__all__ = ["EditorAgentPolicy"]
