"""ActAgent — execute tool-call plans (UniVA Plan-Act dual-agent, Act side).

Borrowed pattern (cite): **UniVA — Universal Video Agent** (arXiv:2511.08521).
UniVA's two-agent split:
    Plan Agent  → high-level reasoning, decomposes a user request into structured
                  tool-call steps.
    Act Agent   → low-level execution, invokes the right MCP tool for each step
                  and feeds observations back.

Maestro's existing agents (Screenwriter / Director / PhysicsPlanner / Refiner)
are domain-specialized PLAN agents — they already decompose the task. What we
were missing is a generic *executor*: something that takes the abstract step
("probe this video", "caption this image", "run the metric tool on this clip")
and invokes the right tool from `tools.default_registry()`, logging the call so
the trajectory shows the entire Plan→Act chain end-to-end.

WHY THIS ISN'T A REIMPLEMENTATION OF Refiner: Refiner translates *physics
verdicts* into a *self-improvement repair plan* (image-edit + extra prompt) —
that is C2-specific orchestration logic. ActAgent is general purpose: given any
declarative `ToolCall(name, args)`, route it through the registry and capture
the observation. The two compose — a Refiner output can be re-expressed as a
list of ToolCalls and shipped through ActAgent for unified logging.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..tools.base import ToolRegistry, default_registry
from .base import BaseAgent


def _sandbox_enabled() -> bool:
    """Honor the `MAESTRO_SANDBOX` env var (documented in `.env.example`).

    Sandbox mode: refuse to invoke tools whose `spec.side_effects=True` — useful
    for untrusted plan executions, CI smoke runs, demo boxes. Off by default
    so production pipelines (which legitimately need ffmpeg / file writes)
    are not crippled. Read at call time so an operator can toggle without
    restart.
    """
    val = os.getenv("MAESTRO_SANDBOX", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


@dataclass
class ToolCall:
    """Declarative description of a tool invocation. JSON-serializable so a Plan
    agent (or an LLM Planner in v0.3) can emit it as part of a structured plan.
    """

    name: str                                  # registry tool name
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    note: str = ""                             # free-form human/agent annotation


@dataclass
class ToolResult:
    name: str
    ok: bool
    value: Any = None
    error: str = ""


class ActAgent(BaseAgent):
    """Executes a sequence of `ToolCall`s against a `ToolRegistry`.

    Logs each call as a `tool_call` action in the trajectory so the JSONL log
    captures the full Plan→Act chain (was previously implicit and per-agent).
    """

    def __init__(self, *args, registry: Optional[ToolRegistry] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = registry or default_registry()

    def call(self, tc: ToolCall) -> ToolResult:
        try:
            tool = self.registry.get(tc.name)
        except KeyError as e:
            self._log("tool_call",
                      {"name": tc.name, "args": tc.args, "kwargs": tc.kwargs},
                      {"ok": False, "error": str(e)})
            return ToolResult(name=tc.name, ok=False, error=str(e))
        # Sandbox gate (cf. .env.example MAESTRO_SANDBOX). A side-effecting
        # tool (writes files, calls a paid API) is refused with a clear error
        # so the caller can downgrade or abort — never silently dropped.
        if _sandbox_enabled() and getattr(tool.spec, "side_effects", False):
            err = (f"sandbox: refusing side-effecting tool '{tc.name}' "
                   f"(MAESTRO_SANDBOX=1)")
            self._log("tool_call",
                      {"name": tc.name, "args": tc.args, "kwargs": tc.kwargs},
                      {"ok": False, "error": err, "sandbox": True})
            return ToolResult(name=tc.name, ok=False, error=err)
        try:
            value = tool.run(*tc.args, **tc.kwargs)
            obs = {"ok": True, "category": tool.category}
            # Trim huge return values (lists, paths) for the trajectory; keep
            # type info so a reader knows what was produced.
            obs["value_type"] = type(value).__name__
            if isinstance(value, (str, int, float, bool)):
                obs["value"] = value
            elif isinstance(value, list):
                obs["value_len"] = len(value)
            self._log("tool_call",
                      {"name": tc.name, "args": tc.args, "kwargs": tc.kwargs,
                       "note": tc.note},
                      obs)
            return ToolResult(name=tc.name, ok=True, value=value)
        except Exception as e:
            self._log("tool_call",
                      {"name": tc.name, "args": tc.args, "kwargs": tc.kwargs},
                      {"ok": False, "error": repr(e)})
            return ToolResult(name=tc.name, ok=False, error=repr(e))

    def run(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute a plan (list of ToolCalls), returning all results in order.

        Plans are NOT atomic — a failure on step k does not halt step k+1. This
        matches UniVA's executor which logs failed tool calls but keeps going so
        the next plan iteration can react to them. For atomic semantics callers
        should inspect `r.ok` and short-circuit themselves.
        """
        return [self.call(c) for c in calls]
