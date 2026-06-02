"""Trajectory logger (E4): every agent decision -> JSONL.

Even though v0.1 is training-free, this keeps a clean (state, action, reward)
interface so a future reward model / agentic-RL stage can consume it directly.
"""
from __future__ import annotations

import dataclasses
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .types import AgentStep


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # numpy arrays / unknown -> shape or repr (kept small)
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return f"<ndarray shape={tuple(shape)}>"
    return repr(obj)[:200]


class TrajectoryLogger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.steps: list[AgentStep] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # truncate
            self.path.write_text("", encoding="utf-8")

    def append(
        self,
        agent_name: str,
        action: str,
        action_input: dict,
        observation: dict,
        state_snapshot: Optional[dict] = None,
        reward: Optional[float] = None,
    ) -> AgentStep:
        step = AgentStep(
            timestamp=time.time(),
            agent_name=agent_name,
            state_snapshot=state_snapshot or {},
            action=action,
            action_input=action_input,
            observation=observation,
            reward=reward,
        )
        self.steps.append(step)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_to_jsonable(step), ensure_ascii=False) + "\n")
        return step

    def actions(self) -> list[str]:
        return [s.action for s in self.steps]
