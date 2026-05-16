"""JSONL trajectory logger.

Records ``AgentStep`` instances one per line so the file can be consumed
later for offline analysis or RL training (RAGEN / verl, see design doc §12).

Open-source dependencies: stdlib only.
The chosen JSONL format is interoperable with:
    • Hugging Face ``datasets.load_dataset('json', ...)``
    • OpenAI fine-tune ingestion
    • RAGEN trajectory loader (https://github.com/mll-lab-nu/RAGEN)
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from ..types import AgentStep


def _sanitize(obj: Any, redact_large: bool = True, max_array_len: int = 32) -> Any:
    """Recursively convert numpy / Path / dataclass / set into JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        if redact_large and obj.size > max_array_len:
            return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if is_dataclass(obj):
        return _sanitize(asdict(obj), redact_large, max_array_len)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, redact_large, max_array_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(x, redact_large, max_array_len) for x in obj]
    # Fallback: repr to avoid crashing the logger on exotic types.
    return repr(obj)


class TrajectoryLogger:
    """JSONL writer for AgentSteps.

    By default opens the target file in *truncate* mode (``mode="w"``) so a
    fresh ``run_pipeline`` invocation gets a clean log; pass ``mode="a"`` to
    accumulate across runs (the common case for RL training-data collection).
    """

    def __init__(
        self,
        path: Path | str,
        redact_large_tensors: bool = True,
        mode: str = "w",
    ) -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redact = redact_large_tensors
        self._fh = open(self.path, mode, encoding="utf-8")

    def append(self, step: AgentStep) -> None:
        record = _sanitize(asdict(step), redact_large=self.redact)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def log(
        self,
        agent_name: str,
        action: str,
        action_input: Optional[dict[str, Any]] = None,
        observation: Optional[dict[str, Any]] = None,
        state_snapshot: Optional[dict[str, Any]] = None,
        reward: Optional[float] = None,
        **extra: Any,
    ) -> AgentStep:
        """Convenience wrapper: build an AgentStep and append it in one call."""
        step = AgentStep(
            timestamp=time.time(),
            agent_name=agent_name,
            state_snapshot=state_snapshot or {},
            action=action,
            action_input=action_input or {},
            observation=observation or {},
            reward=reward,
            extra=extra,
        )
        self.append(step)
        return step

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@contextmanager
def open_trajectory(
    path: Path | str,
    redact_large_tensors: bool = True,
    mode: str = "w",
) -> Iterator[TrajectoryLogger]:
    logger_ = TrajectoryLogger(path, redact_large_tensors=redact_large_tensors, mode=mode)
    try:
        yield logger_
    finally:
        logger_.close()


__all__ = ["TrajectoryLogger", "open_trajectory"]
