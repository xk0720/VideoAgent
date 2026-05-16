"""BaseTool ABC.

All tools are callable via ``.run(...)`` and self-describe their function-
call schema via ``.schema()`` so EditorAgent can expose them to a real LLM
in v0.2 (OpenAI function calling / Anthropic tools).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any: ...

    def schema(self) -> dict[str, Any]:
        """Return an OpenAI-style function spec (overridable per tool)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
