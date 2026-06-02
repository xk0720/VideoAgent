"""BaseTool ABC — unified function-call interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str = "tool"

    @abstractmethod
    def run(self, *args, **kwargs):
        ...
