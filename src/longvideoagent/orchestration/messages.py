"""Inter-agent message envelope.

Open-source alignment:
    The shape is borrowed from LangChain Core's ``BaseMessage`` (role +
    content + extras). When we adopt LangGraph in v0.2 these convert
    directly to langchain_core.messages.HumanMessage / AIMessage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    sender: str                                   # agent name
    receiver: str                                 # agent name or "graph"
    role: Literal["request", "response", "feedback", "tool_call"] = "request"
    content: Any = ""
    meta: dict[str, Any] = field(default_factory=dict)
