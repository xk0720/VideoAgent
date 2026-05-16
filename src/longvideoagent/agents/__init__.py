"""Multi-agent system.

Five agents, each a thin BaseAgent subclass:
    ScreenwriterAgent  (DIRECT §4.1)
    DirectorAgent      (DIRECT §4.2)
    OrchestratorAgent  (CineAgents iterative validation)
    EditorAgent        (ReAct loop, this codebase's core contribution)
    ValidatorAgent     (zero-shot MLLM-as-judge → fine-tuned RM in v0.2)
"""
from .base import BaseAgent
from .screenwriter import ScreenwriterAgent
from .director import DirectorAgent
from .orchestrator import OrchestratorAgent
from .editor import EditorAgent
from .validator import ValidatorAgent

__all__ = [
    "BaseAgent", "ScreenwriterAgent", "DirectorAgent",
    "OrchestratorAgent", "EditorAgent", "ValidatorAgent",
]
