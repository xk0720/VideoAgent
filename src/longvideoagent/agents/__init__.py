"""Multi-agent system.

Six agents now (five online + one post-hoc):
    ScreenwriterAgent  (DIRECT §4.1)
    DirectorAgent      (DIRECT §4.2)
    OrchestratorAgent  (CineAgents iterative validation)
    EditorAgent        (ReAct loop, this codebase's core contribution)
    ValidatorAgent     (zero-shot MLLM-as-judge → fine-tuned RM in v0.2)
    CriticAgent        (post-hoc meta-reviewer — Reflexion-style, writes
                        cross-run Lessons to LessonBook)
"""
from .base import BaseAgent
from .screenwriter import ScreenwriterAgent
from .director import DirectorAgent
from .orchestrator import OrchestratorAgent
from .editor import EditorAgent
from .validator import ValidatorAgent
from .critic import CriticAgent

__all__ = [
    "BaseAgent", "ScreenwriterAgent", "DirectorAgent",
    "OrchestratorAgent", "EditorAgent", "ValidatorAgent",
    "CriticAgent",
]
