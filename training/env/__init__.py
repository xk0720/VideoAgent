"""RL environment wrappers around agents.

Reference frameworks:
    • RAGEN Environment Manager — interactive multi-turn env abstraction
    • verl 0.7 AgentLoop        — async tool-based rollouts
    • ORS Open Reward Standard  — HTTP-tool protocol for env↔agent
"""
from .base import AgentEnvBase, EnvObservation, EnvStepResult
from .editor_env import EditorEnv
from .context_manager import ContextManager

__all__ = ["AgentEnvBase", "EnvObservation", "EnvStepResult",
           "EditorEnv", "ContextManager"]
