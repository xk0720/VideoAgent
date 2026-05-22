"""Policy interface for RL rollouts.

Mirrors OpenRLHF's ``AgentInstanceBase`` + ``AgentExecutorBase`` pattern
(https://openrlhf.readthedocs.io/en/latest/async_rl.html):
    • AgentPolicyBase    — analog of AgentInstanceBase (per-episode state)
    • EditorAgentPolicy  — wraps our EditorAgent into the same interface
"""
from .base import AgentPolicyBase, PolicyOutput
from .editor_policy import EditorAgentPolicy

__all__ = ["AgentPolicyBase", "PolicyOutput", "EditorAgentPolicy"]
