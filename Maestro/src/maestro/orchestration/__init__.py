"""Orchestration state. v0.1 uses a light dataclass state + a linear driver in
pipeline/run.py. Swap for LangGraph in v0.2 (state persistence, conditional edges).
"""
from .state import MaestroState

__all__ = ["MaestroState"]
