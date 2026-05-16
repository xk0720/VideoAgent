"""Multi-agent orchestration layer.

The Stage-2 planning graph runs:
    screenwriter → director → orchestrator → (cond) → END | director

v0.1 ships a deterministic Python state machine in ``graph.py`` so the
package runs without the optional ``langgraph`` extra. The function/node
signatures and the state schema are aligned with LangGraph's StateGraph,
so flipping to a real LangGraph backend in v0.2 is a small wrapper.
"""
from .state import PlanState, build_initial_state
from .graph import build_plan_graph, run_plan_graph
from .messages import Message

__all__ = [
    "PlanState", "Message",
    "build_initial_state", "build_plan_graph", "run_plan_graph",
]
