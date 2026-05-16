"""Minimal state-machine implementation of the Stage-2 planning graph.

When ``langgraph`` is available we delegate to it; otherwise we fall back
to a plain Python loop with the same node names. Both paths share the
PlanState schema.

Open-source library considered (but optional in v0.1):
    • langgraph  https://langchain-ai.github.io/langgraph/
"""
from __future__ import annotations

from typing import Any, Callable

from ..logging import logger
from .state import PlanState


# Node = (state -> partial state update)
Node = Callable[[PlanState], dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────
# v0.1 fallback: simple state machine
# ─────────────────────────────────────────────────────────────────────


def _fallback_run(
    state: PlanState,
    screenwriter_node: Node,
    director_node: Node,
    orchestrator_node: Node,
) -> PlanState:
    state["iteration"] = 0

    # Step 1: ScreenwriterAgent once.
    state.update(screenwriter_node(state))    # type: ignore[arg-type]

    while True:
        state["iteration"] = state.get("iteration", 0) + 1
        state.update(director_node(state))                            # type: ignore[arg-type]
        state.update(orchestrator_node(state))                        # type: ignore[arg-type]

        validation = state.get("validation", {})
        if validation.get("passed"):
            logger.info(f"[plan-graph] validated at iteration={state['iteration']}")
            return state
        if state["iteration"] >= state.get("max_iterations", 5):
            logger.warning(f"[plan-graph] hit max_iterations={state['max_iterations']}; "
                           f"accepting plan with {len(validation.get('feedback', []))} feedback items.")
            return state
        logger.info(f"[plan-graph] re-planning, feedback={validation.get('feedback')}")


# ─────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────


def build_plan_graph(
    screenwriter_node: Node,
    director_node: Node,
    orchestrator_node: Node,
    *,
    use_langgraph: bool = False,
):
    """Return a callable graph that takes a PlanState and produces a PlanState."""
    if not use_langgraph:
        def _run(state: PlanState) -> PlanState:
            return _fallback_run(state, screenwriter_node, director_node, orchestrator_node)
        return _run

    try:                                                             # pragma: no cover
        from langgraph.graph import StateGraph, END
    except ImportError:                                              # pragma: no cover
        logger.warning("langgraph not installed; falling back to Python state machine.")
        def _run(state: PlanState) -> PlanState:
            return _fallback_run(state, screenwriter_node, director_node, orchestrator_node)
        return _run

    sg = StateGraph(PlanState)                                       # pragma: no cover
    sg.add_node("screenwriter", screenwriter_node)
    sg.add_node("director", director_node)
    sg.add_node("orchestrator", orchestrator_node)
    sg.set_entry_point("screenwriter")
    sg.add_edge("screenwriter", "director")
    sg.add_edge("director", "orchestrator")

    def _should_continue(state: PlanState):                          # pragma: no cover
        val = state.get("validation", {})
        if val.get("passed"):
            return END
        if state.get("iteration", 0) >= state.get("max_iterations", 5):
            return END
        return "director"

    sg.add_conditional_edges("orchestrator", _should_continue)
    return sg.compile()


def run_plan_graph(graph, initial_state: PlanState) -> PlanState:
    return graph(initial_state)                                      # works for both backends
