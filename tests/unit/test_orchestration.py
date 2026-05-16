"""Orchestration state-machine smoke tests."""
from __future__ import annotations

from longvideoagent.orchestration.graph import build_plan_graph, run_plan_graph
from longvideoagent.orchestration.state import build_initial_state
from longvideoagent.types import GlobalStructuralPlan, NarrativeMemory


def _sw(state):
    return {"global_plan": GlobalStructuralPlan()}


def _dr(state):
    return {"segment_guidances": []}


def _or_pass(state):
    return {"validation": {"passed": True, "feedback": []}}


def _or_fail(state):
    return {"validation": {"passed": False, "feedback": [f"iter={state.get('iteration', 0)}"]}}


def test_graph_passes_first_try():
    g = build_plan_graph(_sw, _dr, _or_pass)
    state = build_initial_state(NarrativeMemory(), user_prompt="x", max_iterations=3)
    out = run_plan_graph(g, state)
    assert out["validation"]["passed"]
    assert out["iteration"] == 1


def test_graph_hits_max_iter():
    g = build_plan_graph(_sw, _dr, _or_fail)
    state = build_initial_state(NarrativeMemory(), user_prompt="x", max_iterations=3)
    out = run_plan_graph(g, state)
    assert out["iteration"] == 3
    assert not out["validation"]["passed"]
