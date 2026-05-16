"""TypedDict schema for the Stage-2 planning graph.

Mirrors LangGraph's ``StateGraph`` convention so the v0.2 swap is a wrapper.
Reference: https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from ..types import GlobalStructuralPlan, NarrativeMemory, SegmentGuidance


class PlanState(TypedDict, total=False):
    user_prompt: str
    memory: NarrativeMemory
    global_plan: Optional[GlobalStructuralPlan]
    segment_guidances: list[SegmentGuidance]
    validation: dict[str, Any]                  # {"passed": bool, "feedback": list[str]}
    iteration: int
    max_iterations: int


def build_initial_state(memory: NarrativeMemory, user_prompt: str,
                        max_iterations: int = 5) -> PlanState:
    return PlanState(
        user_prompt=user_prompt,
        memory=memory,
        global_plan=None,
        segment_guidances=[],
        validation={"passed": False, "feedback": []},
        iteration=0,
        max_iterations=max_iterations,
    )
