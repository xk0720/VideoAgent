"""Stage 2 — multi-agent planning.

Drives screenwriter → director → orchestrator via the orchestration graph
and returns a list[SegmentGuidance].

v0.2 wiring:
    • Optional ``LessonBook`` retrieval pre-pends Reflexion-style cross-run
      lessons into the Screenwriter state.
    • ScreenwriterAgent runs at self_consistency_k = config.plan.self_consistency_k
      (default 1; keeps v0.1 behaviour unchanged).
"""
from __future__ import annotations

from typing import Optional

from ..agents import DirectorAgent, OrchestratorAgent, ScreenwriterAgent
from ..config import Config
from ..memory.lessons import LessonBook
from ..memory.retriever import MemoryRetriever
from ..memory.store import MemoryStore
from ..models.llm import build_llm_from_alias
from ..orchestration.graph import build_plan_graph, run_plan_graph
from ..orchestration.state import build_initial_state
from ..types import NarrativeMemory, SegmentGuidance
from ..utils.trajectory import TrajectoryLogger


def plan(
    memory: NarrativeMemory,
    user_prompt: str,
    config: Config,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    memory_store: Optional[MemoryStore] = None,
    lesson_book: Optional[LessonBook] = None,
    self_consistency_k: int = 1,
) -> list[SegmentGuidance]:
    mocks_llm = config.mocks.llm
    sw_llm = build_llm_from_alias("screenwriter", mocks_enabled=mocks_llm)
    dr_llm = build_llm_from_alias("director", mocks_enabled=mocks_llm)
    or_llm = build_llm_from_alias("orchestrator", mocks_enabled=mocks_llm)

    if memory_store is None:
        memory_store = MemoryStore(config.cache_root)
    retriever = MemoryRetriever(memory_store, embed_dim=config.preprocess.feature_extractor.embed_dim)

    screenwriter = ScreenwriterAgent(sw_llm, trajectory_logger=trajectory_logger,
                                     self_consistency_k=self_consistency_k)
    director = DirectorAgent(dr_llm, retriever, trajectory_logger=trajectory_logger,
                             config={"temperature": 0.3})
    orchestrator = OrchestratorAgent(or_llm, retriever, trajectory_logger=trajectory_logger)

    graph = build_plan_graph(
        screenwriter_node=screenwriter.run,
        director_node=director.run,
        orchestrator_node=orchestrator.run,
        use_langgraph=config.plan.use_langgraph,
    )
    state = build_initial_state(memory, user_prompt, max_iterations=config.plan.max_iterations)

    # Inject cross-run lessons (Reflexion-style) for the Screenwriter.
    if lesson_book is not None:
        keywords = [t for t in user_prompt.split() if len(t) > 3]
        relevant = lesson_book.retrieve_relevant("screenwriter", keywords=keywords, limit=5)
        if relevant:
            state["lessons_for_screenwriter"] = relevant

    final = run_plan_graph(graph, state)
    return final.get("segment_guidances", [])


__all__ = ["plan"]
