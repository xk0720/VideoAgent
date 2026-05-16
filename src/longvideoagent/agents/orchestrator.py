"""OrchestratorAgent — iterative narrative validator.

Reference: CineAgents "iterative narrative planning".
Emits (passed, feedback). Stage-2 graph re-invokes DirectorAgent with
feedback until passed == True or max_iterations is hit.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..types import GlobalStructuralPlan, NarrativeMemory, SegmentGuidance
from .base import BaseAgent


_DEFAULT_CHECKS = [
    "grounding_in_memory",
    "section_coverage",
    "pacing_sanity",
    "no_duplicate_queries",
]


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"

    def __init__(self, llm_client, retriever, trajectory_logger=None, config=None) -> None:
        super().__init__(
            llm_client=llm_client,
            prompt_template="orchestrator_validate",
            config=config or {},
            trajectory_logger=trajectory_logger,
        )
        self.retriever = retriever
        self.checks = list(config.get("checks", _DEFAULT_CHECKS)) if config else list(_DEFAULT_CHECKS)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        guidances: list[SegmentGuidance] = state["segment_guidances"]
        plan: GlobalStructuralPlan = state["global_plan"]
        memory: NarrativeMemory = state["memory"]

        # Cheap deterministic pre-checks; they catch the obvious failures even
        # when the LLM is mocked. The LLM call is still made (and its output
        # respected) so we don't lose the validation surface in v0.2.
        feedback = self._heuristic_checks(guidances, plan, memory)

        prompt = self.render_prompt(
            global_plan=json.dumps([asdict(s) for s in plan.section_plans], default=str)[:4000],
            segment_guidances=json.dumps([asdict(g) for g in guidances], default=str)[:4000],
            memory_summary=memory.summarize(),
            checks="\n".join(f"  - {c}" for c in self.checks),
        )
        resp = self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=self.config.get("temperature", 0.1),
                             max_tokens=1200)
        try:
            data = json.loads(resp.text)
            llm_passed = bool(data.get("passed", True))
            llm_feedback = list(data.get("feedback", []))
        except Exception:
            llm_passed, llm_feedback = True, []

        feedback.extend(llm_feedback)
        passed = llm_passed and not feedback

        self.log_step(
            action="validate",
            observation={"passed": passed, "n_feedback": len(feedback)},
            reward=1.0 if passed else 0.0,
        )
        return {"validation": {"passed": passed, "feedback": feedback}}

    def _heuristic_checks(self, guidances, plan, memory) -> list[str]:
        notes: list[str] = []
        # 1) no_duplicate_queries
        seen = set()
        for g in guidances:
            if g.semantic_query in seen:
                notes.append(f"Segment {g.segment_idx}: duplicate semantic_query "
                             f"'{g.semantic_query}' — consider rewording.")
            seen.add(g.semantic_query)
        # 2) grounding_in_memory
        for g in guidances:
            if g.retrieval_feasibility < 0.05 and len(memory.shots) > 0:
                notes.append(f"Segment {g.segment_idx}: query '{g.semantic_query}' "
                             f"has near-zero retrieval feasibility ({g.retrieval_feasibility:.2f}).")
        # 3) director->screenwriter coverage: every plan section has a guidance
        covered = {g.parent_section_idx for g in guidances}
        expected_from_plan = {sp.music_section_idx for sp in plan.section_plans}
        missing = expected_from_plan - covered
        if missing:
            notes.append(f"Sections in the global plan not covered by any guidance: {sorted(missing)}.")
        # 4) screenwriter->music coverage: every music section appears in the plan
        if memory.music_profile is not None:
            music_idx = set(range(len(memory.music_profile.sections)))
            uncovered = music_idx - expected_from_plan
            if uncovered:
                notes.append(f"ScreenwriterAgent dropped music sections {sorted(uncovered)}; "
                             f"plan should cover every section in the music profile.")
        return notes


__all__ = ["OrchestratorAgent"]
