"""PlanValidatorAgent (a.k.a. Orchestrator) — plan-stage self-improvement.

Borrowed (cite):
  • FilmAgent — Critique-Correct-Verify multi-agent loop.
  • UniVA (arXiv:2511.08521) — self-reflective planning with traceability.
  • the old LongVideoEditAgent OrchestratorAgent — validate a plan against memory.

OUR role for it: validate each ShotSpec BEFORE expensive generation —
(1) every identity/style reference actually exists in AssetMemory (grounding),
(2) the GEST event graph is structurally valid (executable-by-construction),
then hand back per-shot feedback so the Director can correct. This adds a SECOND
self-improvement loop at the planning level, complementing the generation-level
loop (C2/C3). Catching ungroundable plans here is far cheaper than discovering
them after video synthesis.
"""
from __future__ import annotations

from ..planning.event_graph import validate_event_graph
from ..types import AssetMemory, ShotSpec
from .base import BaseAgent


class PlanValidatorAgent(BaseAgent):
    def run(
        self, specs: list[ShotSpec], asset_memory: AssetMemory
    ) -> tuple[bool, dict[int, list[str]]]:
        feedback: dict[int, list[str]] = {}
        for spec in specs:
            issues: list[str] = []
            for rid in spec.identity_refs:
                if rid not in asset_memory.identity_anchors:
                    issues.append(f"identity ref '{rid}' not in AssetMemory")
            known_styles = {s.style_id for s in asset_memory.style_anchors}
            for sid in spec.style_refs:
                if sid not in known_styles:
                    issues.append(f"style ref '{sid}' not in AssetMemory")
            if spec.event_graph is not None:
                ok, graph_issues = validate_event_graph(spec.event_graph)
                if not ok:
                    issues.extend(f"event_graph: {m}" for m in graph_issues)
            if issues:
                feedback[spec.shot_idx] = issues
        passed = not feedback
        self._log(
            "validate_plan",
            {"n_specs": len(specs)},
            {"passed": passed, "n_flagged": len(feedback)},
        )
        return passed, feedback
