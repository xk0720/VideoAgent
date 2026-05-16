"""DirectorAgent — 3 (+1) step CoT per section.

References:
    DIRECT §4.2 (semantic query, heuristic, pacing)
    + retrieval-feasibility estimator (extra step in this codebase).
"""
from __future__ import annotations

import json
from typing import Any

from ..config import configs_dir, load_yaml
from ..types import GlobalStructuralPlan, MusicProfile, SegmentGuidance
from .base import BaseAgent


class DirectorAgent(BaseAgent):
    name = "director"

    def __init__(self, llm_client, retriever, trajectory_logger=None, config=None) -> None:
        super().__init__(
            llm_client=llm_client,
            prompt_template="director_query",   # we use 3 prompts but render them separately
            config=config or {},
            trajectory_logger=trajectory_logger,
        )
        # Pre-load the two extra prompt files (still no hardcoded prompts).
        from ..config import load_prompt
        self.prompt_heuristic = load_prompt("director_heuristic")
        self.prompt_pacing = load_prompt("director_pacing")
        self.retriever = retriever
        self.presets = load_yaml(configs_dir() / "heuristics" / "presets.yaml")["presets"]

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        plan: GlobalStructuralPlan = state["global_plan"]
        music: MusicProfile | None = state["memory"].music_profile
        guidances: list[SegmentGuidance] = []

        # If Orchestrator left feedback on the last iteration, use it as
        # bootstrap history so the new queries diverge from the rejected ones.
        prev_validation = state.get("validation", {}) or {}
        feedback: list[str] = list(prev_validation.get("feedback", []))
        prev_guidances: list[SegmentGuidance] = list(state.get("segment_guidances", []))
        history: list[str] = [g.semantic_query for g in prev_guidances]

        seg_idx = 0
        for section_plan in plan.section_plans:
            music_section = (music.sections[section_plan.music_section_idx]
                             if music and section_plan.music_section_idx < len(music.sections)
                             else None)
            # Step 1 — semantic query (with optional feedback context)
            q_text = self._call_query(section_plan, history, feedback)
            history.append(q_text)
            # Step 2 — heuristic
            heuristic = self._call_heuristic(q_text, section_plan.energy_level)
            # Step 3 — rhythmic pacing
            pacing = self._call_pacing(music_section, section_plan.energy_level)
            # Step 4 (extra) — retrieval feasibility
            feasibility = self.retriever.estimate_feasibility(q_text, top_k=20)

            guidances.append(SegmentGuidance(
                segment_idx=seg_idx,
                parent_section_idx=section_plan.music_section_idx,
                semantic_query=q_text,
                editing_heuristic=heuristic,
                rhythmic_pacing=pacing,
                cinematography_hints=section_plan.visual_tags,
                retrieval_feasibility=feasibility,
            ))
            seg_idx += 1

        self.log_step(
            action="emit_segment_guidances",
            observation={"n_segments": len(guidances),
                         "feedback_consumed": len(feedback)},
        )
        return {"segment_guidances": guidances}

    # ── individual LLM calls ──

    def _call_query(self, section_plan, history, feedback: list[str] | None = None) -> str:
        hist_lines = [f"  - {h}" for h in history[-5:]] or ["  (none)"]
        if feedback:
            hist_lines.append("")
            hist_lines.append("Validator feedback to address on this re-plan:")
            hist_lines.extend(f"  ! {f}" for f in feedback[:5])
        prompt = self.prompt_template.format(
            section_plan=json.dumps(section_plan.__dict__, default=str),
            history="\n".join(hist_lines),
        )
        resp = self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=self.config.get("temperature", 0.3),
                             max_tokens=400)
        try:
            return json.loads(resp.text)["semantic_query"]
        except Exception:
            return f"cinematic shot ({section_plan.energy_level})"

    def _call_heuristic(self, semantic_query: str, energy_level: str) -> str:
        descs = "\n".join(f"  - {name}" for name in self.presets.keys())
        prompt = self.prompt_heuristic.format(
            preset_descriptions=descs,
            semantic_query=semantic_query,
            energy_level=energy_level,
        )
        resp = self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=self.config.get("temperature", 0.3),
                             max_tokens=200)
        try:
            picked = json.loads(resp.text)["editing_heuristic"]
            return picked if picked in self.presets else "default"
        except Exception:
            return "default"

    def _call_pacing(self, music_section, energy_level: str) -> list[int]:
        if music_section is None:
            return [4, 4, 4, 4]
        prompt = self.prompt_pacing.format(
            section_start=music_section.start_time,
            section_end=music_section.end_time,
            section_beats=music_section.num_beats,
            energy_level=energy_level,
        )
        resp = self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=self.config.get("temperature", 0.3),
                             max_tokens=200)
        try:
            pacing = json.loads(resp.text)["rhythmic_pacing"]
            return [int(x) for x in pacing if int(x) >= 1]
        except Exception:
            # Fallback: divide into 4 equal slots.
            n = max(1, music_section.num_beats // 4)
            return [4] * n


__all__ = ["DirectorAgent"]
