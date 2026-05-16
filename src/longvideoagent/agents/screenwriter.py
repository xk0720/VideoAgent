"""ScreenwriterAgent — produces a GlobalStructuralPlan.

Reference: DIRECT §4.1 "Music-Driven Structure Anchoring".
Open-source LLM access via the injected BaseLLMClient (no direct vendor SDK import).
"""
from __future__ import annotations

import json
from typing import Any

from ..types import GlobalStructuralPlan, MusicProfile, NarrativeMemory, SectionPlan
from .base import BaseAgent


class ScreenwriterAgent(BaseAgent):
    name = "screenwriter"

    def __init__(self, llm_client, trajectory_logger=None, config=None) -> None:
        super().__init__(
            llm_client=llm_client,
            prompt_template="screenwriter",
            config=config or {},
            trajectory_logger=trajectory_logger,
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        memory: NarrativeMemory = state["memory"]
        user_prompt: str = state["user_prompt"]
        music: MusicProfile | None = memory.music_profile
        bpm = music.bpm if music else 120.0
        music_sections_text = self._fmt_sections(music)

        prompt = self.render_prompt(
            user_prompt=user_prompt,
            memory_summary=memory.summarize(),
            bpm=bpm,
            music_sections=music_sections_text,
        )
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.config.get("temperature", 0.4),
            max_tokens=self.config.get("max_tokens", 1500),
        )
        plan = self._parse(resp.text, music)
        self.log_step(
            action="emit_global_plan",
            action_input={"user_prompt": user_prompt},
            observation={"n_sections": len(plan.section_plans)},
        )
        return {"global_plan": plan}

    # ── helpers ──

    @staticmethod
    def _fmt_sections(music: MusicProfile | None) -> str:
        if music is None or not music.sections:
            return "  (no music; treat full duration as a single 'instrumental' section)"
        rows = []
        for i, s in enumerate(music.sections):
            rows.append(f"  {i:>2} | {s.name:<12} | {s.start_time:>5.1f}..{s.end_time:<5.1f}s "
                        f"| energy={s.energy_db:+.1f} dB")
        return "\n".join(rows)

    @staticmethod
    def _parse(text: str, music: MusicProfile | None) -> GlobalStructuralPlan:
        try:
            data = json.loads(text)
            sps = [SectionPlan(**sp) for sp in data["section_plans"]]
            return GlobalStructuralPlan(section_plans=sps)
        except Exception:
            # Fallback: assume one section per music section, medium energy.
            n = len(music.sections) if music else 1
            return GlobalStructuralPlan(section_plans=[
                SectionPlan(music_section_idx=i, energy_level="medium",
                            visual_tags=["fallback"], rationale="parser-fallback")
                for i in range(n)
            ])


__all__ = ["ScreenwriterAgent"]
