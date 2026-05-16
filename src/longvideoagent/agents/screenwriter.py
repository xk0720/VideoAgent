"""ScreenwriterAgent — produces a GlobalStructuralPlan.

Reference: DIRECT §4.1 "Music-Driven Structure Anchoring".

v0.2 enhancement — *self-consistency sampling*. References (older + newest):
    • **Self-Consistency Improves CoT Reasoning** (Wang et al., ICLR 2023)
      Original recipe: sample K chains, majority-vote the answer.
    • **rStar** (Microsoft, 2024) / **rStar-Math** (Microsoft, Jan 2025) —
      modern descendant that uses Monte-Carlo Tree Search over reasoning
      branches with a learned verifier; we adopt the much simpler "sample K
      and take the modal structured field" recipe but the framing is the same.
    • **Best-of-N + verifier** (Cobbe et al., 2021; revived by **Tülu-3 BoN**,
      Allen AI 2024) — same idea, K samples scored by an RM.

For Screenwriter — the highest-leverage decision in the whole pipeline —
with ``self_consistency_k > 1`` we draw K plans at non-zero temperature
and pick the most stable one (modal music_section_idx → energy_level mapping).
Cost is K× the LLM calls of the original implementation, but only on this
one agent. v0.3: replace majority-vote with a verifier (Tülu-3 BoN style).

Open-source LLM access via the injected BaseLLMClient (no direct vendor SDK import).
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..types import GlobalStructuralPlan, MusicProfile, NarrativeMemory, SectionPlan
from .base import BaseAgent


class ScreenwriterAgent(BaseAgent):
    name = "screenwriter"

    def __init__(
        self,
        llm_client,
        trajectory_logger=None,
        config=None,
        self_consistency_k: int = 1,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            prompt_template="screenwriter",
            config=config or {},
            trajectory_logger=trajectory_logger,
        )
        if self_consistency_k < 1:
            raise ValueError("self_consistency_k must be >= 1")
        self.self_consistency_k = self_consistency_k

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

        # Optionally inject lessons retrieved from a persistent LessonBook
        # (Reflexion across runs; see memory/lessons.py).
        lessons = state.get("lessons_for_screenwriter") or []
        if lessons:
            lesson_block = "\nLessons from previous runs (avoid these failure modes):\n"
            lesson_block += "\n".join(f"  - {L.lesson}" for L in lessons)
            prompt = prompt + lesson_block

        plans: list[GlobalStructuralPlan] = []
        for i in range(self.self_consistency_k):
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=self.config.get("temperature", 0.4)
                            + 0.1 * i,                          # diversify samples
                max_tokens=self.config.get("max_tokens", 1500),
            )
            plans.append(self._parse(resp.text, music))

        chosen = self._aggregate_plans(plans) if self.self_consistency_k > 1 else plans[0]
        self.log_step(
            action="emit_global_plan",
            action_input={"user_prompt": user_prompt,
                          "self_consistency_k": self.self_consistency_k},
            observation={"n_sections": len(chosen.section_plans),
                         "n_samples": len(plans)},
        )
        return {"global_plan": chosen}

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

    @staticmethod
    def _aggregate_plans(plans: list[GlobalStructuralPlan]) -> GlobalStructuralPlan:
        """Self-consistency aggregation.

        For each ``music_section_idx`` present across the K plans, take the
        modal ``energy_level``; union the visual_tags; keep the first
        rationale seen. This corresponds to Wang et al.'s majority-vote
        but generalised to a structured output.
        """
        if not plans:
            return GlobalStructuralPlan()
        by_idx: dict[int, list[SectionPlan]] = {}
        for p in plans:
            for sp in p.section_plans:
                by_idx.setdefault(sp.music_section_idx, []).append(sp)
        agg: list[SectionPlan] = []
        for idx in sorted(by_idx):
            sps = by_idx[idx]
            energy = Counter(sp.energy_level for sp in sps).most_common(1)[0][0]
            tags: list[str] = []
            for sp in sps:
                for t in sp.visual_tags:
                    if t not in tags:
                        tags.append(t)
            agg.append(SectionPlan(
                music_section_idx=idx,
                energy_level=energy,                            # type: ignore[arg-type]
                visual_tags=tags,
                rationale=f"self-consistency mode of {len(sps)} sample(s)",
            ))
        return GlobalStructuralPlan(section_plans=agg)


__all__ = ["ScreenwriterAgent"]
