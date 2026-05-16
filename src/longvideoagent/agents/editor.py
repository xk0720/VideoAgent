"""EditorAgent — multi-step ReAct loop.

References (older → 2024 → 2025/2026):
    • **ReAct** (Yao et al., ICLR 2023) — interleaved reasoning/acting steps,
      the foundational recipe we follow.
    • **GLANCE: Music-Grounded Non-Linear Video Editing** (arXiv 2604.05076,
      2026) — bi-loop ReAct + verify for music-driven editing; our
      auto-validate-after-each-action pattern is a simplification of GLANCE's
      inner loop (see SYSTEM_GUIDE §5.1.4). GLANCE reports +33.2% / +15.6% over
      the strongest baseline on two task settings.
    • **FilmAgent** (HITsz-TMG, 2024) — Critique-Correct-Verify multi-agent;
      we keep CCV at the *plan* layer (Orchestrator + Director) rather than
      inside Editor so the segment-level loop stays cheap.
    • **Sima 1.0** (arXiv 2604.07721, 2026) — 11-step documentary production
      pipeline; informs our stage decomposition.
    • **LongVideoAgent: Multi-Agent Reasoning with Long Videos** (arXiv
      2512.20618, 2025) — same-name project that does long-video VQA via a
      master+grounding+vision multi-agent loop trained with RL. Our project
      does **editing** (Stage 3 composes a new mp4), but the master-agent
      pattern and RL-trained step-limit budget directly inspire v0.4.
    • LangChain Core ``langchain_core.tools`` — tool-calling surface our action
      menu mirrors; not depended on in v0.1 to keep imports light.

This is the only agent that drives tool use directly. Per segment it chooses
among three actions ({retrieve, generate, fallback}) until it has an
accepted candidate OR the step budget is exhausted; auto-validate runs
after every retrieve/generate to keep the trajectory log clean.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from ..types import (
    EditingScript,
    EditingSegment,
    NarrativeMemory,
    SegmentGuidance,
)
from .base import BaseAgent


class EditorAgent(BaseAgent):
    name = "editor"

    def __init__(
        self,
        llm_client,
        retrieval_tool,
        generation_tool,
        validator,
        trajectory_logger=None,
        config: Optional[dict[str, Any]] = None,
        max_steps: int = 10,
        feasibility_threshold: float = 0.4,
        preference_logger=None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            prompt_template="editor_summary",
            config=config or {},
            trajectory_logger=trajectory_logger,
        )
        self.retrieval_tool = retrieval_tool
        self.generation_tool = generation_tool
        self.validator = validator
        self.max_steps = max_steps
        self.feasibility_threshold = feasibility_threshold
        # Optional: when set, every multi-candidate segment writes a
        # (winner, losers) DPO/IPO-ready preference pair.
        self.preference_logger = preference_logger

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        guidances: list[SegmentGuidance] = state["segment_guidances"]
        memory: NarrativeMemory = state["memory"]
        output_dir: Path = state.get("output_dir", Path("./outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        script = EditingScript(music_path=memory.music_profile.audio_path if memory.music_profile else None)
        beats = list(memory.music_profile.beats) if memory.music_profile else []
        neighbor_context: dict[str, Any] = {
            "end_frame": None, "end_flow": None, "character_anchors": [],
            "beats": beats, "expected_start_time_s": 0.0,
        }

        cumulative_t = 0.0
        for guidance in guidances:
            neighbor_context["expected_start_time_s"] = cumulative_t
            seg = self._compose_one(guidance, memory, neighbor_context, output_dir)
            script.segments.append(seg)
            cumulative_t += seg.duration
            neighbor_context = self._derive_neighbor_context(seg, memory)
            neighbor_context["beats"] = beats
            neighbor_context["expected_start_time_s"] = cumulative_t
            script.total_duration += seg.duration

        return {"script": script}

    # ── per-segment composition ──

    def _compose_one(
        self,
        guidance: SegmentGuidance,
        memory: NarrativeMemory,
        neighbor_context: dict[str, Any],
        output_dir: Path,
    ) -> EditingSegment:
        candidates: list[EditingSegment] = []
        chosen: Optional[EditingSegment] = None
        step = 0

        # Default-routing: feasibility-aware first action.
        first_action = "generate" if guidance.retrieval_feasibility < self.feasibility_threshold \
                                   else "retrieve"

        while step < self.max_steps and chosen is None:
            step += 1
            action = first_action if step == 1 else self._llm_pick_action(
                guidance, candidates, neighbor_context, step)

            if action == "retrieve":
                cand = self.retrieval_tool.run(
                    guidance=guidance, memory=memory,
                    neighbor_context=neighbor_context,
                )
                if cand:
                    candidates.append(cand)
            elif action == "generate":
                cand = self.generation_tool.run(
                    guidance=guidance,
                    neighbor_context=neighbor_context,
                    output_dir=output_dir,
                )
                if cand:
                    candidates.append(cand)
            elif action == "fallback":
                chosen = max(candidates, key=lambda c: sum(c.metric_scores.values())) \
                    if candidates else self._empty_segment(guidance)
                break
            else:
                # Unknown action (e.g. legacy "validate" from a stale prompt) —
                # just log and skip; the auto-validate below still runs for the
                # last produced candidate.
                pass

            self.log_step(action=action,
                          action_input={"segment_idx": guidance.segment_idx, "step": step},
                          observation={"n_candidates": len(candidates)})

            # Auto-validate the most recent candidate, regardless of the
            # chosen action. This is the single point where reward-model
            # calls happen for a segment, so the trajectory log stays clean
            # (at most one judge entry per produced candidate).
            if action in ("retrieve", "generate"):
                last = candidates[-1] if candidates else None
                if last is not None:
                    res = self.validator.score(last, guidance)
                    last.accepted_by_validator = res.accepted
                    last.validator_reasons = res.reasons
                    last.metric_scores["validator"] = res.score
                    if res.accepted:
                        chosen = last

        if chosen is None:
            chosen = candidates[-1] if candidates else self._empty_segment(guidance)

        # Inherit segment_idx for downstream assembly.
        chosen.segment_idx = guidance.segment_idx

        # Emit a (winner, losers) preference pair if we had > 1 candidate.
        # This is free DPO/IPO training data for the v0.3 RM training step
        # (see utils/preferences.py docstring + Rafailov et al., NeurIPS 2023).
        if self.preference_logger is not None and len(candidates) > 1:
            losers = [c for c in candidates if c is not chosen]
            if losers:
                self.preference_logger.log_pair(
                    guidance=guidance, winner=chosen, losers=losers,
                    judge_name=type(self.validator.reward_model).__name__,
                )

        # One summary line per segment in the trajectory log — captures the
        # actual metric_scores + accept/reject decision so RL training data is
        # complete without having to scrape per-step entries.
        self.log_step(
            action="segment_finalized",
            action_input={
                "segment_idx": guidance.segment_idx,
                "semantic_query": guidance.semantic_query,
                "heuristic": guidance.editing_heuristic,
            },
            observation={
                "source": chosen.source,
                "accepted": chosen.accepted_by_validator,
                "metric_scores": dict(chosen.metric_scores),
                "n_shot_ids": len(chosen.shot_ids),
                "duration": chosen.duration,
            },
            reward=chosen.metric_scores.get("validator"),
        )
        return chosen

    # ── helpers ──

    def _llm_pick_action(self, guidance, candidates, neighbor_context, step) -> str:
        prompt = self.render_prompt(
            max_steps=self.max_steps,
            guidance=json.dumps(asdict(guidance), default=str),
            candidates=json.dumps([asdict(c) for c in candidates], default=str)[:2000],
            neighbor_context=json.dumps(
                {k: (v if not hasattr(v, "shape") else f"<ndarray {v.shape}>")
                 for k, v in neighbor_context.items()}, default=str)[:500],
            feasibility_threshold=self.feasibility_threshold,
        )
        resp = self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=self.config.get("temperature", 0.2),
                             max_tokens=200)
        try:
            return json.loads(resp.text)["action"]
        except Exception:
            return "validate"

    def _empty_segment(self, guidance: SegmentGuidance) -> EditingSegment:
        # Last-ditch: emit an empty segment so the timeline isn't broken.
        return EditingSegment(segment_idx=guidance.segment_idx, source="generation",
                              duration=2.0, gen_prompt=guidance.semantic_query,
                              metric_scores={"validator": 1.0}, accepted_by_validator=False)

    def _derive_neighbor_context(self, seg: EditingSegment, memory: NarrativeMemory) -> dict[str, Any]:
        # v0.1: just return whatever ShotFeatures the last retrieved shot exposes.
        if seg.source == "retrieval" and seg.shot_ids:
            shot = memory.shots.get(seg.shot_ids[-1])
            if shot and shot.features is not None:
                return {
                    "end_frame": None,                        # v0.2 will store an actual ndarray
                    "end_flow": shot.features.end_flow,
                    "character_anchors": shot.character_ids,
                }
        return {"end_frame": None, "end_flow": None, "character_anchors": []}


__all__ = ["EditorAgent"]
