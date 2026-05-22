"""EditorEnv — RL environment wrapping EditorAgent's per-segment ReAct loop.

What's the episode?
    A *single SegmentGuidance*. The agent picks {retrieve, generate, fallback}
    until it has an accepted candidate or hits the step budget. The terminal
    reward is the final EditingSegment.metric_scores["validator"] score.

Why this granularity?
    Per ``docs/AGENTIC_RL_PROPOSAL.md`` §3.2 we adopt **Turn-PPO macro-action**
    (EACL 2026) — each segment is one macro-action. This eliminates token-level
    credit-assignment variance while still giving the policy meaningful turns.

Reward shaping (composite — see ``training/rewards/composite.py``):
    r = α * RM_score + β * mean(m1..m6) + γ * beat_alignment_bonus
    Default α=1.0, β=0.3, γ=0.1.

Action space (ORS-style tool calls):
    {"action": "retrieve"|"generate"|"fallback", "rationale": "<str>"}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from longvideoagent.tools import GenerationTool, RetrievalTool
from longvideoagent.types import (
    EditingSegment,
    NarrativeMemory,
    SegmentGuidance,
)
from longvideoagent.models.reward.base import BaseRewardModel

from .base import AgentEnvBase, EnvObservation, EnvStepResult


_ALLOWED_ACTIONS = ("retrieve", "generate", "fallback")


@dataclass
class _Episode:
    guidance: SegmentGuidance
    neighbor_context: dict[str, Any]
    candidates: list[EditingSegment]
    step: int = 0
    chosen: Optional[EditingSegment] = None


class EditorEnv(AgentEnvBase):
    """One episode = one SegmentGuidance. Action set = {retrieve, generate, fallback}."""

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        memory: NarrativeMemory,
        guidances: list[SegmentGuidance],
        retrieval_tool: RetrievalTool,
        generation_tool: GenerationTool,
        reward_model: BaseRewardModel,
        max_steps: int = 6,
        gen_output_dir: Optional[Path] = None,
        reward_alpha: float = 1.0,
        reward_beta: float = 0.3,
        reward_gamma: float = 0.1,
    ) -> None:
        self.memory = memory
        self.guidances = list(guidances)
        if not self.guidances:
            raise ValueError("EditorEnv needs at least one SegmentGuidance.")
        self.retrieval_tool = retrieval_tool
        self.generation_tool = generation_tool
        self.reward_model = reward_model
        self.max_steps = max_steps
        self.gen_output_dir = gen_output_dir or Path("./outputs/rl_generated")
        self.gen_output_dir.mkdir(parents=True, exist_ok=True)
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma

        self._cur_idx = 0
        self._ep: Optional[_Episode] = None

    # ─── gym API ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        if seed is not None:
            self._cur_idx = seed % len(self.guidances)
        guidance = self.guidances[self._cur_idx]
        beats = list(self.memory.music_profile.beats) if self.memory.music_profile else []
        neighbor_context = {
            "end_frame": None, "end_flow": None, "character_anchors": [],
            "beats": beats, "expected_start_time_s": 0.0,
        }
        self._ep = _Episode(guidance=guidance, neighbor_context=neighbor_context,
                            candidates=[], step=0)
        self._cur_idx = (self._cur_idx + 1) % len(self.guidances)
        return self._observe()

    def step(self, action: dict[str, Any]) -> EnvStepResult:
        if self._ep is None:
            raise RuntimeError("EditorEnv.step before reset()")
        act_name = (action or {}).get("action", "retrieve")
        if act_name not in _ALLOWED_ACTIONS:
            # ORS principle: unknown tool call ⇒ small negative reward + no env change.
            self._ep.step += 1
            return EnvStepResult(
                observation=self._observe(),
                reward=-0.5,
                terminated=False,
                truncated=self._ep.step >= self.max_steps,
                info={"error": f"unknown action {act_name!r}"},
            )

        cand: Optional[EditingSegment] = None
        if act_name == "retrieve":
            cand = self.retrieval_tool.run(
                guidance=self._ep.guidance,
                memory=self.memory,
                neighbor_context=self._ep.neighbor_context,
            )
        elif act_name == "generate":
            cand = self.generation_tool.run(
                guidance=self._ep.guidance,
                neighbor_context=self._ep.neighbor_context,
                output_dir=self.gen_output_dir,
            )
        elif act_name == "fallback":
            cand = self._best_so_far() or self._empty_segment()

        if cand is not None:
            self._ep.candidates.append(cand)

        step_reward = -0.05  # small per-step cost — encourages early termination
        self._ep.step += 1

        # Auto-validate the new candidate; if accepted, terminate.
        terminated = False
        if cand is not None:
            res = self.reward_model.score(cand, self._ep.guidance)
            cand.accepted_by_validator = res.accepted
            cand.validator_reasons = list(res.reasons)
            cand.metric_scores = dict(cand.metric_scores) if cand.metric_scores else {}
            cand.metric_scores["validator"] = res.score
            step_reward += self._composite_reward(cand)
            if res.accepted or act_name == "fallback":
                self._ep.chosen = cand
                terminated = True

        truncated = (not terminated) and self._ep.step >= self.max_steps
        if truncated and self._ep.chosen is None and self._ep.candidates:
            self._ep.chosen = self._best_so_far()

        return EnvStepResult(
            observation=self._observe(),
            reward=step_reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "step": self._ep.step,
                "n_candidates": len(self._ep.candidates),
                "chosen": (self._ep.chosen.segment_idx if self._ep.chosen else None),
            },
        )

    def render(self) -> Optional[str]:
        if self._ep is None:
            return "<not reset>"
        return (f"segment_idx={self._ep.guidance.segment_idx} step={self._ep.step}"
                f" candidates={len(self._ep.candidates)} chosen="
                f"{self._ep.chosen.segment_idx if self._ep.chosen else None}")

    def tools(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {
                "name": "retrieve",
                "description": "Run beam-search retrieval over the narrative memory.",
                "parameters": {"type": "object", "properties": {
                    "rationale": {"type": "string"}}}}},
            {"type": "function", "function": {
                "name": "generate",
                "description": "Synthesize a new clip via the video-gen client.",
                "parameters": {"type": "object", "properties": {
                    "rationale": {"type": "string"}}}}},
            {"type": "function", "function": {
                "name": "fallback",
                "description": "Accept best-of-candidates so far and end the segment.",
                "parameters": {"type": "object", "properties": {
                    "rationale": {"type": "string"}}}}},
        ]

    # ─── helpers ──────────────────────────────────────────────────────

    def _observe(self) -> EnvObservation:
        assert self._ep is not None
        g = self._ep.guidance
        return EnvObservation(
            state={
                "segment_idx": g.segment_idx,
                "semantic_query": g.semantic_query,
                "heuristic": g.editing_heuristic,
                "rhythmic_pacing": g.rhythmic_pacing,
                "retrieval_feasibility": g.retrieval_feasibility,
                "n_candidates": len(self._ep.candidates),
                "step": self._ep.step,
                "max_steps": self.max_steps,
            },
            available_actions=list(_ALLOWED_ACTIONS),
            info={"neighbor_has_flow": self._ep.neighbor_context.get("end_flow") is not None},
        )

    def _composite_reward(self, cand: EditingSegment) -> float:
        scores = cand.metric_scores or {}
        rm = scores.get("validator", 0.0) / 10.0           # → [0, 1]
        m_mean = (sum(scores.get(f"m{i}", 0.0) for i in range(1, 7)) / 6.0)
        beat = self._beat_alignment_bonus(cand)
        return self.reward_alpha * rm + self.reward_beta * m_mean + self.reward_gamma * beat

    def _beat_alignment_bonus(self, cand: EditingSegment) -> float:
        # Simple proxy: if any shot_trim start time is near a music beat → bonus.
        if not self.memory.music_profile or not cand.shot_trims:
            return 0.0
        beats = self.memory.music_profile.beats
        if not beats:
            return 0.0
        starts = [t[0] for t in cand.shot_trims]
        nearest = [min(abs(s - b) for b in beats) for s in starts]
        return float(max(0.0, 1.0 - sum(nearest) / len(nearest)))

    def _best_so_far(self) -> Optional[EditingSegment]:
        if not self._ep or not self._ep.candidates:
            return None
        return max(self._ep.candidates,
                   key=lambda c: sum((c.metric_scores or {}).values()))

    def _empty_segment(self) -> EditingSegment:
        assert self._ep is not None
        return EditingSegment(
            segment_idx=self._ep.guidance.segment_idx,
            source="generation", duration=2.0,
            gen_prompt=self._ep.guidance.semantic_query,
            metric_scores={"validator": 1.0}, accepted_by_validator=False,
        )


__all__ = ["EditorEnv"]
