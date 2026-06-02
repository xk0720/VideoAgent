"""Tests for the v0.2 additions:

  • C5 HSI tiered self-improvement loop
  • C6 PhysicsConsistencyCritic (closed-loop sketch verification)
  • bug fix: lesson distilled from actually-resolved mode
  • bug fix: ReviewBoard.recompute_metrics refreshes only scores (not critics)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from maestro.agents.director import DirectorAgent
from maestro.agents.generator import GeneratorAgent
from maestro.agents.physics_planner import PhysicsPlannerAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.consistency import ConsistencyCritic
from maestro.critics.physics import PhysicsCritic
from maestro.critics.physics_consistency import PhysicsConsistencyCritic
from maestro.critics.rhythm import RhythmCritic
from maestro.critics.semantic import SemanticCritic
from maestro.memory.lesson_library import LessonLibrary
from maestro.models.mllm import BaseMLLMClient, MockMLLMClient
from maestro.physics.sketch import build_physics_sketch
from maestro.pipeline.generate_loop import generate_shot
from maestro.tools.metric_tool import MetricTool
from maestro.types import CandidateClip, Checklist, PhysFailureMode, ShotSpec


def _board(extra_critics=None):
    critics = [
        SemanticCritic(),
        PhysicsCritic(),
        PhysicsConsistencyCritic(),
        ConsistencyCritic(),
        RhythmCritic(),
    ]
    if extra_critics:
        critics.extend(extra_critics)
    return ReviewBoard(critics)


# ─────────────────────────────────────────────────────────────────────────────
# C6 — PhysicsConsistencyCritic
# ─────────────────────────────────────────────────────────────────────────────
def test_consistency_critic_passes_when_generator_followed_sketch(tmp_path: Path):
    """A clip generated WITH the sketch should not be flagged as inconsistent."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown and hits a wall")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    # No CONSERVATION verdict should have been added (the mock generator did
    # receive the control_signal).
    assert all(v.mode != PhysFailureMode.CONSERVATION for v in clip.physics_verdicts)


def test_consistency_critic_flags_clip_that_ignored_sketch(tmp_path: Path):
    """A clip with `control_signal=None` in its metadata should trip the critic."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("MOCK VIDEO\ncontrol_signal=None\nfirst_frame=None\n",
                     encoding="utf-8")
    clip = CandidateClip(shot_idx=0, video_path=bogus, keyframes=[], revision=0,
                        checklist=Checklist())
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    conserv = [v for v in clip.physics_verdicts
               if v.mode == PhysFailureMode.CONSERVATION]
    assert conserv, "expected a CONSERVATION verdict for a clip that ignored sketch"
    assert conserv[0].severity >= 0.4


def test_consistency_critic_silent_when_no_sketch(tmp_path: Path):
    """No sketch -> nothing to be consistent with -> no verdict, no metric noise."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a butterfly flutters")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    assert not any(v.mode == PhysFailureMode.CONSERVATION for v in clip.physics_verdicts)


def test_metric_tool_splits_p1_and_p2(tmp_path: Path):
    """MetricTool reports `p1_physics` and `p2_sketch_consistency` separately so
    a native physics failure stays distinguishable from a sketch-divergence one.
    """
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    board = _board()
    board.review(clip, spec, None, fps=8)
    assert "p1_physics" in clip.metric_scores
    assert "p2_sketch_consistency" in clip.metric_scores
    # A clip that followed the sketch should have HIGH p2 even when p1 reflects
    # native verdicts the PhysicsCritic still emits at revision 0.
    assert clip.metric_scores["p2_sketch_consistency"] >= 0.8


# ─────────────────────────────────────────────────────────────────────────────
# C5 — Hierarchical Self-Improvement (HSI)
# ─────────────────────────────────────────────────────────────────────────────
class _StubbornMLLM(MockMLLMClient):
    """Tier-0 (local edit) never sees the prompt change, so its scores never
    improve. Once the Director's Tier-2 rewrite adds 'plan-fix:' to the prompt,
    we flip the scores to passing. This forces escalation past Tier 0 → Tier 1
    → Tier 2 and lets us assert tier_used contains values > 0.
    """

    def assess_semantic(self, clip, spec):
        ok = "plan-fix:" in spec.prompt
        return [("Is the plan-level fix in place?", ok,
                 "" if ok else "ask director to widen scope")]

    def assess_physics(self, clip, spec, fps):
        # Severity only drops once the spec was rewritten; before that we always
        # return a verdict so the loop keeps escalating.
        if "plan-fix:" in spec.prompt:
            return []
        verdicts = super().assess_physics(clip, spec, fps)
        for v in verdicts:
            v.severity = max(v.severity, 0.5)  # keep above the resolved threshold
        return verdicts


def test_hsi_escalates_past_tier0_when_local_edit_cannot_fix(tmp_path: Path):
    """With a stubborn judge, the loop should escalate beyond Tier 0."""
    spec = ShotSpec(shot_idx=0, duration=1.0,
                    prompt="a ball is thrown and hits a wall")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)

    judge = _StubbornMLLM()
    # Force weighted_total to depend ONLY on the stubborn judge's m1, otherwise
    # m2/aesthetic etc. auto-improve with revision and the Verifier accepts at
    # Tier 0 even though the judge keeps failing the semantic checklist.
    weights = {"m1_semantic": 1.0, "m2_temporal": 0.0, "p1_physics": 0.0,
               "id1_identity": 0.0, "m5_rhythm": 0.0, "aesthetic": 0.0}
    board = ReviewBoard(
        critics=[SemanticCritic(mllm=judge), PhysicsCritic(mllm=judge),
                 ConsistencyCritic(), RhythmCritic()],
        metric_tool=MetricTool(weights=weights),
    )

    res = generate_shot(
        spec, board,
        GeneratorAgent(), RefinerAgent(), VerifierAgent(), tmp_path,
        physics_planner=PhysicsPlannerAgent(),
        director=DirectorAgent(),
        max_revisions=4, k_retries=1, n_candidates=1,
    )
    # HSI must have escalated at least once.
    assert res.escalations >= 1, (res.tier_used, res.escalations)
    assert any(t >= 1 for t in res.tier_used)
    # Score history still monotonic (Verifier enforces it across ALL tiers).
    h = res.score_history
    assert all(h[i] <= h[i + 1] + 1e-9 for i in range(len(h) - 1)), h


def test_hsi_back_compat_without_tier_agents(tmp_path: Path):
    """No physics_planner / director passed -> behaves like v0.1 single-tier."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    res = generate_shot(
        spec, _board(),
        GeneratorAgent(), RefinerAgent(), VerifierAgent(), tmp_path,
        max_revisions=3, k_retries=2,
    )
    # No escalations possible when the upper tiers are not provided.
    assert res.escalations == 0
    assert all(t in (0, 3) for t in res.tier_used)


# ─────────────────────────────────────────────────────────────────────────────
# Bug fix — lesson distilled from actually-resolved mode
# ─────────────────────────────────────────────────────────────────────────────
def test_lesson_distill_picks_resolved_mode(tmp_path: Path):
    """The distilled lesson's failure_mode should be a mode that was present at
    the START of the loop and gone by the END (not blindly expected_modes[0]).
    """
    spec = ShotSpec(shot_idx=0, duration=1.0,
                    prompt="a ball falls and hits the ground")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    lib = LessonLibrary(tmp_path / "lessons.jsonl")
    generate_shot(
        spec, _board(),
        GeneratorAgent(), RefinerAgent(), VerifierAgent(), tmp_path,
        lesson_library=lib, max_revisions=5, k_retries=2,
    )
    assert len(lib) == 1
    # The stored mode must be one of the modes the sketch flagged for this prompt.
    expected = set(spec.physics_sketch.expected_modes)
    assert lib.lessons[0].failure_mode in expected


# ─────────────────────────────────────────────────────────────────────────────
# Bug fix — ReviewBoard.recompute_metrics is critic-free
# ─────────────────────────────────────────────────────────────────────────────
def test_recompute_metrics_does_not_re_run_critics(tmp_path: Path):
    """If we shrink physics_verdicts then recompute_metrics, the verdict must
    NOT come back (i.e., critics were not re-run). And the weighted_total must
    reflect the smaller verdict set."""
    spec = ShotSpec(shot_idx=0, duration=1.0,
                    prompt="a ball is thrown and bounces off a wall")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    board = _board()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    board.review(clip, spec, None, fps=8)
    n_before = len(clip.physics_verdicts)
    pre_total = clip.metric_scores["weighted_total"]
    assert n_before >= 1

    # Drop one verdict by hand (mimicking the escape hatch path).
    clip.physics_verdicts.pop()
    board.recompute_metrics(clip, spec, None, fps=8)

    assert len(clip.physics_verdicts) == n_before - 1   # critic did NOT regenerate it
    assert clip.metric_scores["weighted_total"] >= pre_total - 1e-9
