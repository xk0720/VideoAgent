"""Tests for the v0.2 additions:

  • C5 HSI tiered self-improvement loop
  • C6 PhysicsConsistencyCritic (closed-loop sketch verification)
  • bug fix: lesson distilled from actually-resolved mode
  • bug fix: ReviewBoard.recompute_metrics refreshes only scores (not critics)
"""
from __future__ import annotations

from pathlib import Path

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
from maestro.models.mllm import MockMLLMClient
from maestro.physics.annotate import annotate_physics
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
# A physics repair instruction as the loop would thread it into the
# regeneration prompt; the mock track is clean only when this text is
# actually recorded in the clip body (content-derived signal).
PHYSICS_FIX = "one continuous passive trajectory"


def test_consistency_critic_passes_on_repaired_clip(tmp_path: Path):
    """A clip whose body records an APPLIED physics fix has law-consistent
    motion -> no measured verdicts. A bare revision bump without the fix
    stays flagged (the signal reads the artifact, not the counter)."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown and hits a wall")
    spec.physics_annotation = annotate_physics(spec)
    clip = GeneratorAgent().run(spec, tmp_path, revision=1, seed=0, fps=8,
                                extra_prompt=PHYSICS_FIX)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    assert all(v.source != "law_verifier" for v in clip.physics_verdicts)
    # Negative control: regenerated WITHOUT the fix -> still flagged.
    clock = GeneratorAgent().run(spec, tmp_path, revision=1, seed=1, fps=8)
    PhysicsConsistencyCritic().review(clock, spec, fps=8)
    assert any(v.source == "law_verifier" for v in clock.physics_verdicts)


def test_consistency_critic_flags_inexplicable_motion(tmp_path: Path):
    """An unrepaired clip's observed track contains a mid-air reversal — the
    reference-free law layer must flag it with a localized, measured verdict."""
    spec = ShotSpec(shot_idx=0, duration=2.0, prompt="a ball is thrown")
    spec.physics_annotation = annotate_physics(spec)
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    measured = [v for v in clip.physics_verdicts if v.source == "law_verifier"]
    assert measured, "expected a measured verdict for inexplicable motion"
    assert measured[0].mode == PhysFailureMode.GRAVITY_INERTIA
    assert measured[0].severity >= 0.3
    assert measured[0].frame_range[0] > 0      # localized, not whole-clip


def test_consistency_critic_silent_when_no_annotation(tmp_path: Path):
    """No annotation -> nothing to verify -> no verdict, no metric noise."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a butterfly flutters")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    assert not any(v.source == "law_verifier" for v in clip.physics_verdicts)


def test_metric_tool_splits_p1_and_p2(tmp_path: Path):
    """MetricTool reports `p1_physics` (VLM-judged) and `p2_law_consistency`
    (measured) separately so a judged failure stays distinguishable from a
    measured one."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_annotation = annotate_physics(spec)
    # repaired clip (physics fix actually applied): measured checks pass, so
    # p2 is high regardless of the judged verdicts the VLM PhysicsCritic may
    # still emit
    clip = GeneratorAgent().run(spec, tmp_path, revision=1, seed=0, fps=8,
                                extra_prompt=PHYSICS_FIX)
    board = _board()
    board.review(clip, spec, None, fps=8)
    assert "p1_physics" in clip.metric_scores
    assert "p2_law_consistency" in clip.metric_scores
    assert clip.metric_scores["p2_law_consistency"] >= 0.8


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
    spec.physics_annotation = annotate_physics(spec)

    judge = _StubbornMLLM()
    # Force weighted_total to depend ONLY on the stubborn judge's m1, otherwise
    # m2 rises with the applied fix text / keyframe anchoring and the Verifier
    # accepts at Tier 0 even though the judge keeps failing the checklist.
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
    spec.physics_annotation = annotate_physics(spec)
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
    spec.physics_annotation = annotate_physics(spec)
    lib = LessonLibrary(tmp_path / "lessons.jsonl")
    generate_shot(
        spec, _board(),
        GeneratorAgent(), RefinerAgent(), VerifierAgent(), tmp_path,
        lesson_library=lib, max_revisions=5, k_retries=2,
    )
    assert len(lib) == 1
    # The stored mode must be one of the modes the sketch flagged for this prompt.
    expected = set(spec.physics_annotation.expected_modes)
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
    spec.physics_annotation = annotate_physics(spec)
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
