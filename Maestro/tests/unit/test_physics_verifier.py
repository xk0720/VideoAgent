"""Tests for the v0.4 physics verification stack (reference-free).

Guards the repositioning:
  • the verifier asks "is there ANY physical explanation for the observed
    track?" — no simulated expectation anywhere;
  • verdicts are localized (entity, frame range, failure mode) and the
    coverage report says which tier verified what (S3 transparency);
  • the self-improve signal path: revision-0 mock clips contain a violation,
    refined clips don't — so HSI has something real (in mock terms) to fix;
  • optional world-model reward adds a `wm_reward` dimension without breaking
    metric output when unset.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro.agents.generator import GeneratorAgent
from maestro.critics.physics_consistency import PhysicsConsistencyCritic
from maestro.models.world_reward import MockWorldReward, build_world_reward
from maestro.physics.annotate import annotate_physics
from maestro.physics.verifier import PhysicsFromPixelsVerifier
from maestro.tools.metric_tool import MetricTool
from maestro.types import CandidateClip, Checklist, PhysFailureMode, ShotSpec


def _spec(prompt="a ball falls to the ground") -> ShotSpec:
    spec = ShotSpec(shot_idx=0, duration=2.0, prompt=prompt)
    spec.physics_annotation = annotate_physics(spec)
    return spec


# ─────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────
def test_verifier_none_without_annotation(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a still life")
    spec.physics_annotation = None
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    assert PhysicsFromPixelsVerifier().verify(clip, spec, fps=8) is None


def test_verifier_silent_when_clip_unreadable(tmp_path: Path):
    spec = _spec()
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "missing.mp4",
                         keyframes=[], revision=0, checklist=Checklist())
    assert PhysicsFromPixelsVerifier().verify(clip, spec, fps=8) is None


def test_verifier_flags_revision0_and_clears_after_refinement(tmp_path: Path):
    """The mock signal path: rev-0 tracks contain a mid-air reversal, refined
    tracks are clean — violations must fall monotonically."""
    spec = _spec()
    verifier = PhysicsFromPixelsVerifier()
    v0 = verifier.verify(GeneratorAgent().run(spec, tmp_path, revision=0), spec, 8)
    v1 = verifier.verify(GeneratorAgent().run(spec, tmp_path, revision=1), spec, 8)
    assert v0 is not None and v1 is not None
    assert v0.worst_violation > 0.3        # flagged
    assert v1.worst_violation < 0.1        # clean after refinement
    assert v0.worst_violation > v1.worst_violation


def test_verifier_coverage_reports_every_tier(tmp_path: Path):
    """S3: entities the measurement tier can't check appear as explicit
    deferrals, never disappear."""
    spec = _spec("a ball falls while a person runs")
    clip = GeneratorAgent().run(spec, tmp_path, revision=1)
    result = PhysicsFromPixelsVerifier().verify(clip, spec, fps=8)
    assert "measurement" in result.coverage and "world_model" in result.coverage
    assert "ball" in result.coverage["measurement"]
    assert "person" in result.coverage["world_model"]
    measured = {e.entity for e in result.entities if e.measured}
    assert measured == {"ball"}            # person was deferred, not measured


# ─────────────────────────────────────────────────────────────
# Critic: verdicts derive from the verifier
# ─────────────────────────────────────────────────────────────
def test_critic_flags_violation_with_localized_verdict(tmp_path: Path):
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    assert clip.physics_verdicts
    verdict = clip.physics_verdicts[0]
    assert verdict.mode == PhysFailureMode.GRAVITY_INERTIA   # mid-air reversal
    assert verdict.frame_range[0] > 0                        # localized
    assert "conditioning" not in verdict.suggested_intervention


def test_critic_clears_on_refined_clip(tmp_path: Path):
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=1)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    assert clip.physics_verdicts == []


def test_critic_silent_without_annotation(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="x")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    assert clip.physics_verdicts == []


def test_strictness_tightens_the_bar(tmp_path: Path):
    """HSI tier-1 replan raises strictness → a borderline clip that passed at
    1.0 must fail at high strictness. We use a threshold above the rev-0
    violation to construct the borderline."""
    spec = _spec()
    clip0 = GeneratorAgent().run(spec, tmp_path, revision=0)
    lenient = PhysicsConsistencyCritic(violation_threshold=0.99)
    lenient.review(clip0, spec, fps=8)
    assert clip0.physics_verdicts == []      # passes the lenient bar

    spec.physics_annotation.strictness = 3.0   # tier-1 replan effect
    clip0b = GeneratorAgent().run(spec, tmp_path, revision=0)
    lenient.review(clip0b, spec, fps=8)
    assert clip0b.physics_verdicts             # same clip now fails


# ─────────────────────────────────────────────────────────────
# World-model reward (WMReward-style slot, world_model tier hook)
# ─────────────────────────────────────────────────────────────
def test_world_reward_factory():
    assert build_world_reward(None) is None
    assert isinstance(build_world_reward({"name": "mock-world-reward"}), MockWorldReward)
    with pytest.raises(ValueError):
        build_world_reward({"name": "vjepa2"})  # not wired yet -> loud failure


def test_metric_tool_back_compat_without_world_reward(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    scores = MetricTool().run(clip, spec, None, fps=8)
    assert "wm_reward" not in scores     # unset -> identical keys to v0.2.2


def test_metric_tool_adds_wm_reward_and_it_improves(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    tool = MetricTool(
        weights={"wm_reward": 1.0},      # isolate the dimension
        world_reward=MockWorldReward(),
    )
    c0 = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    c2 = GeneratorAgent().run(spec, tmp_path, revision=2, seed=0, fps=8)
    s0, s2 = tool.run(c0, spec, None, 8), tool.run(c2, spec, None, 8)
    assert "wm_reward" in s0
    assert s2["wm_reward"] >= s0["wm_reward"]          # improves with refinement
    assert s2["weighted_total"] >= s0["weighted_total"]  # drives the search
