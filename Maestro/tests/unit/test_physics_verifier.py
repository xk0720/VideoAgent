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

import math
import random

from maestro.agents.generator import GeneratorAgent
from maestro.critics.physics import PhysicsCritic
from maestro.critics.physics_consistency import PhysicsConsistencyCritic
from maestro.models.world_reward import MockWorldReward, build_world_reward
from maestro.physics.annotate import annotate_physics
from maestro.physics.tracks import BaseTrackExtractor
from maestro.physics.verifier import PhysicsFromPixelsVerifier
from maestro.tools.metric_tool import MetricTool
from maestro.types import CandidateClip, Checklist, PhysFailureMode, ShotSpec


class _JitterExtractor(BaseTrackExtractor):
    """Returns a lost-point jitter track for every entity — certification
    must fail and the entity must be demoted, never measured."""

    def extract(self, clip, spec, entities, fps):
        rng = random.Random(0)
        return {
            e.name: [(0.5 + rng.uniform(-0.1, 0.1), 0.5 + rng.uniform(-0.1, 0.1))
                     for _ in range(24)]
            for e in entities
        }


class _CircleExtractor(BaseTrackExtractor):
    """Smooth circular motion: certifiable, anomaly-free, but no passive law
    fits — the residual-only violation path."""

    def extract(self, clip, spec, entities, fps):
        return {
            e.name: [(0.5 + 0.3 * math.sin(0.5 * t), 0.5 + 0.3 * math.cos(0.5 * t))
                     for t in range(24)]
            for e in entities
        }


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
    deferrals, never disappear. Coverage reflects FINAL tiers: a certified
    measurement entity stays in the measurement bucket."""
    spec = _spec("a ball falls while a person runs")
    clip = GeneratorAgent().run(spec, tmp_path, revision=1)
    result = PhysicsFromPixelsVerifier().verify(clip, spec, fps=8)
    assert "measurement" in result.coverage and "world_model" in result.coverage
    assert "ball" in result.coverage["measurement"]
    assert "person" in result.coverage["world_model"]
    measured = {e.entity for e in result.entities if e.measured}
    assert measured == {"ball"}            # person was deferred, not measured
    ball = next(e for e in result.entities if e.entity == "ball")
    assert ball.certificate.certified and ball.tier == "measurement"
    assert result.uncertified == []


def test_verifier_mixed_tier_unreadable_clip(tmp_path: Path):
    """F9 regression: when the clip is unreadable but non-measurement tiers
    exist, measurement entities must say 'clip_unreadable' — not the
    'too_short' lie — and stay uncertified measurement deferrals (a VLM has
    nothing to look at either)."""
    spec = _spec("a ball falls while a person runs")
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "missing.mp4",
                         keyframes=[], revision=0, checklist=Checklist())
    result = PhysicsFromPixelsVerifier().verify(clip, spec, fps=8)
    assert result is not None              # person still gets its deferral
    ball = next(e for e in result.entities if e.entity == "ball")
    assert ball.tier == "measurement"      # NOT demoted: no pixels for a vlm
    assert not ball.certificate.certified
    assert ball.certificate.reason == "clip_unreadable"
    assert ball.report is None
    person = next(e for e in result.entities if e.entity == "person")
    assert person.tier == "world_model"
    assert "ball" in result.coverage["measurement"]
    assert result.uncertified == ["ball"]


def test_verifier_no_track_reason_and_demotion(tmp_path: Path):
    """F9: an entity missing from a readable extraction gets 'no_track' (and
    is demoted: the clip is viewable, so a VLM can still check it)."""
    class _EmptyExtractor(BaseTrackExtractor):
        def extract(self, clip, spec, entities, fps):
            return {}                      # readable clip, no entity found

    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    result = PhysicsFromPixelsVerifier(extractor=_EmptyExtractor()).verify(
        clip, spec, fps=8)
    ball = next(e for e in result.entities if e.entity == "ball")
    assert ball.certificate.reason == "no_track"
    assert ball.tier == "vlm"


def test_verifier_demotes_uncertified_entity_to_vlm(tmp_path: Path):
    """F10 regression: a decertified measurement entity is DEMOTED to the
    vlm tier and the coverage report is rebuilt from final tiers — the
    docstring's promised fallback routing actually happens."""
    spec = _spec("a ball falls while a person runs")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    result = PhysicsFromPixelsVerifier(extractor=_JitterExtractor()).verify(
        clip, spec, fps=8)
    ball = next(e for e in result.entities if e.entity == "ball")
    assert ball.tier == "vlm"                          # demoted
    assert not ball.certificate.certified
    assert ball.certificate.reason == "jitter"
    assert ball.report is None                         # never measured
    assert result.measured_reports == []
    # coverage is truthful: ball sits in the vlm bucket, not measurement
    assert "ball" in result.coverage["vlm"]
    assert "ball" not in result.coverage.get("measurement", [])
    assert "person" in result.coverage["world_model"]
    assert result.uncertified == ["ball"]


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


def test_board_order_independence_measured_verdicts_survive(tmp_path: Path):
    """F8 regression: PhysicsCritic must EXTEND clip.physics_verdicts, never
    replace it — running the consistency critic BEFORE the VLM critic must
    not lose the measured verdicts."""
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)     # appends measured
    measured_before = [v for v in clip.physics_verdicts
                       if v.source == "law_verifier"]
    assert measured_before
    PhysicsCritic().review(clip, spec, fps=8)                # VLM critic second
    measured_after = [v for v in clip.physics_verdicts
                      if v.source == "law_verifier"]
    assert measured_after == measured_before                 # survived
    assert any(v.source != "law_verifier" for v in clip.physics_verdicts)


def test_residual_only_violation_labeled_unexplained(tmp_path: Path):
    """F11 regression: a certifiable track with no localized anomaly but no
    fitting passive law is UNEXPLAINED — not a fabricated gravity claim."""
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    PhysicsConsistencyCritic(extractor=_CircleExtractor()).review(
        clip, spec, fps=8)
    measured = [v for v in clip.physics_verdicts if v.source == "law_verifier"]
    assert measured
    assert measured[0].mode == PhysFailureMode.UNEXPLAINED
    assert "no physical explanation" in measured[0].suggested_intervention


def test_critic_silent_without_annotation(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="x")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0)
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    assert clip.physics_verdicts == []


def test_strictness_tightens_the_bar(tmp_path: Path):
    """MECHANISM test: annotation strictness > 1.0 shrinks the critic's
    violation threshold, so a borderline clip that passed at 1.0 fails at
    high strictness. NOTE this is NOT the tier-1 repair path — tier-1
    replans keep strictness at 1.0 (tightening the bar on a failing shot
    inverts the repair incentive); strictness > 1.0 is only used by the
    log-only post-acceptance hardening pass (physics.post_accept_strictness).
    We use a threshold above the rev-0 violation to construct the borderline."""
    spec = _spec()
    clip0 = GeneratorAgent().run(spec, tmp_path, revision=0)
    lenient = PhysicsConsistencyCritic(violation_threshold=0.99)
    lenient.review(clip0, spec, fps=8)
    assert clip0.physics_verdicts == []      # passes the lenient bar

    spec.physics_annotation.strictness = 3.0   # post-accept watchdog effect
    clip0b = GeneratorAgent().run(spec, tmp_path, revision=0)
    lenient.review(clip0b, spec, fps=8)
    assert clip0b.physics_verdicts             # same clip now fails


def test_tier1_accepts_repair_at_unchanged_bar(tmp_path: Path):
    """Tier-1 semantics (F1): a shot whose physics check fails at rev 0
    escalates to Tier 1 when local edits cannot help, and the Tier-1
    regeneration (verdict-derived hints; revision bump → clean mock track)
    is ACCEPTED at the UNCHANGED 0.4 violation bar — no strictness
    tightening anywhere in the failing-shot path."""
    from maestro.agents.physics_planner import PhysicsPlannerAgent
    from maestro.agents.refiner import RefinerAgent
    from maestro.agents.verifier import VerifierAgent
    from maestro.critics.board import ReviewBoard
    from maestro.pipeline.generate_loop import generate_shot

    class _LocalEditCannotFix(GeneratorAgent):
        """Tier-0 candidates (seeds < 100) keep the violating motion — the
        keyframe-level local edit cannot repair this defect. Tier-1 seeds
        (100+) regenerate honestly (revision bump → clean mock track)."""

        def run(self, spec, cache_dir, revision=0, seed=0, **kwargs):
            if seed < 100:
                revision = 0
            return super().run(spec, cache_dir, revision=revision,
                               seed=seed, **kwargs)

    spec = _spec("a ball falls to the ground")
    critic = PhysicsConsistencyCritic(violation_threshold=0.4)  # unchanged bar
    # Weights pinned so p2 (the measured law check) dominates the Verifier.
    board = ReviewBoard(critics=[critic],
                        metric_tool=MetricTool(weights={"p2_law_consistency": 1.0}))

    res = generate_shot(
        spec, board, _LocalEditCannotFix(), RefinerAgent(), VerifierAgent(),
        tmp_path, physics_planner=PhysicsPlannerAgent(),
        n_candidates=1, max_revisions=3, k_retries=1,
    )
    # Escalated to Tier 1 and the Tier-1 candidate was accepted.
    assert res.tier_used == [1], res.tier_used
    assert res.converged and not res.escape_hatched
    assert res.clip.physics_verdicts == []         # clean at the 0.4 bar
    # The bar never moved: critic threshold unchanged, strictness back at 1.0.
    assert critic.threshold == 0.4
    assert spec.physics_annotation.strictness == 1.0


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
