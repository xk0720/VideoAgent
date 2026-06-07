"""Tests for the v0.3 physics repositioning (sim = verification oracle).

Guards the conclusions of PHYSICS_LITERATURE_REVIEW.md:
  • oracle compares EXPECTED (sim) vs OBSERVED (extracted) motion — PISA-style
  • critic verdicts derive from that comparison, not from conditioning metadata
  • optional world-model reward adds a `wm_reward` dimension (WMReward-style)
    without breaking v0.2.2 metric output when unset
"""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro.agents.generator import GeneratorAgent
from maestro.critics.physics_consistency import PhysicsConsistencyCritic
from maestro.models.world_reward import MockWorldReward, build_world_reward
from maestro.physics.oracle import TrajectoryOracle, _max_l2, _motion_range
from maestro.physics.sketch import build_physics_sketch
from maestro.tools.metric_tool import MetricTool
from maestro.types import CandidateClip, Checklist, PhysFailureMode, ShotSpec


# ─────────────────────────────────────────────────────────────────────────────
# Oracle math
# ─────────────────────────────────────────────────────────────────────────────
def test_motion_range_and_l2_basics():
    track = [(0.5, 0.5), (0.6, 0.5), (0.7, 0.5)]
    assert abs(_motion_range(track) - 0.2) < 1e-9
    flat = [(0.5, 0.5)] * 3
    assert abs(_max_l2(track, flat) - 0.2) < 1e-9
    assert _motion_range([]) == 0.0


def test_oracle_returns_none_without_sketch(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a still life")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    assert TrajectoryOracle().compare(clip, spec, fps=8) is None


def test_oracle_low_deviation_when_motion_observed_matches(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    cmp = TrajectoryOracle().compare(clip, spec, fps=8)
    assert cmp is not None
    assert cmp.deviation <= 0.2          # followed: small relative error
    assert cmp.worst_entity in cmp.per_entity


def test_oracle_max_deviation_when_motion_ignored(tmp_path: Path):
    """A clip whose observed motion is flat (no gravity arc) must score ~1.0
    relative deviation — regardless of any conditioning metadata semantics."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("MOCK VIDEO\ncontrol_signal=None\n", encoding="utf-8")
    clip = CandidateClip(shot_idx=0, video_path=bogus, keyframes=[],
                         revision=0, checklist=Checklist())
    cmp = TrajectoryOracle().compare(clip, spec, fps=8)
    assert cmp is not None
    assert cmp.deviation >= 0.9


def test_oracle_silent_when_extraction_fails(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "missing.mp4",
                         keyframes=[], revision=0, checklist=Checklist())
    assert TrajectoryOracle().compare(clip, spec, fps=8) is None


def test_oracle_deviation_shrinks_with_revisions(tmp_path: Path):
    """Refinement rounds must drive observed motion toward the expectation —
    the property that makes the oracle a usable test-time search signal."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    oracle = TrajectoryOracle()
    devs = []
    for rev in (0, 1, 2):
        clip = GeneratorAgent().run(spec, tmp_path, revision=rev, seed=0, fps=8)
        devs.append(oracle.compare(clip, spec, fps=8).deviation)
    assert devs[0] >= devs[1] >= devs[2]


# ─────────────────────────────────────────────────────────────────────────────
# Critic uses the oracle (no conditioning-metadata semantics in the verdict)
# ─────────────────────────────────────────────────────────────────────────────
def test_critic_verdict_comes_from_oracle(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    bogus = tmp_path / "b.mp4"
    bogus.write_text("MOCK VIDEO\ncontrol_signal=None\n", encoding="utf-8")
    clip = CandidateClip(shot_idx=0, video_path=bogus, keyframes=[],
                         revision=0, checklist=Checklist())
    PhysicsConsistencyCritic().review(clip, spec, fps=8)
    conserv = [v for v in clip.physics_verdicts
               if v.mode == PhysFailureMode.CONSERVATION]
    assert conserv and conserv[0].severity >= 0.9
    # the intervention is oracle-language (re-anchor keyframes / reseed), not
    # "tighten conditioning" — the retired claim must not resurface
    assert "conditioning" not in conserv[0].suggested_intervention


# ─────────────────────────────────────────────────────────────────────────────
# World-model reward (WMReward-style slot)
# ─────────────────────────────────────────────────────────────────────────────
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
