"""Unit tests for the reference-free physics core (C6 v0.4):
annotation → router → reliability gate → law fitting / anomaly detection.

No simulator, no expected trajectory anywhere — the question under test is
"does the LAW layer recognize physically explainable vs inexplicable tracks?"
"""
from __future__ import annotations

from maestro.physics.annotate import annotate_physics
from maestro.physics.failure_modes import detect_expected_modes, suggest_intervention
from maestro.physics.laws import (
    analyze_track,
    detect_anomalies,
    fit_best_law,
    motion_range,
)
from maestro.physics.reliability import certify, certify_pair, cross_agreement
from maestro.physics.router import coverage_summary, route
from maestro.types import PhysFailureMode, ShotSpec


def _ballistic(n=24):
    """Clean constant-acceleration fall (y grows downward)."""
    return [(0.2 + 0.3 * t / (n - 1), 0.1 + 0.6 * (t / (n - 1)) ** 2)
            for t in range(n)]


def _linear(n=24):
    return [(0.1 + 0.5 * t / (n - 1), 0.5) for t in range(n)]


# ─────────────────────────────────────────────────────────────
# Failure-mode keyword layer (unchanged from v0.1)
# ─────────────────────────────────────────────────────────────
def test_detect_expected_modes_keywords():
    modes = detect_expected_modes("a ball is thrown and falls to the ground")
    assert PhysFailureMode.GRAVITY_INERTIA in modes


def test_suggest_intervention_has_entries_for_all_modes():
    for mode in PhysFailureMode:
        assert suggest_intervention(mode)


# ─────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────
def test_annotation_extracts_entities_and_modes():
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball falls and bounces")
    ann = annotate_physics(spec)
    names = [e.name for e in ann.entities]
    assert "ball" in names
    ball = next(e for e in ann.entities if e.name == "ball")
    assert ball.motion_class == "ballistic"
    assert any(it.kind == "collision" for it in ann.interactions)
    assert ann.expected_modes  # failure modes to watch


def test_annotation_motion_classes():
    agentive = annotate_physics(ShotSpec(0, 1.0, prompt="a person runs"))
    assert agentive.entities[0].motion_class == "agentive"
    fluid = annotate_physics(ShotSpec(0, 1.0, prompt="water pours into a cup"))
    assert any(e.motion_class == "fluid" for e in fluid.entities)
    still = annotate_physics(ShotSpec(0, 1.0, prompt="a bottle on a table"))
    assert all(e.motion_class == "static" for e in still.entities)


def test_annotation_strictness_passthrough():
    spec = ShotSpec(0, 1.0, prompt="a ball falls")
    assert annotate_physics(spec, strictness=2.0).strictness == 2.0


# ─────────────────────────────────────────────────────────────
# Router (S3: explicit coverage, never silent)
# ─────────────────────────────────────────────────────────────
def test_router_tiers_by_motion_class():
    spec = ShotSpec(0, 1.0, prompt="a ball falls while a person runs")
    decisions = route(annotate_physics(spec))
    tier = {d.entity: d.tier for d in decisions}
    assert tier["ball"] == "measurement"
    assert tier["person"] == "world_model"     # agentive: no parametric law
    cov = coverage_summary(decisions)
    assert set(cov) <= {"measurement", "world_model", "vlm", "none"}


def test_router_fluid_interaction_demotes_measurement():
    spec = ShotSpec(0, 1.0, prompt="a ball falls with a splash of water")
    decisions = route(annotate_physics(spec))
    ball = next(d for d in decisions if d.entity == "ball")
    assert ball.tier == "world_model" and "fluid" in ball.reason


def test_router_none_without_annotation():
    assert route(None) == []


# ─────────────────────────────────────────────────────────────
# Law fitting — "is there ANY physical explanation?"
# ─────────────────────────────────────────────────────────────
def test_fit_recognizes_clean_laws():
    assert fit_best_law(_linear()).law == "constant_velocity"
    ballistic = fit_best_law(_ballistic())
    assert ballistic.law == "constant_acceleration"
    assert ballistic.residual < 0.05
    assert fit_best_law([(0.5, 0.5)] * 10).law == "static"


def test_fit_flags_inexplicable_motion():
    import math
    wiggle = [(0.5 + 0.3 * math.sin(t * 1.7), 0.5 + 0.3 * math.cos(t * 2.3))
              for t in range(24)]
    assert fit_best_law(wiggle).residual > 0.2


def test_anomaly_midair_reversal():
    """Falls, reverses upward mid-air, then falls deeper — inexplicable."""
    from maestro.physics.tracks import _violating_fall
    track = _violating_fall(0.5, 24)
    anomalies = detect_anomalies(track)
    kinds = {a.kind for a in anomalies}
    assert "midair_reversal" in kinds
    rev = next(a for a in anomalies if a.kind == "midair_reversal")
    assert rev.mode == PhysFailureMode.GRAVITY_INERTIA
    assert rev.frame_range[0] > 0      # localized, not whole-clip


def test_anomaly_teleport():
    track = _linear(24)
    track[12] = (track[12][0] + 0.4, track[12][1])     # discontinuity
    kinds = {a.kind for a in detect_anomalies(track)}
    assert "teleport" in kinds


def test_clean_track_has_no_anomalies_and_low_violation():
    report = analyze_track("ball", _ballistic(), fps=8)
    assert report.anomalies == []
    assert report.violation < 0.1


def test_motion_range_basics():
    assert motion_range([]) == 0.0
    assert abs(motion_range([(0.5, 0.5), (0.7, 0.5)]) - 0.2) < 1e-9


# ─────────────────────────────────────────────────────────────
# Reliability gate (S2: certify before trusting)
# ─────────────────────────────────────────────────────────────
def test_certify_accepts_clean_and_static():
    assert certify(_ballistic()).certified
    assert certify([(0.5, 0.5)] * 10).certified


def test_certify_rejects_garbage():
    assert not certify([]).certified
    assert not certify([(0.5, 0.5)] * 2).certified            # too short
    assert not certify([(5.0, 5.0)] * 10 + [(6.0, 6.0)]).certified  # off-frame
    import random
    rng = random.Random(0)
    jitter = [(0.5 + rng.uniform(-0.1, 0.1), 0.5 + rng.uniform(-0.1, 0.1))
              for _ in range(30)]
    cert = certify(jitter)
    assert not cert.certified and cert.reason == "jitter"


def test_cross_tracker_disagreement_decertifies():
    a = _ballistic()
    b = [(x, 1.0 - y) for x, y in a]      # a "tracker" seeing opposite motion
    assert cross_agreement(a, a) < 0.01
    pair = certify_pair(a, b)
    assert not pair.certified and pair.reason == "tracker_disagreement"
    assert certify_pair(a, a).certified
