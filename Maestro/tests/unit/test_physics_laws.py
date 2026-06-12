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


def test_unexplained_mode_is_verdict_only():
    """UNEXPLAINED (residual-only verdicts) has a real intervention entry but
    is never prompt-predicted: no keyword cue exists, so detect_expected_modes
    cannot emit it."""
    from maestro.physics.failure_modes import (
        FAILURE_MODE_KEYWORDS,
        INTERVENTION_LIBRARY,
    )

    assert PhysFailureMode.UNEXPLAINED in INTERVENTION_LIBRARY
    assert INTERVENTION_LIBRARY[PhysFailureMode.UNEXPLAINED]
    assert PhysFailureMode.UNEXPLAINED not in FAILURE_MODE_KEYWORDS
    every_keyword = " ".join(
        kw for kws in FAILURE_MODE_KEYWORDS.values() for kw in kws
    )
    assert PhysFailureMode.UNEXPLAINED not in detect_expected_modes(every_keyword)


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


def test_teleport_severity_is_graded_not_degenerate():
    """F1 regression: severity must span [0.5, 1.0] (double-trigger
    convention), not saturate at 1.0 for every detected jump."""
    # borderline jump: ~7x the median step -> severity ~7/12
    borderline = _linear(24)
    borderline[12] = (borderline[12][0] + 0.13, borderline[12][1])
    tele = [a for a in detect_anomalies(borderline) if a.kind == "teleport"]
    assert tele
    assert 0.5 <= tele[0].severity < 0.7

    # huge jump: saturates
    huge = _linear(24)
    huge[12] = (huge[12][0] + 0.4, huge[12][1])
    tele = [a for a in detect_anomalies(huge) if a.kind == "teleport"]
    assert tele and tele[0].severity == 1.0


def test_teleport_absolute_fallback_for_static_object():
    """F2 regression: a static object that teleports has median speed 0 —
    the relative gate is degenerate, the absolute fallback must catch it."""
    track = [(0.3, 0.3)] * 12 + [(0.7, 0.7)] * 12
    tele = [a for a in detect_anomalies(track) if a.kind == "teleport"]
    assert tele
    assert tele[0].frame_range == (11, 12)
    assert tele[0].severity == 1.0
    # a smaller jump grades below saturation: step 0.2 -> 0.2/(2*0.15)
    subtle = [(0.3, 0.3)] * 12 + [(0.4414, 0.4414)] * 12
    tele = [a for a in detect_anomalies(subtle) if a.kind == "teleport"]
    assert tele and 0.6 < tele[0].severity < 0.75


def test_jerk_absolute_fallback_for_polynomial_track():
    """F2 regression: an otherwise jerk-free track (static, then constant
    velocity) has median jerk 0 — the kink must still be flagged."""
    track = [(0.2, 0.5)] * 14 + [(0.2 + 0.06 * (i + 1), 0.5) for i in range(10)]
    anomalies = detect_anomalies(track)
    kinds = {a.kind for a in anomalies}
    assert kinds == {"jerk_spike"}        # no teleport: 0.06 < ABS_TELEPORT
    spike = next(a for a in anomalies if a.kind == "jerk_spike")
    assert abs(spike.severity - 0.6) < 0.01    # 0.06 / (2 * ABS_JERK)


def test_short_track_is_indeterminate_not_static():
    """F3 regression: < 4 frames cannot fit ANY law — labeling that 'static'
    certifies arbitrary garbage as explainable. (In the assembled verifier
    reliability.certify() MIN_FRAMES=4 gates this path as 'too_short'.)"""
    fit = fit_best_law([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])
    assert fit.law == "indeterminate"
    assert fit.residual == 0.0
    assert fit.params["n"] == 3
    # violation stays 0 (no evidence either way) but the label is honest
    assert analyze_track("x", [(0.1, 0.1), (0.9, 0.9)]).violation == 0.0
    # genuinely motionless tracks keep the meaningful "static" label
    assert fit_best_law([(0.5, 0.5)] * 10).law == "static"
    # and the verifier's reliability gate rejects short tracks upstream
    assert certify([(0.1, 0.1)] * 3).reason == "too_short"


def test_legit_bounce_at_global_ymax_not_flagged_as_reversal():
    """F4 (documented ambiguity, pinned behavior): a reversal AT the track's
    global y-max is indistinguishable from a legitimate bounce without a
    ground-plane estimate, so the detector must NOT flag it. (The dual
    ambiguity — a real bounce on an elevated surface IS flagged — is the
    documented false-positive class.)"""
    n = 24
    track = []
    for t in range(n):
        if t < n // 2:                       # fall to the deepest point
            y = 0.2 + 0.6 * t / (n // 2 - 1)
        else:                                # rebound upward
            y = 0.8 - 0.5 * (t - n // 2 + 1) / (n - n // 2)
        track.append((0.5, y))
    kinds = {a.kind for a in detect_anomalies(track)}
    assert "midair_reversal" not in kinds


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
    # F5 regression: IID positional noise (lost point) has direction churn
    # ~0.67-0.72 — within sampling noise of the old 0.65 limit. The 0.55
    # limit must reject it for any seed, not just a lucky one.
    import random
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        jitter = [(0.5 + rng.uniform(-0.1, 0.1), 0.5 + rng.uniform(-0.1, 0.1))
                  for _ in range(30)]
        cert = certify(jitter)
        assert not cert.certified and cert.reason == "jitter", f"seed {seed}"


def test_cross_tracker_disagreement_decertifies():
    a = _ballistic()
    b = [(x, 1.0 - y) for x, y in a]      # a "tracker" seeing opposite motion
    assert cross_agreement(a, a) < 0.01
    pair = certify_pair(a, b)
    assert not pair.certified and pair.reason == "tracker_disagreement"
    assert certify_pair(a, a).certified


def test_cross_agreement_static_static_uses_absolute_tolerance():
    """F6 regression: two healthy trackers staring at the same still object
    from slightly different seed points must AGREE — normalizing a constant
    offset by MIN_RANGE=1e-3 decertified them spuriously."""
    a = [(0.50, 0.50)] * 10
    b = [(0.51, 0.51)] * 10               # ~0.014 apart: within tolerance
    assert cross_agreement(a, b) < 0.5
    pair = certify_pair(a, b)
    assert pair.certified
    # but two static trackers far apart are watching DIFFERENT things
    c = [(0.70, 0.70)] * 10               # ~0.28 apart: not the same object
    assert cross_agreement(a, c) > 1.0
    assert not certify_pair(a, c).certified


def test_certify_pair_returns_a_failed_cert_on_confidence_tie(monkeypatch):
    """F7 regression: when one tracker fails certification, certify_pair must
    return one of the FAILED certs — a confidence rounding-tie used to let a
    CERTIFIED cert through, silently certifying the pair."""
    import maestro.physics.reliability as rel

    certs = iter([
        rel.TrackCertificate(True, 0.4, "ok"),       # a: certified
        rel.TrackCertificate(False, 0.4, "jitter"),  # b: failed, same confidence
    ])
    monkeypatch.setattr(rel, "certify", lambda track, fps=8: next(certs))
    pair = rel.certify_pair([(0.1, 0.1)] * 8, [(0.9, 0.9)] * 8)
    assert not pair.certified
    assert pair.reason == "jitter"
