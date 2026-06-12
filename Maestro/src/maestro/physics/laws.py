"""Reference-free motion-law verification (C6, v0.4 "physics-from-pixels").

The question this module answers is NOT "does the motion match a simulation?"
(that presumes masses/friction/scale we cannot know from one prompt) but the
parameter-free question:

    "Is there ANY physically consistent explanation for the observed track?"

Given an observed screen-space track we (1) fit the small family of motion
laws a passive rigid body can follow — static, constant velocity, constant
acceleration (free gravity vector, so no scale calibration is needed) — and
take the residual to the BEST fit as the violation measure; and (2) run
anomaly detectors for the discrete failure modes a smooth fit can miss
(teleports, mid-air reversals, energy gain, acceleration spikes).

Why this is defensible (docs/research/survey_physics_2026_06.md):
  • Morpheus (2504.02918) shows conservation-style residuals work without a
    reference video — but uses them only for benchmarking, never selection.
  • PISA (2503.09595) shows trajectory residuals are a usable reward — but
    against sim ground truth for ONE phenomenon.
  • Equation-discovery forecasting (2507.06830) fits parametric dynamics to
    observed tracks — for forecasting, not verification.
  OUR increment: best-physical-fit residual + anomaly localization as a
  training-free, per-entity, per-frame-range VERDICT that drives best-of-N
  selection and targeted regeneration. Nobody closes that loop.

Everything is normalized screen space ([0,1], y grows downward), so gravity
appears as a positive-y constant acceleration of UNKNOWN magnitude — which is
exactly why we fit it instead of asserting 9.81.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import PhysFailureMode

Track = list[tuple[float, float]]   # normalized screen-space (x, y) per frame

# anomaly kind → the PhysFailureMode it evidences
ANOMALY_TO_MODE: dict[str, PhysFailureMode] = {
    "teleport": PhysFailureMode.OBJECT_PERMANENCE,
    "midair_reversal": PhysFailureMode.GRAVITY_INERTIA,
    "energy_gain": PhysFailureMode.CONSERVATION,
    "jerk_spike": PhysFailureMode.COLLISION,
}


@dataclass
class LawFit:
    """Best-fitting member of the passive-motion family."""

    law: str               # "static" | "constant_velocity" | "constant_acceleration"
                           # | "indeterminate" (< 4 frames: nothing fittable)
    residual: float        # RMS error / motion range, clipped to [0,1]
    params: dict = field(default_factory=dict)


@dataclass
class MotionAnomaly:
    kind: str                          # key of ANOMALY_TO_MODE
    frame_range: tuple[int, int]
    severity: float                    # 0-1
    note: str = ""

    @property
    def mode(self) -> PhysFailureMode:
        return ANOMALY_TO_MODE[self.kind]


@dataclass
class LawReport:
    """Verdict material for one entity: interpretable and localized."""

    entity: str
    fit: LawFit
    anomalies: list[MotionAnomaly] = field(default_factory=list)
    motion_range: float = 0.0

    @property
    def violation(self) -> float:
        """0 = physically explainable, 1 = no physical explanation."""
        worst_anomaly = max((a.severity for a in self.anomalies), default=0.0)
        return round(min(1.0, max(self.fit.residual, worst_anomaly)), 3)


MIN_RANGE = 1e-3      # below this motion there is nothing physical to verify
_TELEPORT_FACTOR = 6.0     # step > 6× median step = discontinuity
_REVERSAL_MIN_SPEED = 0.01  # per-frame speed that counts as "really moving"

# Robust-scale fallbacks. The teleport / jerk detectors normalize by the
# track's MEDIAN step / jerk (a robust scale estimate). On the cleanest
# violations that scale collapses to zero — a static object that teleports
# has median speed 0, an otherwise-polynomial track has median jerk 0 — and
# a relative gate would divide by nothing and silence the detector exactly
# when the evidence is strongest. When the median is degenerate (<= 1e-6) we
# fall back to these ABSOLUTE thresholds in normalized screen units/frame.
ABS_TELEPORT = 0.15   # any single step larger than 15% of the screen
ABS_JERK = 0.05       # acceleration discontinuity, units/frame³


def motion_range(track: Track) -> float:
    """Max displacement from the start — the scale everything normalizes by."""
    if not track:
        return 0.0
    arr = np.asarray(track, dtype=float)
    return float(np.max(np.linalg.norm(arr - arr[0], axis=1)))


def _fit_poly(track: "np.ndarray", deg: int) -> tuple[float, "np.ndarray"]:
    """Least-squares polynomial fit per axis; returns (RMS residual, coeffs)."""
    t = np.arange(len(track), dtype=float)
    coeffs = np.stack([np.polyfit(t, track[:, d], deg) for d in range(2)])
    pred = np.stack([np.polyval(coeffs[d], t) for d in range(2)], axis=1)
    rms = float(np.sqrt(np.mean(np.sum((track - pred) ** 2, axis=1))))
    return rms, coeffs


def fit_best_law(track: Track, fps: int = 8) -> LawFit:
    """Fit static / constant-velocity / constant-acceleration; keep the
    simplest law whose normalized residual is not meaningfully beaten by a
    more complex one (parsimony margin 20%)."""
    arr = np.asarray(track, dtype=float)
    rng = motion_range(track)
    if len(arr) < 4:
        # Too few samples to fit ANY law — calling that "static" would label
        # arbitrary garbage as physically explainable. "indeterminate" keeps
        # violation at 0 (no evidence either way) but the label is honest.
        return LawFit(law="indeterminate", residual=0.0, params={"n": len(arr)})
    if rng < MIN_RANGE:
        return LawFit(law="static", residual=0.0, params={"range": rng})

    candidates: list[LawFit] = []
    rms0 = float(np.sqrt(np.mean(np.sum((arr - arr.mean(axis=0)) ** 2, axis=1))))
    candidates.append(LawFit("static", min(1.0, rms0 / rng)))
    rms1, c1 = _fit_poly(arr, 1)
    candidates.append(LawFit(
        "constant_velocity", min(1.0, rms1 / rng),
        {"velocity": (float(c1[0][0]) * fps, float(c1[1][0]) * fps)},
    ))
    rms2, c2 = _fit_poly(arr, 2)
    # acceleration = 2 * quadratic coefficient, per frame² → per second²
    accel = (float(c2[0][0]) * 2 * fps * fps, float(c2[1][0]) * 2 * fps * fps)
    candidates.append(LawFit(
        "constant_acceleration", min(1.0, rms2 / rng), {"acceleration": accel},
    ))

    best = candidates[0]
    for cand in candidates[1:]:
        if cand.residual < best.residual * 0.8:   # parsimony: must clearly win
            best = cand
    return best


def detect_anomalies(track: Track, fps: int = 8) -> list[MotionAnomaly]:
    """Discrete physical impossibilities a smooth best-fit can absorb."""
    arr = np.asarray(track, dtype=float)
    if len(arr) < 4 or motion_range(track) < MIN_RANGE:
        return []
    vel = np.diff(arr, axis=0)                       # per-frame velocity
    speed = np.linalg.norm(vel, axis=1)
    anomalies: list[MotionAnomaly] = []

    # 1. teleport — a step hugely out of scale with the rest of the motion.
    #    Severity follows the jerk detector's double-trigger convention: a
    #    step at exactly the trigger (6× median) scores 0.5, twice the
    #    trigger (12× median) saturates at 1.0 — so severity spans [0.5, 1.0]
    #    instead of degenerating to a constant.
    med = float(np.median(speed))
    if med > 1e-6:
        jumps = np.where(speed > _TELEPORT_FACTOR * med)[0]
        if len(jumps):
            f = int(jumps[0])
            anomalies.append(MotionAnomaly(
                kind="teleport", frame_range=(f, f + 1),
                severity=min(1.0, float(speed[f] / (2 * _TELEPORT_FACTOR * med))),
                note=f"step {speed[f]:.3f} vs median {med:.3f}",
            ))
    else:
        # Median speed ~0: a static object that teleports — the relative
        # gate has no scale, so use the absolute fallback (see ABS_TELEPORT).
        jumps = np.where(speed > ABS_TELEPORT)[0]
        if len(jumps):
            f = int(jumps[0])
            anomalies.append(MotionAnomaly(
                kind="teleport", frame_range=(f, f + 1),
                severity=min(1.0, float(speed[f] / (2 * ABS_TELEPORT))),
                note=f"step {speed[f]:.3f} from rest (absolute gate)",
            ))

    # 2. mid-air reversal — vertical velocity flips sign while clearly moving,
    #    NOT at the lowest screen point (a bounce at a surface is legitimate;
    #    y grows downward so a surface contact is a local y-maximum).
    #
    #    SINGLE-GROUND-PLANE ASSUMPTION + two known ambiguity classes:
    #    we have no ground-plane estimate, so "the surface" is approximated
    #    by the track's GLOBAL y-maximum. Consequences:
    #      (a) MISS — a reversal AT the track's global y-max (the object
    #          never falls deeper afterwards) is indistinguishable from a
    #          legitimate bounce, so it is NOT flagged, even if it was in
    #          fact a mid-air reversal that ended the clip.
    #      (b) FALSE-POSITIVE — a legitimate bounce on an ELEVATED surface
    #          (a table top that sits above the track's deepest later point)
    #          IS flagged: the contact happened above the global y-max, which
    #          to this detector looks like thin air.
    #    A heuristic "local maximum" test cannot discriminate either case:
    #    at any +→− reversal, y[i] IS a local maximum by construction. Doing
    #    better requires an actual ground/support-plane estimate.
    vy = vel[:, 1]
    y = arr[:, 1]
    y_max = float(y.max())
    for i in range(1, len(vy)):
        if (vy[i - 1] > _REVERSAL_MIN_SPEED and vy[i] < -_REVERSAL_MIN_SPEED
                and y[i] < y_max - 0.05):
            anomalies.append(MotionAnomaly(
                kind="midair_reversal", frame_range=(i - 1, i + 1),
                severity=min(1.0, float(abs(vy[i] - vy[i - 1]) / max(speed.mean(), 1e-6)) / 4),
                note=f"vy {vy[i-1]:+.3f} → {vy[i]:+.3f} away from contact",
            ))
            break

    # 3. energy gain — a passive object steadily speeding up with no contact
    #    (monotone speed growth over the whole track, ending ≥2× initial)
    if len(speed) >= 6:
        s0, s1 = float(speed[: len(speed) // 3].mean()), float(speed[-len(speed) // 3:].mean())
        increments = np.diff(speed)
        if s0 > 1e-4 and s1 > 2.0 * s0 and float((increments > 0).mean()) > 0.8:
            anomalies.append(MotionAnomaly(
                kind="energy_gain", frame_range=(0, len(track)),
                severity=min(1.0, (s1 / s0 - 1.0) / 4.0),
                note=f"speed {s0:.3f} → {s1:.3f} with no cause",
            ))

    # 4. jerk spike — acceleration discontinuity far above the track's norm
    if len(vel) >= 4:
        acc = np.diff(vel, axis=0)
        jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1)
        med_j = float(np.median(jerk))
        if med_j > 1e-6:
            spikes = np.where(jerk > 8.0 * med_j)[0]
            if len(spikes):
                f = int(spikes[0])
                anomalies.append(MotionAnomaly(
                    kind="jerk_spike", frame_range=(f, f + 3),
                    severity=min(1.0, float(jerk[f] / (16.0 * med_j))),
                    note=f"jerk {jerk[f]:.4f} vs median {med_j:.4f}",
                ))
        else:
            # Median jerk ~0: an otherwise-polynomial track (constant jerk
            # everywhere except the spike) — the robust scale collapses, so
            # use the absolute fallback (see ABS_JERK).
            spikes = np.where(jerk > ABS_JERK)[0]
            if len(spikes):
                f = int(spikes[0])
                anomalies.append(MotionAnomaly(
                    kind="jerk_spike", frame_range=(f, f + 3),
                    severity=min(1.0, float(jerk[f] / (2 * ABS_JERK))),
                    note=f"jerk {jerk[f]:.4f} on a jerk-free track (absolute gate)",
                ))
    return anomalies


def analyze_track(entity: str, track: Track, fps: int = 8) -> LawReport:
    """Fit the best passive law + run anomaly detectors for one entity.

    Tracks with fewer than 4 frames come back as law="indeterminate"
    (violation 0: no evidence either way, but never labeled "static"). In the
    assembled verifier this case is gated upstream: reliability.certify()
    rejects tracks shorter than MIN_FRAMES=4 ("too_short"), so analyze_track
    only sees an indeterminate-length track when called directly.
    """
    fit = fit_best_law(track, fps)
    anomalies = detect_anomalies(track, fps)
    # A falling object legitimately gains speed (gravity does work). Energy
    # gain is only evidence of a violation when NO passive law explains the
    # track — if a clean constant-velocity/acceleration fit exists, the speed
    # change is already accounted for.
    if fit.law != "static" and fit.residual < 0.1:
        anomalies = [a for a in anomalies if a.kind != "energy_gain"]
    return LawReport(
        entity=entity,
        fit=fit,
        anomalies=anomalies,
        motion_range=round(motion_range(track), 4),
    )
