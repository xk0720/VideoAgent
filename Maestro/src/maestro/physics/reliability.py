"""Tracker-reliability gating (C6 / S2) — certify the verifier's own input.

Point trackers are trained on REAL video. On generated video (flicker,
morphing, identity drift) they can silently emit plausible-looking tracks of
implausible objects — so a physics verdict is only as trustworthy as the
track it was computed from. This module certifies tracks BEFORE the law
checks run; the assembled verifier (physics/verifier.py) DEMOTES a
decertified measurement entity to the "vlm" tier instead of letting it
produce a confident-but-wrong measured verdict (exception: when the clip
itself is unreadable there is nothing for a VLM to look at either, so the
entity stays an uncertified measurement deferral).

No published work quantifies or gates on tracker reliability over generated
content (survey_physics_2026_06.md §4.1) — the gate, and "tracker
disagreement is itself an implausibility cue", are our increment.

Checks (all training-free, all cheap):
  • integrity      — finite, in-frame, long enough to fit laws on;
  • jitter         — frame-to-frame direction churn far above what any
                     physical motion produces (tracker lost the point);
  • cross-tracker  — when two extractors are available, their per-frame
                     disagreement normalized by motion range. High
                     disagreement ⇒ either a bad tracker or a morphing
                     object — both mean "don't trust a measured verdict".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .laws import MIN_RANGE, Track, motion_range

MIN_FRAMES = 4          # below this no law can be fit
# Fraction of frame-pairs allowed to reverse direction. IID positional noise
# (a tracker that lost its point) produces direction churn ≈ 0.67-0.72 —
# within sampling noise of the old 0.65 limit, so it could slip through.
# 0.55 keeps every physical motion family (churn ≈ 0 for smooth tracks)
# while rejecting lost-point noise with margin. KNOWN BLIND SPOT: a tracker
# drifting as a random WALK (IID velocity increments) has churn ≈ 0.5 and
# still certifies — catching drift needs a smoothness/autocorrelation check,
# which is future work; this gate is honest only about churn.
_JITTER_LIMIT = 0.55
_AGREEMENT_LIMIT = 0.5  # cross-tracker mean L2 / motion range
# When BOTH tracks are static, motion range is no scale at all (it collapses
# to MIN_RANGE and a constant seed offset between two healthy trackers reads
# as huge disagreement). Two static tracks agree if they sit within this
# absolute distance of each other (normalized screen units).
STATIC_TOLERANCE = 0.05


@dataclass
class TrackCertificate:
    certified: bool
    confidence: float        # 0-1
    reason: str = "ok"


def certify(track: Track, fps: int = 8) -> TrackCertificate:
    """Single-track self-diagnosis. Conservative: when in doubt, decertify —
    a silent verifier is honest, a confidently wrong one poisons the loop."""
    if not track or len(track) < MIN_FRAMES:
        return TrackCertificate(False, 0.0, "too_short")
    arr = np.asarray(track, dtype=float)
    if not np.all(np.isfinite(arr)):
        return TrackCertificate(False, 0.0, "non_finite")
    if arr.min() < -0.25 or arr.max() > 1.25:
        return TrackCertificate(False, 0.1, "out_of_frame")
    if motion_range(track) < MIN_RANGE:
        # static is trackable and trivially certifiable
        return TrackCertificate(True, 1.0, "static")

    vel = np.diff(arr, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 1e-6
    if moving.sum() >= 2:
        v = vel[moving]
        # direction churn: cosine between consecutive moving steps
        cos = np.sum(v[1:] * v[:-1], axis=1) / (
            np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[:-1], axis=1) + 1e-12
        )
        churn = float((cos < 0.0).mean())
        if churn > _JITTER_LIMIT:
            return TrackCertificate(False, round(1.0 - churn, 3), "jitter")
        return TrackCertificate(True, round(1.0 - churn, 3), "ok")
    return TrackCertificate(True, 0.8, "barely_moving")


def cross_agreement(a: Track, b: Track) -> float:
    """Mean per-frame disagreement between two extractors' tracks of the same
    entity, normalized by motion range. 0 = identical, ≥1 = unrelated.

    Static-static special case: when NEITHER track moves there is no motion
    range to normalize by (dividing a constant seed offset by MIN_RANGE=1e-3
    would decertify two perfectly healthy trackers staring at the same still
    object). Two static tracks are instead compared absolutely: they agree
    when their mean offset is within STATIC_TOLERANCE."""
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    aa, bb = np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)
    offset = float(np.mean(np.linalg.norm(aa - bb, axis=1)))
    if motion_range(a) < MIN_RANGE and motion_range(b) < MIN_RANGE:
        return offset / STATIC_TOLERANCE
    rng = max(motion_range(a), motion_range(b), MIN_RANGE)
    return offset / rng


def certify_pair(a: Track, b: Track, fps: int = 8) -> TrackCertificate:
    """Two-extractor certification: both must self-certify AND agree.
    Disagreement between healthy trackers is evidence the OBJECT is unstable
    (morphing/flicker) — which is itself a physics-plausibility signal the
    caller may surface, but never a measured law verdict."""
    ca, cb = certify(a, fps), certify(b, fps)
    if not (ca.certified and cb.certified):
        # Select among the certs that actually FAILED — picking by raw
        # confidence alone could return a CERTIFIED cert on a rounding tie,
        # silently certifying the pair despite one tracker failing.
        failed = [c for c in (ca, cb) if not c.certified]
        return min(failed, key=lambda c: c.confidence)
    agreement = cross_agreement(a, b)
    if agreement > _AGREEMENT_LIMIT:
        return TrackCertificate(False, round(max(0.0, 1.0 - agreement), 3),
                                "tracker_disagreement")
    return TrackCertificate(True, round(min(ca.confidence, cb.confidence,
                                            1.0 - agreement), 3), "ok")
