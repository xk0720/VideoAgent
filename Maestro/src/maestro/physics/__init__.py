"""Physics-from-pixels verification (C6, v0.4).

The sketch/simulator line is GONE (a frozen video model cannot be controlled
by a synthetic sketch, and comparing against one simulated rollout presumes
unknowable parameters). What replaced it, all training-free:

  annotate.py    — WHICH entities, what MOTION CLASS, which failure modes
                   (verification seeds — never trajectories or control)
  router.py      — which verification TIER can actually check each entity
                   (measurement / world_model / vlm / none), reported openly
  tracks.py      — observed tracks from the generated pixels (mock now,
                   CoTracker/TAPIR in track_extractor_backends.py)
  reliability.py — certify tracks before trusting them (trackers lie on
                   generated video; disagreement is itself a signal)
  laws.py        — "is there ANY physically consistent explanation?" —
                   best-law residual + anomaly localization, reference-free
  verifier.py    — the assembled stack feeding critics/physics_consistency.py
"""
from .annotate import annotate_physics
from .failure_modes import (
    FAILURE_MODE_KEYWORDS,
    INTERVENTION_LIBRARY,
    detect_expected_modes,
    suggest_intervention,
)
from .laws import LawFit, LawReport, MotionAnomaly, analyze_track, detect_anomalies, fit_best_law
from .reliability import TrackCertificate, certify, certify_pair, cross_agreement
from .router import RouteDecision, coverage_summary, route
from .tracks import BaseTrackExtractor, MockTrackExtractor, build_track_extractor
from .verifier import EntityVerification, PhysicsFromPixelsVerifier, VerificationResult

__all__ = [
    "FAILURE_MODE_KEYWORDS",
    "INTERVENTION_LIBRARY",
    "detect_expected_modes",
    "suggest_intervention",
    "annotate_physics",
    "LawFit",
    "LawReport",
    "MotionAnomaly",
    "analyze_track",
    "detect_anomalies",
    "fit_best_law",
    "TrackCertificate",
    "certify",
    "certify_pair",
    "cross_agreement",
    "RouteDecision",
    "coverage_summary",
    "route",
    "BaseTrackExtractor",
    "MockTrackExtractor",
    "build_track_extractor",
    "EntityVerification",
    "PhysicsFromPixelsVerifier",
    "VerificationResult",
]
