"""TrajectoryOracle — the simulator as a VERIFICATION ORACLE, not a controller.

Repositioned per PHYSICS_LITERATURE_REVIEW.md (2026-06):
  • Conditioning a frozen video model on an abstract sim trajectory is an
    unvalidated gap with structural flaws (trajectory under-determines physics;
    OOD synthetic tracks untested; see review §2-3).
  • What IS validated: compare expected motion (from simulation) against the
    motion OBSERVED in the generated video, then drive test-time search.
    Direct precedents: PISA (arXiv:2503.09595, Trajectory-L2 vs sim ground
    truth), Morpheus (2504.02918, tracked physical quantities vs conservation
    laws), PhyCoBench (2502.05503, flow deviation), Physics-IQ (2501.09038).

Pipeline:  sim expected tracks ─┐
                                ├─→ per-entity deviation + conservation check
  generated video ─ extractor ──┘        (feeds PhysicsConsistencyCritic)

The track extractor is an ABC: v0.2 ships a deterministic CPU mock; v0.3 swaps
in CoTracker/TAPIR point tracking or RAFT optical flow behind the same
interface — the oracle math does not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..types import CandidateClip, ShotSpec
from .control_render import load_control_spec

Track = list[tuple[float, float]]   # normalized screen-space (x, y) per frame


@dataclass
class TrajectoryComparison:
    """Result of expected-vs-observed comparison. Deviation is RELATIVE: the
    worst per-entity displacement error divided by the expected motion range,
    so 'ignored the physics entirely' ≈ 1.0 and 'followed it' ≈ 0.0
    regardless of how large the motion is on screen (PISA-style Trajectory-L2,
    normalized)."""

    deviation: float                          # [0,1]; 0 = follows expectation
    per_entity: dict[str, float] = field(default_factory=dict)
    worst_entity: Optional[str] = None
    expected_range: float = 0.0               # motion magnitude the sim predicts
    note: str = ""


class BaseTrackExtractor(ABC):
    """Recover per-entity screen-space tracks from a generated clip.
    v0.3: CoTracker / TAPIR (point tracking) or RAFT (flow integration)."""

    @abstractmethod
    def extract(
        self, clip: CandidateClip, spec: ShotSpec, expected: dict[str, Track], fps: int
    ) -> Optional[dict[str, Track]]:
        """Return observed tracks keyed like `expected`, or None if the clip
        is unreadable / tracking failed (oracle then stays silent)."""


class MockTrackExtractor(BaseTrackExtractor):
    """Deterministic CPU stand-in for CoTracker/RAFT.

    Semantics mirror what a real tracker would see in the mock pipeline:
      • If the generator metadata records that NO physics conditioning /
        anchoring reached it (`control_signal=None`), the observed motion is a
        flat drift (no gravity arc) — maximally divergent from the sim.
      • Otherwise the observed track equals the expectation plus an error that
        shrinks with each refinement revision (test-time improvement).
    """

    def extract(self, clip, spec, expected, fps):
        try:
            body = Path(clip.video_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        ignored = "control_signal=None" in body
        observed: dict[str, Track] = {}
        for name, track in expected.items():
            if not track:
                observed[name] = []
                continue
            if ignored:
                # flat drift: stays at the starting point (no physics followed)
                observed[name] = [track[0]] * len(track)
            else:
                err = max(0.0, 0.15 - 0.15 * clip.revision)
                rng = _motion_range(track)
                dx = err * rng  # error proportional to motion scale
                observed[name] = [(x + dx, y) for (x, y) in track]
        return observed


def build_track_extractor(spec: str | dict | None) -> BaseTrackExtractor:
    """Factory. None / 'mock-track' → deterministic MockTrackExtractor (default,
    CPU, no deps). A real name ('cotracker' / 'tapir') lazy-imports the heavy
    backend so importing this module never drags in torch.

    config:  models.track_extractor.name: "mock-track"   # v0.3: "cotracker"
    """
    name = "mock-track"
    config: dict = {}
    if isinstance(spec, dict):
        name = spec.get("name", name)
        config = spec
    elif isinstance(spec, str):
        name = spec
    if not name or name.startswith("mock"):
        return MockTrackExtractor()
    from .track_extractor_backends import build_real_track_extractor
    return build_real_track_extractor(name, config)


def _motion_range(track: Track) -> float:
    """Max displacement from the starting point — the scale of expected motion."""
    if not track:
        return 0.0
    x0, y0 = track[0]
    return max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for x, y in track)


def _max_l2(a: Track, b: Track) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return max(
        ((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2) ** 0.5 for i in range(n)
    )


class TrajectoryOracle:
    """Compare sim-expected motion with observed motion in the generated clip."""

    # below this expected motion range there is nothing physical to verify
    MIN_RANGE = 1e-3

    def __init__(self, extractor: Optional[BaseTrackExtractor] = None):
        self.extractor = extractor or MockTrackExtractor()

    def expected_tracks(self, spec: ShotSpec) -> Optional[dict[str, Track]]:
        if not spec.physics_sketch or not spec.physics_sketch.control_signal:
            return None
        cspec = load_control_spec(spec.physics_sketch.control_signal)
        if cspec is None or not cspec.tracks_2d:
            return None
        return cspec.tracks_2d

    def compare(
        self, clip: CandidateClip, spec: ShotSpec, fps: int = 8
    ) -> Optional[TrajectoryComparison]:
        """None = nothing to verify (no sketch / no motion / extraction failed)."""
        expected = self.expected_tracks(spec)
        if not expected:
            return None
        observed = self.extractor.extract(clip, spec, expected, fps)
        if observed is None:
            return None

        per_entity: dict[str, float] = {}
        worst_name, worst_dev, max_range = None, 0.0, 0.0
        for name, exp_track in expected.items():
            rng = _motion_range(exp_track)
            max_range = max(max_range, rng)
            if rng < self.MIN_RANGE:
                continue  # static expectation -> nothing to verify for this entity
            obs_track = observed.get(name, [])
            dev = min(1.0, _max_l2(exp_track, obs_track) / rng)
            per_entity[name] = round(dev, 3)
            if dev >= worst_dev:
                worst_dev, worst_name = dev, name

        if not per_entity:
            return None  # sim predicts no meaningful motion
        return TrajectoryComparison(
            deviation=round(worst_dev, 3),
            per_entity=per_entity,
            worst_entity=worst_name,
            expected_range=round(max_range, 4),
            note="mock extractor (v0.3: CoTracker/RAFT)",
        )
