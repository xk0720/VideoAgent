"""Track extraction — recover observed per-entity motion from generated clips.

This is the perception half of physics-from-pixels verification (C6 v0.4):
the OBSERVED track comes from a point tracker; the law checks (laws.py) then
ask whether that track has any physically consistent explanation. There is no
expected/simulated trajectory anywhere — the dead sketch line is gone.

v0.4 mock: a deterministic CPU stand-in whose synthesized tracks exercise the
law checks end-to-end (revision-0 clips contain a violation, refined clips
don't — so the self-improvement loop has a real signal path to demonstrate).
Real backends (CoTracker / TAPIR) live in track_extractor_backends.py behind
the same ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..types import CandidateClip, PhysEntity, ShotSpec
from .laws import Track


class BaseTrackExtractor(ABC):
    """Recover normalized screen-space tracks for the named entities."""

    @abstractmethod
    def extract(
        self, clip: CandidateClip, spec: ShotSpec, entities: list[PhysEntity], fps: int
    ) -> Optional[dict[str, Track]]:
        """Return observed tracks keyed by entity name, or None if the clip is
        unreadable / tracking failed (the verifier then stays silent)."""


class MockTrackExtractor(BaseTrackExtractor):
    """Deterministic CPU stand-in for CoTracker/TAPIR.

    Synthesis rules (mirroring what a tracker would plausibly observe):
      • static entities  → a fixed point;
      • revision 0       → a track with a MID-AIR REVERSAL (falls, rises with
        no contact, falls deeper) — physically inexplicable, so the law layer
        flags it;
      • revision ≥ 1     → a clean constant-acceleration arc — the refined
        generation obeys passive dynamics, so the verdict clears.
    """

    def extract(self, clip, spec, entities, fps):
        try:
            Path(clip.video_path).read_bytes()   # unreadable clip → silent
        except Exception:
            return None
        n = max(8, int(round(spec.duration * fps)))
        observed: dict[str, Track] = {}
        for i, ent in enumerate(entities):
            x0 = (i + 1) / (len(entities) + 1)
            if ent.motion_class == "static":
                observed[ent.name] = [(x0, 0.5)] * n
            elif clip.revision == 0:
                observed[ent.name] = _violating_fall(x0, n)
            else:
                observed[ent.name] = _clean_fall(x0, n)
        return observed


def _clean_fall(x0: float, n: int) -> Track:
    """Pure constant-acceleration drop with slight drift — law-consistent."""
    return [
        (x0 + 0.10 * t / (n - 1), 0.2 + 0.5 * (t / (n - 1)) ** 2)
        for t in range(n)
    ]


def _violating_fall(x0: float, n: int) -> Track:
    """Falls to y≈0.5, reverses upward MID-AIR, then falls deeper to y≈0.8.
    The reversal happens well above the track's deepest point, which is what
    laws.detect_anomalies recognizes as a gravity/inertia violation.

    A deterministic micro-oscillation rides on y: a real point tracker never
    returns perfectly piecewise-linear tracks, and a jerk-free synthetic
    track would (correctly) trip the jerk detector's absolute fallback for
    degenerate median-jerk tracks (laws.ABS_JERK). The oscillation gives the
    track a realistic nonzero jerk floor so the segment junctions are judged
    by the robust 8x-median gate — keeping the mid-air reversal the dominant
    (gravity/inertia) violation, as a tracker watching this motion would see.
    Amplitude scales 1/n so per-frame noise stays ~10% of segment speeds."""
    import math

    a, b = n // 3, n // 2
    track: Track = []
    for t in range(n):
        x = x0 + 0.10 * t / (n - 1)
        if t < a:                       # fall 0.2 → 0.5
            y = 0.2 + 0.3 * t / max(1, a - 1)
        elif t < b:                     # reverse mid-air 0.5 → 0.4
            y = 0.5 - 0.1 * (t - a + 1) / max(1, b - a)
        else:                           # fall again 0.4 → 0.8
            y = 0.4 + 0.4 * (t - b) / max(1, n - 1 - b)
        y += (0.07 / n) * math.sin(2.1 * t + 1.3)   # tracker-like noise floor
        track.append((x, y))
    return track


def build_track_extractor(spec: str | dict | None) -> BaseTrackExtractor:
    """Factory. None / 'mock-track' → MockTrackExtractor (CPU, no deps). A real
    name ('cotracker' / 'tapir') lazy-imports the heavy backend so importing
    this module never drags in torch.

    config:  models.track_extractor.name: "mock-track"   # real: "cotracker"
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
