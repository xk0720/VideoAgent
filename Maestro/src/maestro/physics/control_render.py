"""Physics-sketch → ControlSpec parser (REPOSITIONED, see PHYSICS_LITERATURE_REVIEW.md).

v0.2 framing ("condition the frozen generator on the sim trajectory — engine
does physics, diffusion does rendering") is RETIRED: the literature review
(2026-06) found no work that validates abstract-sim-trajectory conditioning on
frozen general video models, and sparse trajectories under-determine physical
plausibility (review §2-3). Honest roles for the parsed `ControlSpec` now:

  1. ORACLE REFERENCE (primary): `tracks_2d` is the sim-EXPECTED motion that
     physics/oracle.py compares against motion OBSERVED in the generated clip
     (PISA-style trajectory-L2, arXiv:2503.09595) → PhysicsConsistencyCritic.
  2. KEYFRAME ANCHOR HINTS (secondary): the trajectory's salient points
     (apex / contact) tell the loop WHERE a physically plausible keyframe
     should be anchored — fed through I2V first/last-frame conditioning, the
     one conditioning channel frozen models reliably obey (review §2).
  3. PROMPT HINTS: `interaction_hints` are folded into the text prompt.

The dataclass / `load_control_spec` API is unchanged for back-compat.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ControlSpec:
    """Model-agnostic control description derived from the physics sketch.

    Backends map this onto their own conditioning API (ControlNet / drag / I2V
    first-frame / motion-vector). Keeping it abstract is what lets one physics
    sketch drive OmniWeaving, Wan, or an API model unchanged.
    """

    kind: str = "trajectory"          # trajectory | depth | flow | pose | none
    fps: int = 8
    n_frames: int = 0
    # per-entity normalized 2D screen tracks in [0,1] (x,y) for drag/motion control
    tracks_2d: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # interaction hints (collision/support/fluid) the generator prompt can stress
    interaction_hints: list[str] = field(default_factory=list)
    raw_path: Optional[Path] = None


def _project_to_screen(pos3: list[float]) -> tuple[float, float]:
    """Crude orthographic projection of a sim (x, y, z) point to screen (x, y) in
    [0,1]. y is flipped (screen y grows downward). v0.2: use a real camera model.
    """
    x, y, _z = pos3
    sx = 0.5 + 0.08 * x           # center + scale; clamp below
    sy = 0.5 - 0.08 * y
    return (min(1.0, max(0.0, sx)), min(1.0, max(0.0, sy)))


def load_control_spec(control_signal: Optional[Path]) -> Optional[ControlSpec]:
    """Parse a sim trajectory JSON into a ControlSpec. Returns None if absent."""
    if not control_signal:
        return None
    p = Path(control_signal)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    tracks_2d: dict[str, list[tuple[float, float]]] = {}
    for name, pts in data.get("tracks", {}).items():
        tracks_2d[name] = [_project_to_screen(pt) for pt in pts]
    hints = [
        f"{it.get('kind')}({','.join(it.get('entities', []))})"
        for it in data.get("interactions", [])
    ]
    return ControlSpec(
        kind=data.get("type", "trajectory"),
        fps=int(data.get("fps", 8)),
        n_frames=int(data.get("n_frames", 0)),
        tracks_2d=tracks_2d,
        interaction_hints=hints,
        raw_path=p,
    )
