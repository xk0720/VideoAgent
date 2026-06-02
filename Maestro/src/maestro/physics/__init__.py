"""Physics grounding (C1 / E2) — Maestro's differentiation core.

Two layers:
  - sketch layer:  lightweight physical representation + simulated control signal
                   that conditions the neural generator ("engine does physics,
                   diffusion does rendering").
  - critic layer:  per-failure-mode, frame-localized verdicts (in critics/physics.py).
"""
from .failure_modes import (
    FAILURE_MODE_KEYWORDS,
    INTERVENTION_LIBRARY,
    detect_expected_modes,
    suggest_intervention,
)
from .sketch import build_physics_sketch
from .sim_wrapper import BaseSimulator, MockSimulator
from .control_render import ControlSpec, load_control_spec

__all__ = [
    "FAILURE_MODE_KEYWORDS",
    "INTERVENTION_LIBRARY",
    "detect_expected_modes",
    "suggest_intervention",
    "build_physics_sketch",
    "BaseSimulator",
    "MockSimulator",
    "ControlSpec",
    "load_control_spec",
]
