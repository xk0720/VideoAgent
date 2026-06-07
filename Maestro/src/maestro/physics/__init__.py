"""Physics grounding (C1, repositioned 2026-06 — see PHYSICS_LITERATURE_REVIEW.md).

Two layers:
  - sketch layer:  lightweight physical representation + simulated EXPECTED
                   trajectory. The sim is a VERIFICATION ORACLE (PISA/Morpheus
                   -style), NOT a conditioning signal for the generator; its
                   salient points also serve as keyframe-anchor hints (I2V).
  - critic layer:  per-failure-mode, frame-localized verdicts
                   (critics/physics.py = VLM judge; critics/physics_consistency
                   .py = oracle expected-vs-observed comparison).
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
from .oracle import (
    BaseTrackExtractor,
    MockTrackExtractor,
    TrajectoryComparison,
    TrajectoryOracle,
)

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
    "BaseTrackExtractor",
    "MockTrackExtractor",
    "TrajectoryComparison",
    "TrajectoryOracle",
]
