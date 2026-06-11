"""PhysicsPlannerAgent — attach a PhysicsAnnotation to each ShotSpec (C6 v0.4).

The planner no longer simulates anything. It records what is physically at
stake in a shot (entities, motion classes, interactions, expected failure
modes) so the verification stack knows what to check and the skill library
knows the shot's physical signature. On an HSI tier-1 replan it tightens the
verification strictness and lets the caller regenerate with intervention
hints derived from the OBSERVED violations — repair guidance comes from
measured failures, not from a re-simulated trajectory.
"""
from __future__ import annotations

from pathlib import Path

from ..physics.annotate import annotate_physics
from ..types import ShotSpec
from .base import BaseAgent


class PhysicsPlannerAgent(BaseAgent):
    def run(self, spec: ShotSpec, cache_dir: Path | None = None, fps: int = 8) -> ShotSpec:
        annotation = annotate_physics(spec)
        spec.physics_annotation = annotation
        self._log(
            "annotate_physics",
            {"shot_idx": spec.shot_idx, "prompt": spec.prompt},
            {
                "entities": [(e.name, e.motion_class) for e in annotation.entities],
                "interactions": [it.kind for it in annotation.interactions],
                "expected_modes": [m.value for m in annotation.expected_modes],
            },
        )
        return spec

    def replan(
        self, spec: ShotSpec, cache_dir: Path | None = None, fps: int = 8,
        strictness: float = 0.6,
    ) -> ShotSpec:
        """HSI Tier-1: re-annotate with TIGHTER verification.

        `strictness` < 1.0 (legacy convention from the sim era) maps to a
        verification multiplier > 1.0: the law-residual threshold shrinks, so
        the regenerated candidate must be MORE physically consistent to pass.
        The spec's annotation is swapped in place so callers' references stay
        valid."""
        multiplier = 1.0 / max(0.1, strictness)
        spec.physics_annotation = annotate_physics(spec, strictness=multiplier)
        self._log(
            "replan_physics",
            {"shot_idx": spec.shot_idx, "strictness": multiplier},
            {"entities": [(e.name, e.motion_class)
                          for e in spec.physics_annotation.entities]},
        )
        return spec
