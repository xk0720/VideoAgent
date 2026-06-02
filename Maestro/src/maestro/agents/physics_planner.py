"""PhysicsPlannerAgent — attach a PhysicsSketch (C1 sketch layer) to each ShotSpec."""
from __future__ import annotations

from pathlib import Path

from ..physics.sketch import build_physics_sketch
from ..physics.sim_wrapper import BaseSimulator, MockSimulator
from ..types import ShotSpec
from .base import BaseAgent


class PhysicsPlannerAgent(BaseAgent):
    def __init__(self, *args, simulator: BaseSimulator | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = simulator or MockSimulator()

    def run(self, spec: ShotSpec, cache_dir: Path, fps: int = 8) -> ShotSpec:
        sketch = build_physics_sketch(spec, Path(cache_dir), fps=fps, simulator=self.simulator)
        spec.physics_sketch = sketch
        self._log(
            "build_sketch",
            {"shot_idx": spec.shot_idx, "prompt": spec.prompt},
            {
                "entities": [e.name for e in sketch.entities],
                "interactions": [it.kind for it in sketch.interactions],
                "expected_modes": [m.value for m in sketch.expected_modes],
                "control_signal": str(sketch.control_signal),
            },
        )
        return spec
