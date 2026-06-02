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

    def replan(
        self, spec: ShotSpec, cache_dir: Path, fps: int = 8, strictness: float = 0.6
    ) -> ShotSpec:
        """HSI Tier-1: rebuild the sketch with tighter physics (slower velocities).

        Higher strictness < 1.0 scales initial velocities down so the simulated
        trajectory is gentler, which gives the conditional generator an easier
        target. Returns the same spec with `physics_sketch` swapped in place so
        callers' references stay valid.
        """
        new_sketch = build_physics_sketch(
            spec, Path(cache_dir), fps=fps, simulator=self.simulator
        )
        for ent in new_sketch.entities:
            vx, vy, vz = ent.init_velocity
            ent.init_velocity = (vx * strictness, vy * strictness, vz * strictness)
        # Re-simulate with the dampened velocities so control_signal matches.
        sig_path = Path(cache_dir) / f"sketch_shot{spec.shot_idx:03d}_strict.json"
        new_sketch.control_signal = self.simulator.simulate(
            new_sketch.entities, new_sketch.interactions, spec.duration, fps, sig_path
        )
        spec.physics_sketch = new_sketch
        self._log(
            "replan_sketch",
            {"shot_idx": spec.shot_idx, "strictness": strictness},
            {"control_signal": str(new_sketch.control_signal),
             "entities": [(e.name, e.init_velocity) for e in new_sketch.entities]},
        )
        return spec
