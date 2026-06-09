"""Simulator wrapper (C1 sketch layer) — the physics ORACLE's ground truth.

The simulator produces the *expected* motion that `physics/oracle.py` later
compares the generated video against (sketch-as-verifier, see
PHYSICS_LITERATURE_REVIEW.md). So the richer and more correct this trajectory
is, the sharper the oracle's verdict.

v0.1 shipped a pure projectile integrator that IGNORED interactions — a ball
"bouncing off a wall" never actually bounced; collision/support were metadata
only. v0.3 `MockSimulator` is a real (if minimal) semi-implicit Euler integrator
with:
  • a ground plane at y=0 with restitution (objects bounce, don't fall through);
  • a vertical wall ahead of horizontally-moving objects when a `collision`
    interaction is present (lateral bounce);
  • `support` constraints (object rests on the ground, no gravity fall-through);
  • recorded CONTACT EVENTS (frame + entity + kind) so the critic/Refiner can
    localize repairs to the exact bounce frame.

Training-free either way: the simulator is a *tool*. v0.4 swaps in
MuJoCo / NVIDIA Newton behind the same `BaseSimulator.simulate` — the oracle
math and the pipeline do not change.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..types import PhysEntity, PhysInteraction


class BaseSimulator(ABC):
    @abstractmethod
    def simulate(
        self,
        entities: list[PhysEntity],
        interactions: list[PhysInteraction],
        duration: float,
        fps: int,
        out_path: Path,
    ) -> Optional[Path]:
        """Produce a control-signal artifact, return its path (or None)."""
        ...

    @abstractmethod
    def supported_signals(self) -> set[str]:
        ...


class MockSimulator(BaseSimulator):
    """Analytic rigid-body integrator with ground + wall collision response.

    Coordinates: y is height above the ground plane (y=0 is the floor; objects
    stay at y>=0). x is horizontal screen-depth; gravity acts on y only.
    """

    GRAVITY = -9.81
    GROUND_Y = 0.0
    WALL_X = 1.0          # a wall this far ahead of a moving object (when collision)

    def simulate(
        self,
        entities: list[PhysEntity],
        interactions: list[PhysInteraction],
        duration: float,
        fps: int,
        out_path: Path,
    ) -> Optional[Path]:
        n_frames = max(1, int(round(duration * fps)))
        dt = 1.0 / fps

        # Which entities participate in a collision / support interaction.
        collide_names: set[str] = set()
        support_names: set[str] = set()
        for it in interactions:
            if it.kind == "collision":
                collide_names.update(it.entities)
            elif it.kind == "support":
                support_names.update(it.entities)

        tracks: dict[str, list[list[float]]] = {}
        events: list[dict] = []

        # A rebound slower than what gravity adds back in one frame can't leave
        # the surface for even one step — snapping it to rest is both physically
        # motivated and avoids fixed-timestep contact jitter.
        rest_speed = abs(self.GRAVITY) * dt

        for ent in entities:
            x, y, z = ent.init_position
            vx, vy, vz = ent.init_velocity
            has_gravity = "gravity" in ent.forces
            supported = ent.name in support_names
            e = max(0.0, min(1.0, ent.restitution))
            resting = supported           # supported objects start at rest
            # One-sided wall ahead of a collider moving in x (the object hits it
            # from one side and rebounds away; it does NOT re-collide afterwards).
            wall_x: Optional[float] = None
            wall_dir = 0                  # +1: wall at larger x; -1: smaller x
            if ent.name in collide_names and abs(vx) > 1e-6:
                wall_dir = 1 if vx > 0 else -1
                wall_x = x + wall_dir * self.WALL_X

            pos: list[list[float]] = []
            for f in range(n_frames):
                pos.append([round(x, 4), round(y, 4), round(z, 4)])

                # Semi-implicit Euler: update velocity, then position.
                if has_gravity and not resting:
                    vy += self.GRAVITY * dt
                x += vx * dt
                y += vy * dt
                z += vz * dt

                # Ground-plane collision / resting.
                if y < self.GROUND_Y or resting:
                    y = self.GROUND_Y
                    if vy < 0:
                        if -vy * e < rest_speed:
                            vy = 0.0                  # too slow to rebound -> rest
                            resting = True
                        else:
                            vy = -vy * e              # bounce
                            events.append({"frame": f + 1, "entity": ent.name,
                                           "kind": "ground_bounce"})

                # One-sided wall collision (lateral bounce, fires at most when the
                # object first reaches the wall while still moving toward it).
                if wall_x is not None and wall_dir * vx > 0:
                    reached = (wall_dir > 0 and x >= wall_x) or \
                              (wall_dir < 0 and x <= wall_x)
                    if reached:
                        x = wall_x
                        vx = -vx * e
                        events.append({"frame": f + 1, "entity": ent.name,
                                       "kind": "wall_bounce"})

            tracks[ent.name] = pos

        signal = {
            "type": "trajectory",
            "fps": fps,
            "n_frames": n_frames,
            "tracks": tracks,
            "interactions": [
                {"kind": it.kind, "entities": it.entities} for it in interactions
            ],
            "events": events,                 # contact frames for localization
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(signal, ensure_ascii=False), encoding="utf-8")
        return out_path

    def supported_signals(self) -> set[str]:
        return {"trajectory"}
