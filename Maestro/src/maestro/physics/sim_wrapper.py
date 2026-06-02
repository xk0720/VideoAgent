"""Simulator wrapper (C1 sketch layer).

v0.1: MockSimulator uses analytic geometric priors (projectile motion etc.) and
writes a tiny 'control signal' file. v0.2: swap for MuJoCo / NVIDIA Newton / a
particle sim that renders a real trajectory/depth/flow control map. Training-free
either way: the simulator is a *tool*, the base generative model is never trained.
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
    """Analytic projectile/inertia priors; emits a JSON trajectory as control signal."""

    GRAVITY = -9.81

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
        tracks = {}
        for ent in entities:
            vx, vy, vz = ent.init_velocity
            has_gravity = "gravity" in ent.forces
            pos = []
            x = y = z = 0.0
            cvy = vy
            for _ in range(n_frames):
                pos.append([round(x, 4), round(y, 4), round(z, 4)])
                x += vx * dt
                y += cvy * dt
                z += vz * dt
                if has_gravity:
                    cvy += self.GRAVITY * dt
            tracks[ent.name] = pos
        signal = {
            "type": "trajectory",
            "fps": fps,
            "n_frames": n_frames,
            "tracks": tracks,
            "interactions": [
                {"kind": it.kind, "entities": it.entities} for it in interactions
            ],
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(signal, ensure_ascii=False), encoding="utf-8")
        return out_path

    def supported_signals(self) -> set[str]:
        return {"trajectory"}
