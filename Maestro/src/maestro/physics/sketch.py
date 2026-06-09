"""Build a PhysicsSketch from a ShotSpec (C1 sketch layer / E2).

v0.1 parses entities heuristically from the prompt and runs the MockSimulator to
emit a control signal. The sketch doubles as an *executable, inspectable* mid-level
representation (GEST-style) so we can visualize 'what the system thinks the physics
is' — see REPORT section E2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..types import (
    PhysEntity,
    PhysInteraction,
    PhysicsSketch,
    ShotSpec,
)
from .failure_modes import detect_expected_modes
from .sim_wrapper import BaseSimulator, MockSimulator

# crude noun cues -> a movable entity in the scene
_ENTITY_CUES = [
    "ball", "car", "person", "man", "woman", "dog", "cat", "bottle", "cup",
    "rock", "drone", "bird", "leaf", "box", "球", "车", "人", "瓶", "杯", "石", "鸟",
]


def _extract_entities(prompt: str) -> list[PhysEntity]:
    low = (prompt or "").lower()
    found = [cue for cue in _ENTITY_CUES if cue in low]
    if not found:
        found = ["subject"]
    # A "fall/drop" scenario needs an initial HEIGHT or there is nothing to
    # fall — the integrator otherwise starts the object already on the ground.
    falling = any(v in low for v in ("fall", "drop", "落", "掉"))
    ents = []
    for name in found[:4]:  # cap to keep sketch lightweight
        # naive initial velocity: motion verbs imply horizontal launch / lift
        vx = 2.0 if any(v in low for v in ("throw", "run", "fly", "抛", "跑", "飞")) else 0.0
        vy = 3.0 if any(v in low for v in ("jump", "throw", "leap", "跳", "抛")) else 0.0
        # start height: dropped objects begin aloft; thrown/jumping start at ground
        y0 = 3.0 if (falling and vy == 0.0) else 0.0
        ents.append(PhysEntity(
            name=name,
            init_velocity=(vx, vy, 0.0),
            init_position=(0.0, y0, 0.0),
        ))
    return ents


def _infer_interactions(prompt: str, entities: list[PhysEntity]) -> list[PhysInteraction]:
    low = (prompt or "").lower()
    names = [e.name for e in entities]
    inter: list[PhysInteraction] = []
    if any(k in low for k in ("hit", "crash", "collide", "bounce", "wall",
                              "撞", "碰", "弹", "墙")):
        inter.append(PhysInteraction(kind="collision", entities=names[:2]))
    if any(k in low for k in ("water", "pour", "splash", "水", "倒")):
        inter.append(PhysInteraction(kind="fluid", entities=names[:1]))
    if any(k in low for k in ("ground", "floor", "stand", "地", "站")):
        inter.append(PhysInteraction(kind="support", entities=names[:1]))
    return inter


def build_physics_sketch(
    spec: ShotSpec,
    cache_dir: Path,
    fps: int = 8,
    simulator: Optional[BaseSimulator] = None,
) -> PhysicsSketch:
    simulator = simulator or MockSimulator()
    entities = _extract_entities(spec.prompt)
    interactions = _infer_interactions(spec.prompt, entities)
    expected = detect_expected_modes(spec.prompt)

    out = Path(cache_dir) / f"sketch_shot{spec.shot_idx:03d}.json"
    control = simulator.simulate(entities, interactions, spec.duration, fps, out)

    return PhysicsSketch(
        entities=entities,
        interactions=interactions,
        control_signal=control,
        expected_modes=expected,
    )
