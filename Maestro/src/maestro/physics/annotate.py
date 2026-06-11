"""Build a PhysicsAnnotation from a ShotSpec (C6 v0.4).

The annotation answers three planning-time questions, all training-free:
  1. WHICH entities in the prompt are expected to move (tracker seeds);
  2. what CLASS of motion each implies (VerifiabilityRouter tier);
  3. which failure modes to watch (C7 skill signature + critic focus).

It deliberately does NOT contain trajectories, velocities, or control
signals — predicting the exact motion was the dead sketch-as-controller
line. v0.1-0.3 parsed entities with crude noun cues; a real deployment swaps
`annotate_physics` internals for an LLM call behind the same signature.
"""
from __future__ import annotations

from ..types import (
    MotionClass,
    PhysEntity,
    PhysicsAnnotation,
    PhysInteraction,
    ShotSpec,
)
from .failure_modes import detect_expected_modes

# crude noun cues -> a movable entity in the scene
_ENTITY_CUES = [
    "ball", "car", "person", "man", "woman", "dog", "cat", "bottle", "cup",
    "rock", "drone", "bird", "leaf", "box", "球", "车", "人", "瓶", "杯", "石", "鸟",
]

# entities that move by their own agency — no parametric law family applies
_AGENTIVE = {"person", "man", "woman", "dog", "cat", "bird", "人", "鸟"}

_BALLISTIC_VERBS = ("fall", "drop", "throw", "bounce", "leap", "jump",
                    "落", "掉", "抛", "弹", "跳")
_RIGID_VERBS = ("roll", "slide", "drive", "run", "fly", "滚", "滑", "跑", "飞", "开")
_FLUID_CUES = ("water", "pour", "splash", "rain", "smoke", "水", "倒", "雨", "烟")


def _motion_class(name: str, low_prompt: str) -> MotionClass:
    if name in _AGENTIVE:
        return "agentive"
    if any(v in low_prompt for v in _BALLISTIC_VERBS):
        return "ballistic"
    if any(v in low_prompt for v in _RIGID_VERBS):
        return "rigid"
    if any(v in low_prompt for v in _FLUID_CUES):
        return "fluid"
    return "static"


def _extract_entities(prompt: str) -> list[PhysEntity]:
    low = (prompt or "").lower()
    found = [cue for cue in _ENTITY_CUES if cue in low]
    if not found:
        found = ["subject"]
    return [
        PhysEntity(name=name, motion_class=_motion_class(name, low))
        for name in found[:4]   # cap to keep the annotation lightweight
    ]


def _infer_interactions(prompt: str, entities: list[PhysEntity]) -> list[PhysInteraction]:
    low = (prompt or "").lower()
    names = [e.name for e in entities]
    inter: list[PhysInteraction] = []
    if any(k in low for k in ("hit", "crash", "collide", "bounce", "wall",
                              "撞", "碰", "弹", "墙")):
        inter.append(PhysInteraction(kind="collision", entities=names[:2]))
    if any(k in low for k in _FLUID_CUES):
        inter.append(PhysInteraction(kind="fluid", entities=names[:1]))
    if any(k in low for k in ("ground", "floor", "stand", "地", "站")):
        inter.append(PhysInteraction(kind="support", entities=names[:1]))
    return inter


def annotate_physics(spec: ShotSpec, strictness: float = 1.0) -> PhysicsAnnotation:
    entities = _extract_entities(spec.prompt)
    return PhysicsAnnotation(
        entities=entities,
        interactions=_infer_interactions(spec.prompt, entities),
        expected_modes=detect_expected_modes(spec.prompt),
        strictness=strictness,
    )
