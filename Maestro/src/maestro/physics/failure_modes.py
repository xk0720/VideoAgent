"""Physical failure-mode taxonomy + the 'localizable -> actionable' bridge (空白④).

The PhysicsCritic does NOT emit one blurry score. It classifies into these modes,
localizes to a frame range, and maps each to a concrete intervention. This module
holds the static knowledge: which prompt cues imply which modes, and which fix to
apply for each mode.
"""
from __future__ import annotations

from ..types import PhysFailureMode

# Prompt keyword cues -> physical phenomena we should watch for.
FAILURE_MODE_KEYWORDS: dict[PhysFailureMode, list[str]] = {
    PhysFailureMode.GRAVITY_INERTIA: [
        "fall", "drop", "jump", "throw", "fly", "float", "leap", "落", "跳", "抛", "飞",
    ],
    PhysFailureMode.COLLISION: [
        "hit", "crash", "collide", "bounce", "kick", "punch", "撞", "碰", "踢", "弹",
    ],
    PhysFailureMode.FLUID: [
        "water", "pour", "splash", "rain", "wave", "smoke", "fire", "liquid",
        "水", "倒", "溅", "雨", "浪", "烟", "火",
    ],
    PhysFailureMode.DEFORMATION: [
        "bend", "stretch", "squeeze", "break", "tear", "弯", "拉", "压", "碎", "撕",
    ],
    PhysFailureMode.OBJECT_PERMANENCE: [
        "behind", "occlude", "disappear", "reappear", "遮", "消失", "再现",
    ],
    PhysFailureMode.PENETRATION: [
        "through", "wall", "ground", "floor", "穿", "墙", "地面",
    ],
    PhysFailureMode.CONSERVATION: [
        "spin", "rotate", "swing", "momentum", "旋转", "摆", "动量",
    ],
}

# Each failure mode -> a concrete, executable intervention hint (drives RefinerAgent).
INTERVENTION_LIBRARY: dict[PhysFailureMode, str] = {
    PhysFailureMode.GRAVITY_INERTIA: (
        "re-condition keyframes on simulated parabolic trajectory; "
        "add 'consistent gravity, natural arc' to prompt; slow motion ~0.8x"
    ),
    PhysFailureMode.COLLISION: (
        "regenerate impact frames with collision-response control signal; "
        "add 'objects rebound on contact, no overlap'"
    ),
    PhysFailureMode.FLUID: (
        "enforce fluid-continuity control map; add 'continuous fluid volume, "
        "no teleporting droplets'; reduce per-frame motion magnitude"
    ),
    PhysFailureMode.DEFORMATION: (
        "constrain shape with rigidity prior; add 'rigid body keeps shape "
        "unless force applied'"
    ),
    PhysFailureMode.OBJECT_PERMANENCE: (
        "carry identity anchor across occlusion frames; add 'object persists "
        "behind occluder and reappears consistently'"
    ),
    PhysFailureMode.PENETRATION: (
        "add depth/occlusion control signal; add 'solid surfaces are "
        "impenetrable, no clipping through walls/ground'"
    ),
    PhysFailureMode.CONSERVATION: (
        "lock angular/linear momentum with trajectory control; add 'preserve "
        "momentum and mass, no sudden velocity jumps'"
    ),
}


def detect_expected_modes(text: str) -> list[PhysFailureMode]:
    """Cheap keyword scan -> which physical modes a prompt is likely to stress."""
    low = (text or "").lower()
    modes: list[PhysFailureMode] = []
    for mode, kws in FAILURE_MODE_KEYWORDS.items():
        if any(kw in low for kw in kws):
            modes.append(mode)
    return modes


def suggest_intervention(mode: PhysFailureMode) -> str:
    return INTERVENTION_LIBRARY.get(mode, "regenerate region with stronger physics prompt")
