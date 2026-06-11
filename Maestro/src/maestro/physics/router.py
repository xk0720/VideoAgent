"""VerifiabilityRouter (C6 / S3) — choose HOW each entity can be verified.

A measurement oracle only covers a thin slice of prompts (rigid/ballistic
motion that a tracker can follow). Pretending otherwise is how open-loop
physics methods die in review. The router makes coverage EXPLICIT: every
entity is assigned the strongest tier that can actually check it, and the
final report says which tier verified what — partial-verification
transparency instead of a silent coverage hole.

Tiers (strongest evidence first):
  measurement  — tracker + reference-free law residuals (physics/laws.py).
                 For ballistic / rigid motion.
  world_model  — learned surprise from a frozen video world model
                 (models/world_reward.py, V-JEPA-2-style; WMReward 2601.10553).
                 For fluids / deformables / agentive (biological) motion that
                 has no small parametric law family.
  vlm          — MLLM judge with trajectory-aware prompting (TRAVL
                 2510.07550 caveats apply). For semantic violations (object
                 count changes, impossible interactions) and as fallback.
  none         — static entities: nothing physical to verify.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..types import PhysicsAnnotation

Tier = Literal["measurement", "world_model", "vlm", "none"]

_CLASS_TO_TIER: dict[str, Tier] = {
    "ballistic": "measurement",
    "rigid": "measurement",
    "fluid": "world_model",
    "agentive": "world_model",
    "static": "none",
}


@dataclass
class RouteDecision:
    entity: str
    tier: Tier
    reason: str


def route(annotation: PhysicsAnnotation | None) -> list[RouteDecision]:
    """Map every annotated entity to its strongest feasible verification tier."""
    if annotation is None or not annotation.entities:
        return []
    fluid_parties = {
        name for it in annotation.interactions if it.kind == "fluid"
        for name in it.entities
    }
    decisions = []
    for ent in annotation.entities:
        tier = _CLASS_TO_TIER.get(ent.motion_class, "vlm")
        reason = f"motion_class={ent.motion_class}"
        if ent.name in fluid_parties and tier == "measurement":
            # an interaction can demote a trackable entity (splashing ball)
            tier, reason = "world_model", "fluid interaction"
        decisions.append(RouteDecision(entity=ent.name, tier=tier, reason=reason))
    return decisions


def coverage_summary(decisions: list[RouteDecision]) -> dict[str, list[str]]:
    """tier → entity names; logged so nobody mistakes partial coverage for
    full verification (the 'no silent caps' rule)."""
    out: dict[str, list[str]] = {}
    for d in decisions:
        out.setdefault(d.tier, []).append(d.entity)
    return out
