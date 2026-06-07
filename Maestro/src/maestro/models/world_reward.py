"""World-model reward (v0.3 slot) — the strongest validated training-free
physics signal.

Borrowed (cite, see PHYSICS_LITERATURE_REVIEW.md §4):
  • WMReward (arXiv:2601.10553, Meta FAIR): V-JEPA-2 latent world-model priors
    as an inference-time reward to search/steer denoising candidates —
    ICCV 2025 PhysicsIQ Challenge #1 (62.64%, +7.42%), and crucially found
    world-model rewards OUTPERFORM VLM-as-critic rewards.
  • VJEPA-2 Reward (arXiv:2510.21840): plain Best-of-N on V-JEPA-2 "surprise"
    → ~+6% on PhysicsIQ and VideoPhy, no training.

Role in Maestro: an additional, optional metric dimension (`wm_reward`) that
the Tournament/Verifier already consume through `weighted_total` — i.e. our
existing best-of-N + monotonic-improvement loop becomes WMReward-style
test-time search the moment a real backend is plugged in.

v0.2 ships a deterministic mock; v0.3 wires V-JEPA-2 (encode clip, score
physical predictability / negative surprise) behind the same `score()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import CandidateClip, ShotSpec


class BaseWorldReward(ABC):
    name: str = "world-reward"

    @abstractmethod
    def score(self, clip: CandidateClip, spec: ShotSpec, fps: int = 8) -> float:
        """Return a physics-plausibility reward in [0,1] (higher = better).
        Real impl: 1 - normalized V-JEPA-2 surprise over the clip."""


class MockWorldReward(BaseWorldReward):
    """Deterministic stand-in: reward improves as the clip is refined and is
    dented by outstanding physics verdicts — mirroring how a real world model
    scores clips whose motion is more predictable."""

    def __init__(self, name: str = "mock-world-reward"):
        self.name = name

    def score(self, clip: CandidateClip, spec: ShotSpec, fps: int = 8) -> float:
        base = min(1.0, 0.55 + 0.12 * clip.revision)
        worst = max((v.severity for v in clip.physics_verdicts), default=0.0)
        return round(max(0.0, base - 0.2 * worst), 3)


def build_world_reward(spec: str | dict | None) -> Optional[BaseWorldReward]:
    """Factory. None (unconfigured) keeps MetricTool output identical to v0.2.2.
    config:  models.world_reward.name: "mock-world-reward"  (v0.3: "vjepa2")
    """
    if spec is None:
        return None
    name = spec.get("name", "") if isinstance(spec, dict) else str(spec)
    if not name:
        return None
    if name.startswith("mock"):
        return MockWorldReward(name=name)
    # DESIGN_DECISION: v0.3 — `vjepa2` backend loads V-JEPA-2 and scores
    # negative surprise; until then, fail loudly rather than silently degrade.
    raise ValueError(
        f"world_reward backend '{name}' not wired yet; use 'mock-world-reward' "
        "or leave models.world_reward unset."
    )
