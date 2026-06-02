"""PhysicsCritic (C1 critic layer) — the differentiation core.

Does NOT emit one blurry score. Uses a (mock) layered VLM to produce per-failure-
mode, frame-localized verdicts, then turns each into a failed checklist item whose
fix_instruction is a concrete, executable intervention (空白④: localizable->actionable).
"""
from __future__ import annotations

from typing import Optional

from ..models.mllm import BaseMLLMClient, MockMLLMClient
from ..types import ChecklistItem
from .base import BaseCritic


class PhysicsCritic(BaseCritic):
    kind = "physics"

    def __init__(self, mllm: Optional[BaseMLLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self.mllm = mllm or MockMLLMClient()

    def review(self, clip, spec, asset_memory=None, fps=8) -> None:
        verdicts = self.mllm.assess_physics(clip, spec, fps)
        clip.physics_verdicts = verdicts
        for v in verdicts:
            clip.checklist.items.append(
                ChecklistItem(
                    question=(
                        f"Is '{v.mode.value}' physically plausible in frames "
                        f"{v.frame_range[0]}-{v.frame_range[1]}?"
                    ),
                    kind="physics",
                    passed=False,  # a verdict only exists when it's a violation
                    fix_instruction=v.suggested_intervention,
                )
            )
        self._log(
            {"shot_idx": spec.shot_idx, "revision": clip.revision},
            {"verdicts": [(v.mode.value, v.severity, list(v.frame_range)) for v in verdicts]},
        )
