"""SemanticCritic — prompt/attribute alignment checklist (M3 Checker idea)."""
from __future__ import annotations

from typing import Optional

from ..models.mllm import BaseMLLMClient, MockMLLMClient
from ..types import ChecklistItem
from .base import BaseCritic


class SemanticCritic(BaseCritic):
    kind = "semantic"

    def __init__(self, mllm: Optional[BaseMLLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self.mllm = mllm or MockMLLMClient()

    def review(self, clip, spec, asset_memory=None, fps=8) -> None:
        results = self.mllm.assess_semantic(clip, spec)
        # Items are (question, passed, fix) or (question, passed, fix,
        # frame_range) — real VLMs localize failures (Q3); mocks honestly
        # give no range (they have no pixel evidence to localize with).
        for item in results:
            question, passed, fix = item[0], item[1], item[2]
            frame_range = item[3] if len(item) > 3 else None
            clip.checklist.items.append(
                ChecklistItem(question=question, kind="semantic",
                              passed=passed, fix_instruction=fix,
                              frame_range=frame_range)
            )
        self._log(
            {"shot_idx": spec.shot_idx, "revision": clip.revision},
            {"n_items": len(results),
             "failed": sum(1 for it in results if not it[1]),
             "localized": sum(1 for it in results
                              if len(it) > 3 and it[3] is not None)},
        )
