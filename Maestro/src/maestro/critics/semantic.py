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
        for question, passed, fix in results:
            clip.checklist.items.append(
                ChecklistItem(question=question, kind="semantic",
                              passed=passed, fix_instruction=fix)
            )
        self._log(
            {"shot_idx": spec.shot_idx, "revision": clip.revision},
            {"n_items": len(results),
             "failed": sum(1 for _, p, _ in results if not p)},
        )
