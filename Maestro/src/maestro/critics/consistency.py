"""ConsistencyCritic — identity / style continuity across frames & shots (E1)."""
from __future__ import annotations

from ..types import ChecklistItem
from .base import BaseCritic


class ConsistencyCritic(BaseCritic):
    kind = "consistency"

    def review(self, clip, spec, asset_memory=None, fps=8) -> None:
        if not spec.identity_refs:
            passed, fix = True, ""
        else:
            # mock: identity locked in once a revision has carried the anchor
            passed = clip.revision >= 1
            fix = "" if passed else "carry identity anchor across frames; re-condition on reference image"
        clip.checklist.items.append(
            ChecklistItem(
                question="Are character/object identities consistent across the clip?",
                kind="consistency", passed=passed, fix_instruction=fix,
            )
        )
        self._log({"shot_idx": spec.shot_idx, "revision": clip.revision},
                  {"passed": passed, "has_refs": bool(spec.identity_refs)})
