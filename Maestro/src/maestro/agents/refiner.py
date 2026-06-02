"""RefinerAgent — translate 'localizable' defects into 'executable' fixes (C2).

Physics verdicts already carry a frame range + suggested intervention; checklist
items carry a fix instruction. Refiner aggregates them into:
  - extra_prompt: appended to the regeneration prompt
  - a keyframe to locally edit (image-edit) + its instruction
so we do keyframe-level local repair, NOT VISTA-style whole-segment regeneration.
"""
from __future__ import annotations

from typing import Optional

from ..types import CandidateClip
from .base import BaseAgent


class RefinerAgent(BaseAgent):
    def run(self, clip: CandidateClip) -> dict:
        return self.plan(clip)

    def plan(self, clip: CandidateClip) -> dict:
        fixes: list[str] = []
        # physics first (worst severity first)
        verdicts = sorted(clip.physics_verdicts, key=lambda v: v.severity, reverse=True)
        edit_kf_idx: Optional[int] = None
        edit_instruction = ""
        for v in verdicts:
            fixes.append(f"[{v.mode.value}] {v.suggested_intervention}")
        # semantic / other failed checklist items
        for item in clip.checklist.failed_items:
            if item.fix_instruction:
                fixes.append(f"[{item.kind}] {item.fix_instruction}")

        # choose a keyframe to locally edit: the frame at the worst verdict's start
        if verdicts and clip.keyframes:
            start = verdicts[0].frame_range[0]
            edit_kf_idx = min(start, len(clip.keyframes) - 1)
            edit_instruction = verdicts[0].suggested_intervention
        elif clip.checklist.failed_items and clip.keyframes:
            edit_kf_idx = 0
            edit_instruction = clip.checklist.failed_items[0].fix_instruction

        plan = {
            "extra_prompt": " | ".join(fixes),
            "edit_keyframe_idx": edit_kf_idx,
            "edit_instruction": edit_instruction,
        }
        self._log("plan_fix", {"shot_idx": clip.shot_idx, "revision": clip.revision},
                  {"n_fixes": len(fixes), "edit_keyframe_idx": edit_kf_idx})
        return plan
