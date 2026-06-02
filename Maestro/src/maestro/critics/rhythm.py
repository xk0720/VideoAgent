"""RhythmCritic — beat-cut sync / energy correspondence with music (reuses m5/m6)."""
from __future__ import annotations

from ..types import ChecklistItem
from .base import BaseCritic


class RhythmCritic(BaseCritic):
    kind = "rhythm"

    def review(self, clip, spec, asset_memory=None, fps=8) -> None:
        has_music = bool(asset_memory and asset_memory.music_profile)
        if not has_music:
            passed, fix = True, ""  # nothing to sync to
        else:
            passed = bool(spec.rhythmic_pacing)
            fix = "" if passed else "align cut points to downbeats; set rhythmic_pacing"
        clip.checklist.items.append(
            ChecklistItem(
                question="Are cuts/pacing synced to the music?",
                kind="rhythm", passed=passed, fix_instruction=fix,
            )
        )
        self._log({"shot_idx": spec.shot_idx, "has_music": has_music},
                  {"passed": passed})
