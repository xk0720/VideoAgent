"""ReviewBoard — runs all critics (the multi-agent review of C3) and computes
the metric suite. Critics run as an ensemble; their checklist items + physics
verdicts + metrics are what drive the self-improvement loop.
"""
from __future__ import annotations

from typing import Optional

from ..tools.metric_tool import MetricTool
from ..types import AssetMemory, CandidateClip, Checklist, ShotSpec
from .base import BaseCritic


class ReviewBoard:
    def __init__(self, critics: list[BaseCritic], metric_tool: Optional[MetricTool] = None):
        self.critics = critics
        self.metric_tool = metric_tool or MetricTool()

    def review(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> CandidateClip:
        # fresh review each round
        clip.checklist = Checklist()
        clip.physics_verdicts = []
        for critic in self.critics:
            critic.review(clip, spec, asset_memory, fps)
        self._key_physics_items(clip)
        clip.metric_scores = self.metric_tool.run(clip, spec, asset_memory, fps)
        return clip

    @staticmethod
    def _key_physics_items(clip: CandidateClip) -> None:
        """Key each physics ChecklistItem that MIRRORS a PhysicsVerdict to its
        verdict's failure mode (`item.mode = verdict.mode.value`).

        The critics that author the mirrored items are owned elsewhere, so the
        board does the keying post-hoc: a mirror is identified by an exact
        fix_instruction match (both critics copy `suggested_intervention` into
        it) or by the quoted mode value in the question text. The key lets the
        escape hatch flip the mirror in lock-step with its verdict and lets
        lesson/skill accounting subtract checklist-level skips by typed mode.
        """
        for verdict in clip.physics_verdicts:
            for item in clip.checklist.items:
                if item.kind != "physics" or item.passed or item.mode:
                    continue
                if (item.fix_instruction == verdict.suggested_intervention
                        or f"'{verdict.mode.value}'" in item.question):
                    item.mode = verdict.mode.value
                    break

    def all_passed(self, clip: CandidateClip) -> bool:
        return not clip.checklist.failed_items and not clip.physics_verdicts

    def recompute_metrics(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory] = None,
        fps: int = 8,
    ) -> CandidateClip:
        """Refresh metric scores in-place WITHOUT re-running critics.

        Used after the escape hatch mutates `physics_verdicts` / `checklist` so
        the Verifier's next monotonic check compares against an up-to-date total.
        Skipping the critic pass is intentional — re-running PhysicsCritic would
        regenerate the verdict we just escape-hatched.
        """
        clip.metric_scores = self.metric_tool.run(clip, spec, asset_memory, fps)
        return clip
