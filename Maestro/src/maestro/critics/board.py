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
        clip.metric_scores = self.metric_tool.run(clip, spec, asset_memory, fps)
        return clip

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
