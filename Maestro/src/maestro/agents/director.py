"""DirectorAgent — outline -> list[ShotSpec], with C4 lesson injection."""
from __future__ import annotations

from typing import Optional

from ..memory.lesson_library import LessonLibrary
from ..planning.event_graph import build_event_graph
from ..types import AssetMemory, CinematographyTags, ShotSpec
from .base import BaseAgent


class DirectorAgent(BaseAgent):
    def run(
        self,
        outline: list[str],
        asset_memory: AssetMemory,
        lesson_library: Optional[LessonLibrary] = None,
    ) -> list[ShotSpec]:
        default_dur = float(self.config.get("shot_duration", 3.0))
        identities = list(asset_memory.identity_anchors.keys())
        styles = [s.style_id for s in asset_memory.style_anchors]
        pacing = self._pacing(asset_memory)

        specs: list[ShotSpec] = []
        for i, scene in enumerate(outline):
            lessons: list[str] = []
            if lesson_library is not None:
                lessons = [l.fix for l in lesson_library.retrieve(scene)]
            spec = ShotSpec(
                shot_idx=i,
                duration=default_dur,
                prompt=scene,
                cinematography=self._pick_cinematography(scene),
                identity_refs=identities,
                style_refs=styles,
                rhythmic_pacing=pacing,
                injected_lessons=lessons,
            )
            spec.event_graph = build_event_graph(spec)  # GEST-style IR
            specs.append(spec)
        self.llm.complete(self.prompt_template)
        self._log(
            "expand_shotspecs",
            {"n": len(outline)},
            {"specs": [s.shot_idx for s in specs],
             "lessons_injected": sum(len(s.injected_lessons) for s in specs)},
        )
        return specs

    def revise(self, spec: ShotSpec, asset_memory: AssetMemory, issues: list[str]) -> ShotSpec:
        """Correct a flagged ShotSpec (FilmAgent Critique-Correct-Verify, 'Correct').

        v0.1: drop ungroundable references and rebuild the event graph so the spec
        becomes valid. v0.2: an LLM revises the prompt/refs from the feedback text.
        """
        spec.identity_refs = [
            r for r in spec.identity_refs if r in asset_memory.identity_anchors
        ]
        known_styles = {s.style_id for s in asset_memory.style_anchors}
        spec.style_refs = [s for s in spec.style_refs if s in known_styles]
        spec.event_graph = build_event_graph(spec)
        self._log("revise_shotspec", {"shot_idx": spec.shot_idx, "issues": issues},
                  {"identity_refs": spec.identity_refs})
        return spec

    @staticmethod
    def _pacing(asset_memory: AssetMemory) -> list[int]:
        mp = asset_memory.music_profile
        if mp and mp.beats:
            return [2, 2, 2, 2]  # 4 cuts of 2 beats (placeholder, music-aware)
        return []

    @staticmethod
    def _pick_cinematography(scene: str) -> CinematographyTags:
        low = scene.lower()
        movement = "static"
        if any(k in low for k in ("run", "chase", "fast", "跑", "追")):
            movement = "tracking"
        elif any(k in low for k in ("reveal", "wide", "landscape", "全景")):
            movement = "pan"
        scale = "medium"
        if any(k in low for k in ("face", "eye", "close", "特写")):
            scale = "close_up"
        elif any(k in low for k in ("city", "landscape", "wide", "远景")):
            scale = "long"
        return CinematographyTags(shot_scale=scale, shot_movement=movement)
