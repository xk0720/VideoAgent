"""GeneratorAgent — produce a CandidateClip, conditioned on the physics sketch's
control signal + identity/style references (C1 + E1)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.video_gen import BaseVideoGenClient, MockVideoGenClient
from ..types import AssetMemory, CandidateClip, Checklist, ShotSpec
from .base import BaseAgent


class GeneratorAgent(BaseAgent):
    def __init__(self, *args, video_gen: Optional[BaseVideoGenClient] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_gen = video_gen or MockVideoGenClient()

    def run(
        self,
        spec: ShotSpec,
        cache_dir: Path,
        revision: int = 0,
        seed: int = 0,
        extra_prompt: str = "",
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        fps: int = 8,
        n_keyframes: int = 3,
    ) -> CandidateClip:
        cache_dir = Path(cache_dir)
        prompt = spec.prompt
        if spec.injected_lessons:
            prompt += " | constraints: " + "; ".join(spec.injected_lessons)
        if extra_prompt:
            prompt += " | fix: " + extra_prompt

        control = spec.physics_sketch.control_signal if spec.physics_sketch else None
        ref_images = reference_images  # identity/style anchors from RetrievalTool (E1)

        video_path = self.video_gen.generate(
            prompt=prompt,
            duration=spec.duration,
            out_path=cache_dir / f"shot{spec.shot_idx:03d}_r{revision}_s{seed}.mp4",
            fps=fps,
            control_signal=control,
            first_frame=first_frame,
            reference_images=ref_images,
            seed=seed,
        )

        keyframes = []
        for k in range(n_keyframes):
            kf = cache_dir / f"shot{spec.shot_idx:03d}_r{revision}_kf{k}.txt"
            kf.write_text(f"keyframe {k} of {video_path.name}\nprompt={prompt}\n",
                          encoding="utf-8")
            keyframes.append(kf)

        clip = CandidateClip(
            shot_idx=spec.shot_idx,
            video_path=video_path,
            keyframes=keyframes,
            revision=revision,
            checklist=Checklist(),
        )
        self._log(
            "generate",
            {"shot_idx": spec.shot_idx, "revision": revision, "seed": seed,
             "conditioned_on_control": control is not None},
            {"video_path": str(video_path)},
        )
        return clip
