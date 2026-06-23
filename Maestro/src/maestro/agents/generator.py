"""GeneratorAgent — produce a CandidateClip, conditioned on the keyframe
anchor + identity/style references (C2 + E1). Physics is never injected; it
is verified afterwards from the generated pixels (C6 v0.4)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.video_gen import BaseVideoGenClient, MockVideoGenClient
from ..types import CandidateClip, Checklist, ShotSpec
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

        ref_images = reference_images  # identity/style anchors from RetrievalTool (E1)

        out_path = cache_dir / f"shot{spec.shot_idx:03d}_r{revision}_s{seed}.mp4"

        # Phase-2 capability dispatch: route on the capability chosen for this
        # shot (CapabilityRouter set spec.gen_capability/gen_params upstream)
        # against what THIS backend actually offers. flf2v/edit live on
        # optional methods (hasattr) and are declared in capabilities(); if the
        # backend lacks the requested capability we fall back to generate() and
        # LOG the downgrade — never a silent capability claim. t2v/i2v always
        # go through generate() (i2v auto-swaps on first_frame inside the backend).
        capability = spec.gen_capability or "t2v"
        gen_params = spec.gen_params or {}
        caps = self.video_gen.capabilities()
        used_capability = capability

        if (capability == "flf2v"
                and "flf2v" in caps
                and hasattr(self.video_gen, "frame_to_frame")):
            video_path = self.video_gen.frame_to_frame(
                prompt=prompt,
                first_frame=Path(gen_params["first_frame"]),
                last_frame=Path(gen_params["last_frame"]),
                out_path=out_path,
                duration=max(1, int(round(spec.duration))),
                seed=seed,
            )
        elif (capability == "edit"
                and "edit" in caps
                and hasattr(self.video_gen, "edit_video")):
            video_path = self.video_gen.edit_video(
                prompt=prompt,
                video_path=Path(gen_params["source_video"]),
                out_path=out_path,
                backend=gen_params.get("backend", "runway"),
                task=gen_params.get("task", "depth"),
                seed=seed,
            )
        else:
            if capability not in ("t2v", "i2v"):
                self._log(
                    "capability_downgrade",
                    {"shot_idx": spec.shot_idx, "wanted": capability,
                     "backend_caps": sorted(caps)},
                    {"used": "generate"},
                )
                used_capability = "i2v" if first_frame is not None else "t2v"
            video_path = self.video_gen.generate(
                prompt=prompt,
                duration=spec.duration,
                out_path=out_path,
                fps=fps,
                first_frame=first_frame,
                reference_images=ref_images,
                seed=seed,
            )

        keyframes = []
        for k in range(n_keyframes):
            # Seed is part of the name: same-revision candidates (different
            # seeds) must not overwrite each other's keyframes.
            kf = cache_dir / f"shot{spec.shot_idx:03d}_r{revision}_s{seed}_kf{k}.txt"
            # Record the capability used so the mock signal + trajectory stay
            # honest/inspectable (the clip body carries which model path ran).
            kf.write_text(
                f"keyframe {k} of {video_path.name}\nprompt={prompt}\n"
                f"capability={used_capability}\n",
                encoding="utf-8",
            )
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
             "anchored_on_first_frame": first_frame is not None,
             "capability": used_capability},
            {"video_path": str(video_path)},
        )
        return clip
