"""Real video-generation backends (v0.2 skeletons).

These implement the SAME `BaseVideoGenClient.generate(...)` contract as the mock,
so the rest of the data flow is unchanged — you only fill in the TODO bodies and
flip `models.video_gen.name` in the config.

────────────────────────────────────────────────────────────────────────────
References (existing methods we build on) vs. what is OUR innovation
────────────────────────────────────────────────────────────────────────────
Borrowed / built on (cite these in the paper):
  • OmniWeaving (Tencent-Hunyuan) — free-form text+multi-image+video conditioning;
    ideal for "neighbor frame as anchor + character reference".
  • Wan2.x — open-weight T2V/I2V with first/last-frame & ControlNet-style control.
  • Veo / Sora — API T2V/I2V fallback (no local weights).
  • Conditioning-as-control lineage: ControlNet, MotionCtrl, DragAnything, Tora
    (trajectory/drag control for video).

OUR innovation (highlight these):
  • C1 "physics sketch → control signal": we drive the conditioning input from a
    lightweight PHYSICS SIMULATION (gravity/collision/fluid trajectories), not from
    a user-drawn drag or a reference clip. No prior agentic video system conditions
    the neural generator on a physics proxy while keeping photorealistic rendering.
  • The control contract is model-agnostic (physics.control_render.ControlSpec), so
    ONE physics sketch conditions OmniWeaving / Wan / API models unchanged.
  • Conditioning is produced inside a self-improving loop: failing keyframes are
    re-conditioned locally (C2), not the whole clip (contrast VISTA's whole-segment
    regeneration, arXiv:2510.15831).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..physics.control_render import ControlSpec, load_control_spec
from .video_gen import BaseVideoGenClient


def _compose_prompt(prompt: str, spec: Optional[ControlSpec]) -> str:
    """Fold physics interaction hints into the text prompt."""
    if spec and spec.interaction_hints:
        return prompt + " | physics: " + "; ".join(spec.interaction_hints)
    return prompt


def build_conditioning(
    prompt: str,
    control_signal: Optional[Path],
    first_frame: Optional[Path],
    reference_images: Optional[list[Path]],
) -> dict:
    """Shared C1/C2/E1 wiring point for all real backends.

    control_signal → physics ControlSpec (C1); first_frame → I2V anchor (C2);
    reference_images → identity/style anchors (E1). Each backend maps this bundle
    onto its own conditioning API.
    """
    spec = load_control_spec(control_signal)
    return {
        "prompt": _compose_prompt(prompt, spec),
        "control": spec,
        "first_frame": first_frame,
        "reference_images": reference_images or [],
    }


# ─────────────────────────────────────────────────────────────
# OmniWeaving — preferred local backend (multi-condition native)
# ─────────────────────────────────────────────────────────────
class OmniWeavingClient(BaseVideoGenClient):
    """Tencent-Hunyuan/OmniWeaving. Native text + multi-image + video conditioning.

    Why preferred: our C1 control + E1 identity anchors map cleanly onto its
    free-form composition inputs (first-frame anchor + reference images + text).
    """

    def __init__(self, name: str = "omniweaving", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.weights_path = self.config.get("weights_path") or os.getenv("OMNIWEAVING_WEIGHTS")
        self.device = self.config.get("device", "cuda")
        self._pipe = None  # lazy-loaded

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        # TODO(v0.2): load the real pipeline, e.g.
        #   import torch; from omniweaving import OmniWeavingPipeline
        #   self._pipe = OmniWeavingPipeline.from_pretrained(self.weights_path).to(self.device)
        raise RuntimeError(
            "OmniWeavingClient not wired yet. Set models.video_gen.weights_path "
            "(or $OMNIWEAVING_WEIGHTS) and implement _ensure_loaded()/generate(). "
            "Until then keep models.video_gen.name = 'mock-video-gen'."
        )

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        control_signal: Optional[Path] = None,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        self._ensure_loaded()
        inputs = build_conditioning(prompt, control_signal, first_frame, reference_images)
        # TODO(v0.2): run the pipeline with `inputs` and write `out_path`, e.g.
        #   frames = self._pipe(prompt=inputs["prompt"], image=inputs["first_frame"],
        #                       ref_images=inputs["reference_images"],
        #                       control=inputs["control"], num_frames=int(duration*fps),
        #                       seed=seed)
        #   write_video(out_path, frames, fps)
        raise NotImplementedError(f"OmniWeaving generate() not wired; inputs={list(inputs)}")

    def supported_conditions(self) -> set[str]:
        return {"control_signal", "first_frame", "reference_images"}


# ─────────────────────────────────────────────────────────────
# Wan — open-weight backup
# ─────────────────────────────────────────────────────────────
class WanClient(BaseVideoGenClient):
    """Wan2.x local I2V/T2V with first/last-frame + control. Backup to OmniWeaving."""

    def __init__(self, name: str = "wan", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.weights_path = self.config.get("weights_path") or os.getenv("WAN_WEIGHTS")
        self.device = self.config.get("device", "cuda")
        self._pipe = None

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        control_signal: Optional[Path] = None,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        inputs = build_conditioning(prompt, control_signal, first_frame, reference_images)
        # TODO(v0.2): load Wan pipeline; use inputs["first_frame"] as I2V anchor and
        # inputs["control"] as motion control; write `out_path`.
        raise RuntimeError(
            "WanClient not wired yet. Set models.video_gen.weights_path (or "
            f"$WAN_WEIGHTS) and implement generate(); inputs={list(inputs)}"
        )

    def supported_conditions(self) -> set[str]:
        return {"control_signal", "first_frame", "reference_images"}


# ─────────────────────────────────────────────────────────────
# API fallback — Veo / Sora (no local weights, weaker conditioning)
# ─────────────────────────────────────────────────────────────
class VeoApiClient(BaseVideoGenClient):
    """Hosted API fallback. NOTE: most APIs accept text (+ optional first image)
    only, so the physics control signal degrades to a prompt hint — keep this for
    smoke tests / when no GPU is available, not for the physics-grounding claims.
    """

    def __init__(self, name: str = "veo-api", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("VEO_API_KEY")

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        control_signal: Optional[Path] = None,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        if not self.api_key:
            raise RuntimeError("VeoApiClient needs $VEO_API_KEY (or config api_key).")
        inputs = build_conditioning(prompt, control_signal, first_frame, reference_images)
        # TODO(v0.2): call the hosted endpoint with inputs["prompt"] (physics folded
        # in) + inputs["first_frame"], poll, download the result to out_path.
        raise NotImplementedError(f"Veo generate() not wired; inputs={list(inputs)}")

    def supported_conditions(self) -> set[str]:
        return {"first_frame"}  # control_signal only via prompt hint


_REGISTRY = {
    "omniweaving": OmniWeavingClient,
    "wan": WanClient,
    "veo": VeoApiClient,
    "veo-api": VeoApiClient,
    "sora": VeoApiClient,
}


def build_real_video_gen(name: str, config: Optional[dict] = None) -> BaseVideoGenClient:
    key = name.split("-")[0].lower() if name else ""
    cls = _REGISTRY.get(name.lower()) or _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown video_gen backend '{name}'. Known: {list(_REGISTRY)}")
    return cls(name=name, config=config)
