"""Wan2.x local-inference wrapper (stub).

Open-source: https://github.com/Wan-Video. Weights distributed via Hugging Face.
v0.1 ships only the public surface; v0.2 will load the model with
`transformers` + `accelerate` and run T2V locally.

2024–2025 alternatives behind the same interface:
    • **Wan2.1 / Wan2.6** (Alibaba, late 2024) — current generation;
      strong on motion + camera control.
    • **CogVideoX-5B** (Zhipu, Aug 2024) — HuggingFace integration is
      cleaner (``diffusers.CogVideoXPipeline``); good fallback when Wan
      checkpoints aren't accessible.
    • **HunyuanVideo** (Tencent, Dec 2024) — preferred when first-frame /
      character-reference conditioning matters (see omniweaving.py).
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import BaseVideoGenClient


class WanLocalClient(BaseVideoGenClient):
    backend_name = "wan_local"

    def __init__(self, checkpoint_env: str = "WAN_LOCAL_CHECKPOINT") -> None:
        self.checkpoint = os.getenv(checkpoint_env, "")
        if not self.checkpoint:
            raise RuntimeError(
                "WanLocalClient requires WAN_LOCAL_CHECKPOINT. See .env.example."
            )

    def supported_conditions(self) -> set[str]:
        return {"text", "first_frame"}

    def generate(self, prompt, duration, out_path,
                 first_frame=None, last_frame=None, reference_images=None,
                 flow_field=None, cinematography_hint=None, seed=None) -> Path:
        raise NotImplementedError("WanLocalClient.generate is a v0.2 task.")
