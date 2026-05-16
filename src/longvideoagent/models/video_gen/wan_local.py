"""Wan2.6 local-inference wrapper (stub).

Open-source: https://github.com/Wan-Video. Weights distributed via Hugging Face.
v0.1 ships only the public surface; v0.2 will load the model with
`transformers` + `accelerate` and run T2V locally.
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
