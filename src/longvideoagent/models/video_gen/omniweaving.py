"""OmniWeaving client wrapper.

Open-source repository: https://github.com/Tencent-Hunyuan/OmniWeaving
There is no `omniweaving` package on PyPI as of writing; we vendor the
model weights and inference code under v0.2 and either:
    (a) call an in-process pipeline, or
    (b) hit a self-hosted HTTP endpoint (env: OMNIWEAVING_ENDPOINT).

This file is intentionally a stub — the public surface matches BaseVideoGenClient
so EditorAgent can call it once weights are configured.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import BaseVideoGenClient


class OmniWeavingClient(BaseVideoGenClient):
    backend_name = "omniweaving"

    def __init__(self, endpoint_env: str = "OMNIWEAVING_ENDPOINT") -> None:
        self.endpoint = os.getenv(endpoint_env, "")
        if not self.endpoint:
            raise RuntimeError(
                "OmniWeaving requires OMNIWEAVING_ENDPOINT to be set "
                "(see .env.example). Use mocks=true for v0.1."
            )

    def supported_conditions(self) -> set[str]:
        # Per upstream model card.
        return {"text", "first_frame", "last_frame", "reference_images", "flow_field"}

    def generate(self, prompt, duration, out_path,
                 first_frame=None, last_frame=None, reference_images=None,
                 flow_field=None, cinematography_hint=None, seed=None) -> Path:
        # v0.2 implementation goes here. Pseudo-code:
        #   payload = {"prompt": prompt, "duration": duration, ...}
        #   r = requests.post(f"{self.endpoint}/generate", json=payload)
        #   r.raise_for_status(); save bytes to out_path.
        raise NotImplementedError("OmniWeavingClient.generate is a v0.2 task.")
