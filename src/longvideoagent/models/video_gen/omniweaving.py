"""OmniWeaving client wrapper.

Open-source repository: https://github.com/Tencent-Hunyuan/OmniWeaving
There is no `omniweaving` package on PyPI as of writing; we vendor the
model weights and inference code under v0.2 and either:
    (a) call an in-process pipeline, or
    (b) hit a self-hosted HTTP endpoint (env: OMNIWEAVING_ENDPOINT).

This file is intentionally a stub — the public surface matches BaseVideoGenClient
so EditorAgent can call it once weights are configured.

2024–2025 alternatives (recommended upgrade paths):
    • **HunyuanVideo** (Tencent-Hunyuan, Dec 2024) — same lab as OmniWeaving;
      open weights for 13B T2V; strongest "drop-in" candidate.
      https://github.com/Tencent-Hunyuan/HunyuanVideo
    • **CogVideoX-5B** (Zhipu, Aug 2024) — open-weights, smaller, faster.
      https://github.com/THUDM/CogVideo
    • **Mochi-1** (Genmo, Oct 2024) — open T2V with multi-image conditioning.
      https://github.com/genmoai/mochi
    • **LTX-Video** (Lightricks, Dec 2024) — fast 2B model, good for real-time.
      https://github.com/Lightricks/LTX-Video
For long-edit pipelines the controllability of OmniWeaving / HunyuanVideo
(text + first-frame + character refs + flow field) remains the best fit.
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
