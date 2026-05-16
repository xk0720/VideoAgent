"""Hosted-API video generation fallback (Veo 2 via google-genai).

Open-source SDK wrapped: **google-genai**  https://github.com/google/genai-python

Current API target (Dec 2024 release):
    • **Veo 2** — text-to-video API on Google AI Studio / Vertex AI.
      Higher fidelity than Veo 1; supports 8-second clips at 720p.

Other 2024–2025 hosted-API options behind the same interface (each needs
its own SDK + key):
    • **Sora 2** (OpenAI, late 2024) — when an API becomes generally available.
    • **Kling 2.0** (Kuaishou, 2024) — strong open-API option.
    • **Runway Gen-3 Alpha** — popular for editor workflows.

v0.1 stub — actual call happens once GOOGLE_API_KEY is set.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import BaseVideoGenClient


class ApiVideoGenClient(BaseVideoGenClient):
    backend_name = "api"

    def __init__(self, provider: str = "google_genai", model: str = "veo-2.0",
                 api_key_env: str = "GOOGLE_API_KEY") -> None:
        self.provider = provider
        self.model = model
        self.api_key = os.getenv(api_key_env, "")

    def supported_conditions(self) -> set[str]:
        return {"text"}

    def generate(self, prompt, duration, out_path,
                 first_frame=None, last_frame=None, reference_images=None,
                 flow_field=None, cinematography_hint=None, seed=None) -> Path:
        # v0.2: call google-genai or other provider; save bytes to out_path.
        raise NotImplementedError("ApiVideoGenClient.generate is a v0.2 task.")
