"""DeepSeek wrapper.

DeepSeek exposes an OpenAI-compatible endpoint, so we subclass OpenAIClient
and override base_url / api_key. Reference: https://api-docs.deepseek.com.
Open-source dependency: **openai** SDK.
"""
from __future__ import annotations

import os

from .openai_client import OpenAIClient


class DeepSeekClient(OpenAIClient):
    backend_name = "deepseek"

    def __init__(self, model: str = "deepseek-chat") -> None:
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        super().__init__(model=model, api_key_env="DEEPSEEK_API_KEY", base_url=base_url)
