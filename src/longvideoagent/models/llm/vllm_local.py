"""Local vLLM server wrapper.

vLLM (https://github.com/vllm-project/vllm) ships an OpenAI-compatible
HTTP server (`vllm serve <model>`), so we reuse OpenAIClient pointed at
the local URL declared in the VLLM_BASE_URL env var.
Open-source dependency: **openai** SDK + **vllm** (for the server, not the client).
"""
from __future__ import annotations

import os

from .openai_client import OpenAIClient


class VLLMLocalClient(OpenAIClient):
    backend_name = "vllm"

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        super().__init__(model=model, api_key_env="VLLM_API_KEY", base_url=base_url)
