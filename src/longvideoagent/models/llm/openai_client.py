"""OpenAI ChatCompletion / Responses-API wrapper.

Open-source dependency: **openai>=1.30** (https://github.com/openai/openai-python).

Used for:
    • Direct OpenAI calls (GPT-4o, etc.)
    • Anything OpenAI-compatible — DeepSeek and vLLM specialise this class
      via base_url / api_key.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    backend_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:                                # pragma: no cover
            raise ImportError(
                "openai package not installed. `pip install 'longvideoagent[llm]'`"
            ) from e
        self.model = model
        api_key = os.getenv(api_key_env)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def supports_function_calling(self) -> bool:
        return True

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=1024, **kwargs) -> LLMResponse:
        kwargs_ = dict(model=self.model, messages=messages, temperature=temperature,
                       max_tokens=max_tokens)
        if tools:
            kwargs_["tools"] = tools
        resp = self._client.chat.completions.create(**kwargs_)  # pragma: no cover
        choice = resp.choices[0].message                         # pragma: no cover
        tool_calls = [tc.model_dump() for tc in (choice.tool_calls or [])]  # pragma: no cover
        return LLMResponse(text=choice.content or "", tool_calls=tool_calls, raw=resp)
