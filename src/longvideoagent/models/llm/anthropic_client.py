"""Anthropic Messages API wrapper.

Open-source dependency: **anthropic>=0.30** (https://github.com/anthropics/anthropic-sdk-python).
"""
from __future__ import annotations

import os

from .base import BaseLLMClient, LLMResponse


class AnthropicClient(BaseLLMClient):
    backend_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:                                # pragma: no cover
            raise ImportError(
                "anthropic package not installed. `pip install 'longvideoagent[llm]'`"
            ) from e
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.getenv(api_key_env))

    def supports_function_calling(self) -> bool:
        return True

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=1024, **kwargs) -> LLMResponse:
        # Anthropic separates system messages from user/assistant turns.
        system_chunks = [m["content"] for m in messages if m["role"] == "system"]
        turns = [m for m in messages if m["role"] != "system"]
        kwargs_ = dict(model=self.model, max_tokens=max_tokens, temperature=temperature,
                       system="\n".join(system_chunks) or None, messages=turns)
        if tools:
            kwargs_["tools"] = tools
        resp = self._client.messages.create(**kwargs_)          # pragma: no cover
        text = "".join(b.text for b in resp.content if b.type == "text")  # pragma: no cover
        tool_calls = [b.model_dump() for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=tool_calls, raw=resp)
