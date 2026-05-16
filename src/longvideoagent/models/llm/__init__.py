"""LLM client wrappers.

Available backends (each lives in its own file):
    • OpenAIClient     — ``openai`` SDK             https://github.com/openai/openai-python
    • AnthropicClient  — ``anthropic`` SDK          https://github.com/anthropics/anthropic-sdk-python
    • DeepSeekClient   — OpenAI-compatible endpoint https://api-docs.deepseek.com
    • VLLMLocalClient  — OpenAI-compatible endpoint https://github.com/vllm-project/vllm
    • MockLLMClient    — deterministic stub used by v0.1 tests / mock pipeline
"""
from .base import BaseLLMClient, LLMResponse, MockLLMClient, build_llm_from_alias

__all__ = ["BaseLLMClient", "LLMResponse", "MockLLMClient", "build_llm_from_alias"]
