"""Model wrappers — stable external interfaces. v0.1 ships mocks (CPU-only).

Swap mocks for real backends in v0.2 by implementing the same ABCs:
  - llm:        DeepSeek / GPT / Claude / local vLLM
  - mllm:       Qwen-VL etc. (used as judge / critic)
  - video_gen:  OmniWeaving / Wan / Veo|Sora API  (must accept conditioning)
  - image_edit: Qwen-Image-Edit etc. (keyframe local editing)
"""
from .llm import BaseLLMClient, MockLLMClient, build_llm
from .mllm import BaseMLLMClient, MockMLLMClient, build_mllm
from .video_gen import BaseVideoGenClient, MockVideoGenClient, build_video_gen
from .image_edit import BaseImageEditClient, MockImageEditClient, build_image_edit
from .audio_gen_backends import (
    BaseAudioGenClient,
    MockAudioGenClient,
    WaveSpeedAudioClient,
    build_audio_gen,
)

__all__ = [
    "BaseLLMClient", "MockLLMClient", "build_llm",
    "BaseMLLMClient", "MockMLLMClient", "build_mllm",
    "BaseVideoGenClient", "MockVideoGenClient", "build_video_gen",
    "BaseImageEditClient", "MockImageEditClient", "build_image_edit",
    "BaseAudioGenClient", "MockAudioGenClient", "WaveSpeedAudioClient",
    "build_audio_gen",
]
