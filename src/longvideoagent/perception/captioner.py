"""Shot-level captioner with rolling 10-shot context.

Open-source backbones (2025-2026 SOTA — recommended picks, newest first):
    • **Qwen3-VL** (Alibaba, arXiv 2511.21631, Nov 2025) — current top open
      MLLM family. 256K native interleaved-modality context, MoE variants
      (30B-A3B, 235B-A22B) for diverse latency/quality trade-offs. Strong on
      ultra-long-video needle-in-a-haystack retrieval — exactly our use case.
      https://github.com/QwenLM/Qwen3-VL
    • **InternVL3 / InternVL3.5** (Shanghai AI Lab, 2025) — InternVL3.5-241B-A28B
      tops open-MLLM leaderboards; preferred when document/OCR understanding
      matters (e.g. on-screen text inside our source videos).
    • **Qwen2.5-VL** (Jan 2025) — earlier, smaller, still strong baseline.
    • **GPT-4o** / **Claude Sonnet 4.5** / **Gemini 2.5 Pro** — hosted MLLMs.

v0.1 mock yields a deterministic templated caption so the rest of the
pipeline can flow without an MLLM.

The rolling-buffer trick (concat the previous K captions into the prompt
for shot K+1) follows CineAgents and is the same scheme used by
**Video-of-Thought** (Fei et al., 2024) for long-video reasoning chains.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional

from ..config import PreprocessCfg
from ..models.llm.base import BaseLLMClient


class ShotCaptioner:
    def __init__(
        self,
        cfg: PreprocessCfg,
        mllm_client: Optional[BaseLLMClient] = None,
        mock: bool = False,
    ) -> None:
        self.cfg = cfg
        self.mllm = mllm_client
        self.mock = mock
        self.buffer: Deque[str] = deque(maxlen=cfg.captioner.buffer_size)

    def caption(self, video_path: Path | str, start_s: float, end_s: float) -> str:
        if self.mock or self.mllm is None:
            cap = f"[mock-caption] shot from {Path(video_path).name} @ {start_s:.1f}-{end_s:.1f}s " \
                  f"(context={len(self.buffer)} prev shots)"
            self.buffer.append(cap)
            return cap
        # v0.2: build a multimodal message with last K captions + sampled frames.
        prior = "\n".join(self.buffer)
        prompt = (
            f"Previous shots:\n{prior}\n\n"
            f"Describe the current shot at {start_s:.2f}-{end_s:.2f}s of {video_path}. "
            f"Be concise (<= 30 words)."
        )
        resp = self.mllm.chat([{"role": "user", "content": prompt}])
        text = resp.text if hasattr(resp, "text") else str(resp)
        self.buffer.append(text)
        return text


__all__ = ["ShotCaptioner"]
