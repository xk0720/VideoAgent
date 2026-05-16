"""Shot-level captioner with rolling 10-shot context.

Open-source backbones (real, swap-in for v0.2 — recommended 2024-2025 picks):
    • **Qwen2.5-VL-7B / 72B** (Alibaba, Jan 2025) — current SOTA open MLLM
      for long-video understanding; supersedes Qwen2-VL-7B.
      https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    • **InternVL2.5** (Shanghai AI Lab, Dec 2024) — strong open alternative.
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
