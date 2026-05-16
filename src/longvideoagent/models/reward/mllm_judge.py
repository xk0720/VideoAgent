"""Zero-shot MLLM-as-judge reward model.

Open-source backbones (selected via configs/models/mllm.yaml):
    • Qwen2-VL  https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
    • GPT-4o    https://openai.com
    • Claude Sonnet vision https://www.anthropic.com

v0.1: returns a deterministic ``MockRewardModel`` style result; the prompt
template lives in src/longvideoagent/prompts/reward_judge.txt.
"""
from __future__ import annotations

from typing import Optional

from ...config import load_prompt
from ..llm.base import BaseLLMClient, MockLLMClient
from .base import BaseRewardModel, MockRewardModel, RewardResult


class MLLMJudge(BaseRewardModel):
    def __init__(
        self,
        mllm: Optional[BaseLLMClient] = None,
        accept_threshold: float = 6.0,
        prompt_name: str = "reward_judge",
    ) -> None:
        self.mllm = mllm or MockLLMClient(alias="validator")
        self.accept_threshold = accept_threshold
        self.prompt_template = load_prompt(prompt_name)
        self._fallback = MockRewardModel(accept_threshold=accept_threshold)

    def score(self, candidate, guidance, context=None) -> RewardResult:
        if isinstance(self.mllm, MockLLMClient):
            return self._fallback.score(candidate, guidance, context)

        # v0.2 path: format the prompt, send candidate video frames.
        prompt = self.prompt_template.format(                   # pragma: no cover
            semantic_query=guidance.semantic_query,
            heuristic=guidance.editing_heuristic,
            cinematography=", ".join(guidance.cinematography_hints),
        )
        resp = self.mllm.chat([                                 # pragma: no cover
            {"role": "system", "content": "You are a strict video-editing reward model."},
            {"role": "user", "content": prompt},
        ])
        try:                                                    # pragma: no cover
            parsed = resp.parse_json()
            score = float(parsed["score"])
            return RewardResult(
                score=score,
                accepted=bool(parsed.get("accepted", score >= self.accept_threshold)),
                reasons=list(parsed.get("reasons", [])),
                raw=resp.raw,
            )
        except Exception:                                       # pragma: no cover
            return RewardResult(score=5.0, accepted=False, reasons=["parse_failed"])
