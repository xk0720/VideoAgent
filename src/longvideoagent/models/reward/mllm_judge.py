"""Zero-shot MLLM-as-judge reward model.

Open-source backbones (2024–2025 SOTA — selected via configs/models/mllm.yaml):
    • **Qwen2.5-VL-7B / 72B** (Alibaba, Jan 2025)  — preferred upgrade
      https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    • **InternVL2.5** (Shanghai AI Lab, Dec 2024) — strong alternative
    • **GPT-4o** / **Claude Sonnet 4.5** / **Gemini 2.5 Pro** — hosted MLLMs

Reward-model fine-tuning recipes worth tracking for v0.3 (when we move from
zero-shot judge → trained RM):
    • **Tülu-3-RM** (Allen AI, Nov 2024) — full open RM training recipe
      https://allenai.org/tulu
    • **Skywork-Reward-Gemma-2-27B** (Skywork, Oct 2024) — top RewardBench
      scores; pure pairwise-preference training.
      https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B
    • **JudgeLM / MJ-Bench** (2024) — purpose-trained judges + benchmark.
    • **AgentPRM** (arXiv 2511.08325, late 2025) — Process Reward Model
      specifically for LLM agents; introduces step-wise "promise" and
      "progress" signals trained via Temporal-Difference estimation +
      Generalized Advantage Estimation. The recommended v0.3 recipe when
      our PreferenceLogger trajectories are rich enough.
    • **A Survey of Process Reward Models** (arXiv 2510.08049, late 2025) —
      good landscape of outcome-PRM vs process-PRM trade-offs.
    • **ThinkPRM / GenPRM** (2025) — generative chain-of-thought PRMs that
      verify each step with long-form reasoning; needs much less annotation.
    • **VRPRM** (2025) — Visual Reasoning PRM, directly relevant since our
      candidates are video segments not text steps.

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
