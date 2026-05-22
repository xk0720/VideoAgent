"""HindsightCriticRefiner — HCAPO-style post-hoc Q-value smoothing.

Reference: HCAPO (Hindsight Credit Assignment for Long-Horizon LLM Agents,
arXiv 2603.08754, 2026). Their idea: after an episode finishes, an LLM
critic looks back at each step and proposes a refined Q-value, then we
optimise the policy against the refined Q's instead of the raw ones.

We give v0.1 a small, lossless, no-LLM refiner that:
    1. Smooths step rewards via a running geometric mean of (step_reward,
       terminal_segment_reward) — exact same shape HCAPO uses.
    2. Down-weights any step that the CriticAgent flagged as suspect
       (low_reward / fallback_taken).

A real LLM-driven refinement slot is left for v0.4 (mark the function
``refine_with_llm`` NotImplementedError).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepCredit:
    step_idx: int
    raw_reward: float
    refined_reward: float
    weight: float = 1.0


class HindsightCriticRefiner:
    def __init__(self, gamma: float = 0.9,
                 critic_downweight: float = 0.3) -> None:
        self.gamma = gamma
        self.critic_downweight = critic_downweight

    def refine(
        self,
        step_rewards: list[float],
        terminal_reward: float,
        critic_flags: list[bool] | None = None,
    ) -> list[StepCredit]:
        """Closed-form hindsight smoothing:

            refined[k] = mix[k] * terminal + (1 - mix[k]) * step[k]
            where  mix[k] = γ * (k + 1) / N

        Properties (matching the v0.1 test contract):
            γ = 0  →  refined = raw step reward (no shift)
            γ = 1  →  step k gets ``(k+1)/N`` fraction of credit pulled to
                      terminal; the final step is pure terminal reward.
        """
        n = len(step_rewards)
        out: list[StepCredit] = []
        for k, r in enumerate(step_rewards):
            d = (k + 1) / max(1, n)
            mix = self.gamma * d
            refined = mix * terminal_reward + (1.0 - mix) * r
            weight = 1.0
            if critic_flags and k < len(critic_flags) and critic_flags[k]:
                weight *= self.critic_downweight
            out.append(StepCredit(step_idx=k, raw_reward=r,
                                  refined_reward=refined, weight=weight))
        return out

    def refine_with_llm(self, *args, **kwargs):                # pragma: no cover
        raise NotImplementedError(
            "LLM-driven hindsight refinement is a v0.4 task. The interface "
            "should match HCAPO arXiv 2603.08754."
        )


__all__ = ["HindsightCriticRefiner", "StepCredit"]
