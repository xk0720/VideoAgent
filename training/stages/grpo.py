"""GRPO RL stage — group-relative policy optimization over EditorEnv rollouts.

References:
    • GRPO (Shao et al., DeepSeek-Math, 2024) — used in DeepSeek-V3 / R1
    • EVA Stage 3 (arXiv 2603.22918) — GRPO for video agents
    • verl 0.7 AgentLoop — production GRPO engine for tool-using agents
    • ProRL Agent (arXiv 2603.18815) — multi-turn GRPO with rollout-as-a-service
    • "It Takes Two: Your GRPO Is Secretly DPO" (arXiv 2510.00977) —
      theoretical connection; means GRPO rollouts can also feed DPO loss

The stub backend here exists for tests + dry-runs. The real backend slots
are split: ``backend="verl"`` and ``backend="prorl"`` because the two
frameworks have meaningfully different ergonomics and we want a one-line
flip rather than a re-write.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..env.base import AgentEnvBase
from ..env.context_manager import ContextManager
from ..policy.base import AgentPolicyBase
from ..rewards.hindsight import HindsightCriticRefiner


@dataclass
class GRPOConfig:
    backend: str = "stub"                              # "stub" | "verl" | "prorl"
    n_rollouts_per_step: int = 4                       # GRPO group size G
    n_steps: int = 4
    max_turns_per_episode: int = 6
    kl_coef: float = 0.04
    lr: float = 1e-6
    clip_ratio: float = 0.2
    advantage_normalisation: str = "leave_one_out"     # GRPO default
    use_hindsight_refiner: bool = True
    hindsight_gamma: float = 0.9


@dataclass
class GRPOMetrics:
    backend: str
    n_rollouts: int
    n_steps: int
    mean_reward: float
    reward_std: float
    elapsed_s: float
    sample_advantages: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


class GRPOStage:
    """Stage C: RL via group-relative policy optimization."""

    name = "grpo"

    def __init__(self, config: Optional[GRPOConfig] = None) -> None:
        self.config = config or GRPOConfig()
        self.refiner = HindsightCriticRefiner(gamma=self.config.hindsight_gamma)

    def fit(
        self,
        env_factory: Callable[[], AgentEnvBase],
        policy: AgentPolicyBase,
        output_dir: Path | str,
    ) -> GRPOMetrics:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.backend == "stub":
            return self._fit_stub(env_factory, policy, output_dir)
        if self.config.backend == "verl":               # pragma: no cover
            return self._fit_verl(env_factory, policy, output_dir)
        if self.config.backend == "prorl":              # pragma: no cover
            return self._fit_prorl(env_factory, policy, output_dir)
        raise ValueError(f"Unknown GRPO backend {self.config.backend!r}")

    # ─── stub backend: roll out + score, no gradients ────────────────

    def _fit_stub(self, env_factory, policy, output_dir: Path) -> GRPOMetrics:
        t0 = time.perf_counter()
        per_step_advantages: list[float] = []
        rewards: list[float] = []
        for step in range(self.config.n_steps):
            group_rewards: list[float] = []
            for g in range(self.config.n_rollouts_per_step):
                ep_reward, ep_step_rewards = self._rollout(env_factory(), policy)
                rewards.append(ep_reward)
                group_rewards.append(ep_reward)
            # GRPO leave-one-out baseline
            for r in group_rewards:
                others = [rr for rr in group_rewards if rr is not r]
                baseline = sum(others) / max(1, len(others))
                per_step_advantages.append(r - baseline)
        mean_r = sum(rewards) / max(1, len(rewards))
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(1, len(rewards)))
        elapsed = time.perf_counter() - t0
        metrics = GRPOMetrics(
            backend="stub", n_rollouts=len(rewards), n_steps=self.config.n_steps,
            mean_reward=mean_r, reward_std=std_r, elapsed_s=elapsed,
            sample_advantages=per_step_advantages[:16],
            extras={"hindsight_refiner": self.config.use_hindsight_refiner},
        )
        (output_dir / "grpo_metrics.json").write_text(
            json.dumps(asdict(metrics), indent=2), encoding="utf-8")
        return metrics

    def _rollout(self, env: AgentEnvBase, policy: AgentPolicyBase) -> tuple[float, list[float]]:
        ctx = ContextManager(max_turns=self.config.max_turns_per_episode)
        obs = env.reset()
        ctx.push(action=None, observation=obs, reward=None)
        policy.reset()
        step_rewards: list[float] = []
        for _ in range(self.config.max_turns_per_episode):
            out = policy.act(obs, ctx.to_messages())
            res = env.step(out.action)
            step_rewards.append(res.reward)
            ctx.push(action=out.action, observation=res.observation, reward=res.reward)
            obs = res.observation
            if res.terminated or res.truncated:
                break
        total = sum(step_rewards)
        if self.config.use_hindsight_refiner and step_rewards:
            credits = self.refiner.refine(step_rewards, total)
            total = sum(c.refined_reward * c.weight for c in credits)
        return total, step_rewards

    # ─── real backends — left as v0.4 work ───────────────────────────

    def _fit_verl(self, env_factory, policy, output_dir):       # pragma: no cover
        raise NotImplementedError(
            "GRPOStage(backend='verl') is a v0.4 task. Plug in verl 0.7 "
            "AgentLoop and pass env_factory/policy via verl's server/client "
            "split (see https://verl.readthedocs.io/en/latest/start/agentic_rl.html)."
        )

    def _fit_prorl(self, env_factory, policy, output_dir):      # pragma: no cover
        raise NotImplementedError(
            "GRPOStage(backend='prorl') is a v0.4 task. Use ProRL Agent's "
            "rollout-as-a-service API (arXiv 2603.18815) — env wrapped as "
            "sandbox, policy served via verl or NeMo-RL."
        )


__all__ = ["GRPOStage", "GRPOConfig", "GRPOMetrics"]
