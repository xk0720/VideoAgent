"""Rollout runner — execute one or more episodes and emit transitions.

The signature is deliberately synchronous; verl's AgentLoop / OpenRLHF's
AgentExecutor wrap this in asyncio when GPU rollouts are batched. For CPU
tests we go one episode at a time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..env.base import AgentEnvBase
from ..env.context_manager import ContextManager
from ..policy.base import AgentPolicyBase


@dataclass
class Transition:
    episode: int
    step: int
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    transitions: list[Transition]
    episode_returns: list[float]


class RolloutRunner:
    def __init__(self, max_turns_per_episode: int = 6,
                 context_window: int = 8) -> None:
        self.max_turns = max_turns_per_episode
        self.context_window = context_window

    def run(
        self,
        env_factory: Callable[[], AgentEnvBase],
        policy: AgentPolicyBase,
        n_episodes: int = 1,
        on_step: Optional[Callable[[Transition], None]] = None,
    ) -> RolloutResult:
        transitions: list[Transition] = []
        episode_returns: list[float] = []
        for ep in range(n_episodes):
            env = env_factory()
            ctx = ContextManager(max_turns=self.context_window)
            policy.reset()
            obs = env.reset()
            ctx.push(action=None, observation=obs, reward=None)
            ep_return = 0.0
            for step in range(self.max_turns):
                out = policy.act(obs, ctx.to_messages())
                res = env.step(out.action)
                t = Transition(
                    episode=ep, step=step,
                    observation={"state": obs.state,
                                 "available_actions": obs.available_actions},
                    action=out.action, reward=res.reward,
                    terminated=res.terminated, truncated=res.truncated,
                    info=res.info,
                )
                transitions.append(t)
                if on_step:
                    on_step(t)
                ctx.push(action=out.action, observation=res.observation, reward=res.reward)
                ep_return += res.reward
                obs = res.observation
                if res.terminated or res.truncated:
                    break
            episode_returns.append(ep_return)
        return RolloutResult(transitions=transitions, episode_returns=episode_returns)

    def dump(self, result: RolloutResult, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for t in result.transitions:
                f.write(json.dumps(asdict(t), default=str) + "\n")


__all__ = ["RolloutRunner", "RolloutResult", "Transition"]
