"""On-Policy Distillation stage (Stage A.5).

Sits between Stage A (SFT cold start) and Stage B (KTO preference alignment)
in the v0.4.1 pipeline. The rationale and the literature review for this
position are in [`docs/ON_POLICY_DISTILLATION_ANALYSIS.md`](../../docs/ON_POLICY_DISTILLATION_ANALYSIS.md);
this docstring is the brief reference + pointer.

References (rigorous — every arXiv ID verified via WebSearch in May 2026):
    • **GKD: On-Policy Distillation of Language Models**
      Agarwal et al., ICLR 2024. The original work that formalised OPD as a
      special case of Generalized Knowledge Distillation (GKD) — student-on-policy
      rollouts + reverse-KL toward a teacher's token-level distribution.
      https://arxiv.org/abs/2306.13649
    • **Thinking Machines Lab — On-Policy Distillation (blog, Oct 2025)**
      Popularised the recipe; "7–10× fewer gradient steps for RL-level performance".
      Kevin Lu et al. https://thinkingmachines.ai/blog/on-policy-distillation/
    • **OPSD — Self-Distilled Reasoner (2026)**
      Self-distillation variant when no external teacher available.
      https://arxiv.org/abs/2601.18734
    • **Stable OPD through Adaptive Target Reformulation (2026)**
      Stability fix for early-training noisy rollouts.
      https://arxiv.org/abs/2601.07155
    • **Entropy-Aware OPD (2026)**
      Adaptive reweighting that avoids mode collapse.
      https://arxiv.org/abs/2603.07079
    • **REOPOLD — Relaxed OPD (2026)**
      Relaxed objective; reports 6.7–12× sample efficiency over RL.
      https://arxiv.org/abs/2603.11137
    • **A Survey of OPD for LLMs (2026)**
      Taxonomy; flags trajectory-aware credit assignment as an open problem
      for agentic tasks — which we cover via the HCAPO refiner in v0.4.
      https://arxiv.org/abs/2604.00626
    • **Rethinking OPD: Phenomenology, Mechanism, Recipe (2026)**
      Most useful for engineering: "don't add OPD on faith — baseline first".
      https://arxiv.org/abs/2604.13016
    • **NeMo-RL OPD discussion #1445** — production reference implementation.
      https://github.com/NVIDIA-NeMo/RL/discussions/1445

Why Stage A.5 (not replacing any existing stage):
    • SFT (Stage A) is a *prerequisite* for OPD: student must produce legal
      JSON rollouts before token-level supervision is meaningful.
      ([Rethinking OPD, 2604.13016] "Mechanism" section.)
    • KTO (Stage B) does *pairwise* preference alignment; OPD does
      *distribution* alignment. Different objectives — not interchangeable.
    • GRPO (Stage C) is bounded by the teacher when used after OPD; we keep
      it for capability extension beyond the teacher.
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


@dataclass
class OPDConfig:
    """Configuration for the on-policy distillation stage.

    Defaults follow the recipe in Rethinking OPD (arXiv 2604.13016) §"Recipe"
    plus our practical adjustments for short-episode agent tasks.
    """
    backend: str = "stub"                       # "stub" | "trl" | "nemo_rl"
    teacher_model: str = "claude-sonnet-4-5"    # arbitrary BaseLLMClient alias
    student_model: str = "deepseek-chat"
    loss_type: str = "reverse_kl"               # "reverse_kl" | "forward_kl" | "jsd"
    self_distillation: bool = False             # OPSD mode if True (no external teacher)
    n_rollouts_per_step: int = 8                # student rollouts per gradient step
    n_steps: int = 4
    max_turns_per_episode: int = 6
    teacher_temperature: float = 0.0            # mode-seeking → low T
    student_temperature: float = 0.7            # explore during rollout
    learning_rate: float = 1e-6
    entropy_aware: bool = False                 # Entropy-Aware OPD (arXiv 2603.07079)
    relaxed_alpha: float = 1.0                  # REOPOLD (arXiv 2603.11137); 1.0 = vanilla
    # Safeguards from the analysis doc §4
    max_malformed_rollout_rate: float = 0.6     # red-line: stop if exceeded


@dataclass
class OPDMetrics:
    backend: str
    n_rollouts: int
    n_steps: int
    mean_kl_to_teacher: float
    final_kl_to_teacher: float
    kl_descent_monotonic: bool                  # red-line check from §7
    malformed_rollout_rate: float
    elapsed_s: float
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


class OPDStage:
    """Stage A.5: on-policy distillation.

    The stub backend simulates the *behaviour we expect from a real OPD run* —
    KL-to-teacher decreasing monotonically while student rollouts dominate.
    Real backends (TRL ``GKDTrainer`` / NeMo-RL) plug in at the marked seams.
    """

    name = "opd"

    def __init__(self, config: Optional[OPDConfig] = None) -> None:
        self.config = config or OPDConfig()

    # ─── public API ────────────────────────────────────────────────────

    def fit(
        self,
        env_factory: Callable[[], AgentEnvBase],
        student_policy: AgentPolicyBase,
        teacher_policy: Optional[AgentPolicyBase],
        output_dir: Path | str,
    ) -> OPDMetrics:
        """Run one OPD training session.

        Parameters
        ----------
        env_factory:
            Same shape as ``GRPOStage.fit`` — provides fresh AgentEnv per rollout.
        student_policy:
            The model we're training (post-SFT EditorAgent in practice).
        teacher_policy:
            The stronger model whose token-level distribution we're matching.
            If ``None`` AND ``config.self_distillation`` is True → OPSD mode
            (student is its own teacher with lower temperature).
        output_dir:
            Where to write metrics + (when backend != "stub") checkpoints.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if teacher_policy is None and not self.config.self_distillation:
            raise ValueError(
                "OPDStage.fit needs either a teacher_policy or "
                "config.self_distillation=True (OPSD, arXiv 2601.18734)."
            )

        if self.config.backend == "stub":
            metrics = self._fit_stub(env_factory, student_policy, teacher_policy)
        elif self.config.backend == "trl":                # pragma: no cover
            metrics = self._fit_trl(env_factory, student_policy, teacher_policy, output_dir)
        elif self.config.backend == "nemo_rl":            # pragma: no cover
            metrics = self._fit_nemo_rl(env_factory, student_policy, teacher_policy, output_dir)
        else:
            raise ValueError(f"Unknown OPD backend {self.config.backend!r}")

        # Persist artifacts
        (output_dir / "opd_metrics.json").write_text(
            json.dumps(asdict(metrics), indent=2), encoding="utf-8"
        )
        (output_dir / "opd_config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8"
        )
        return metrics

    # ─── stub backend: behaviour-faithful, no gradients ───────────────

    def _fit_stub(
        self,
        env_factory: Callable[[], AgentEnvBase],
        student_policy: AgentPolicyBase,
        teacher_policy: Optional[AgentPolicyBase],
    ) -> OPDMetrics:
        t0 = time.perf_counter()
        kls: list[float] = []
        n_malformed = 0
        n_total = 0
        notes: list[str] = []

        # Stub mechanic: simulate KL-to-teacher decreasing over training steps.
        # We do *real* student rollouts (so trajectory shape is realistic and
        # malformed_rollout_rate is computed honestly), then fabricate a
        # monotone KL curve to mimic OPD's expected behaviour. A real backend
        # would replace this fabrication with the actual reverse-KL gradient
        # computed against teacher logits.
        for step in range(self.config.n_steps):
            for _ in range(self.config.n_rollouts_per_step):
                rollout_was_legal = self._rollout_once(env_factory(), student_policy)
                n_total += 1
                if not rollout_was_legal:
                    n_malformed += 1
            # Synthetic monotone KL — starts at ~1.5 (post-SFT typical) and
            # decays exponentially toward 0.3 (asymptote). Matches the shape
            # plotted in TM Lab's blog and in Rethinking OPD (2604.13016) Fig 2.
            kl = 0.3 + 1.2 * math.exp(-0.6 * step)
            kls.append(kl)

        malformed_rate = n_malformed / max(1, n_total)
        if malformed_rate > self.config.max_malformed_rollout_rate:
            notes.append(
                f"WARNING: malformed_rollout_rate={malformed_rate:.2f} > "
                f"max_malformed_rollout_rate={self.config.max_malformed_rollout_rate:.2f}. "
                "Per analysis §4.2, OPD's effectiveness degrades. Recommend "
                "re-running SFT before OPD."
            )

        # Red-line: monotone KL descent check.
        monotonic = all(kls[i] >= kls[i + 1] for i in range(len(kls) - 1))
        if not monotonic and self.config.backend == "stub":
            notes.append("Non-monotonic KL descent (only meaningful for real backend).")

        elapsed = time.perf_counter() - t0
        return OPDMetrics(
            backend="stub",
            n_rollouts=n_total,
            n_steps=self.config.n_steps,
            mean_kl_to_teacher=sum(kls) / max(1, len(kls)),
            final_kl_to_teacher=kls[-1] if kls else 0.0,
            kl_descent_monotonic=monotonic,
            malformed_rollout_rate=malformed_rate,
            elapsed_s=elapsed,
            notes=notes,
            extras={
                "loss_type": self.config.loss_type,
                "teacher_model": self.config.teacher_model,
                "student_model": self.config.student_model,
                "self_distillation": self.config.self_distillation,
                "entropy_aware": self.config.entropy_aware,
                "relaxed_alpha": self.config.relaxed_alpha,
                "kl_curve": kls,
            },
        )

    def _rollout_once(self, env: AgentEnvBase, policy: AgentPolicyBase) -> bool:
        """Run one episode and return whether the policy produced legal actions
        throughout. A "malformed" rollout has at least one unknown-action penalty."""
        ctx = ContextManager(max_turns=self.config.max_turns_per_episode)
        obs = env.reset()
        ctx.push(action=None, observation=obs, reward=None)
        policy.reset()
        all_legal = True
        for _ in range(self.config.max_turns_per_episode):
            out = policy.act(obs, ctx.to_messages())
            res = env.step(out.action)
            if "error" in res.info:
                all_legal = False
            ctx.push(action=out.action, observation=res.observation, reward=res.reward)
            obs = res.observation
            if res.terminated or res.truncated:
                break
        return all_legal

    # ─── real backends — left as v0.4.1 swap points ───────────────────

    def _fit_trl(self, env_factory, student_policy, teacher_policy,
                 output_dir: Path) -> OPDMetrics:  # pragma: no cover
        # TRL v1.0 (April 2026) ships a GKDTrainer that implements OPD with
        # reverse KL, JSD, and forward KL via ``loss_type``. We would:
        #   1. Generate K student rollouts per step → list[dict(prompt, completion)].
        #   2. Score each token under teacher (temperature=teacher_temperature).
        #   3. Pass (prompts, completions, teacher_logits) to GKDTrainer.
        #   4. Train for n_steps × n_rollouts_per_step gradient steps.
        raise NotImplementedError(
            "OPDStage(backend='trl') is a v0.4.1 task. See TRL v1.0 docs at "
            "https://huggingface.co/docs/trl. The interface should map "
            "OPDConfig.loss_type → GKDConfig.loss_type."
        )

    def _fit_nemo_rl(self, env_factory, student_policy, teacher_policy,
                     output_dir: Path) -> OPDMetrics:  # pragma: no cover
        # NeMo-RL added native OPD support; see issue #1445.
        # Interface is similar to its GRPO support: register a TeacherModel
        # in the trainer config, set training_strategy="opd".
        raise NotImplementedError(
            "OPDStage(backend='nemo_rl') is a v0.4.1 task. See "
            "https://github.com/NVIDIA-NeMo/RL/discussions/1445 for the "
            "production reference implementation."
        )


__all__ = ["OPDStage", "OPDConfig", "OPDMetrics"]
