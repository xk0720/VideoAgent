"""PipelineRunner — orchestrate v0.3 RM → v0.4 SFT+KTO+GRPO in one call.

In real production each stage runs on its own GPU cluster; this driver
walks them sequentially so testing + dry-runs stay simple.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..data.preference_dataset import PreferenceDataset
from ..data.sft_dataset import SFTDataset
from ..data.trajectory_dataset import TrajectoryDataset
from ..env.base import AgentEnvBase
from ..policy.base import AgentPolicyBase
from ..rewards.editing_quality_rm import EditingQualityRM, EditingQualityRMTrainer
from ..stages.distill import OPDConfig, OPDStage
from ..stages.grpo import GRPOStage, GRPOConfig
from ..stages.kto import KTOStage, KTOConfig
from ..stages.sft import SFTConfig, SFTStage


@dataclass
class PipelineConfig:
    sft: SFTConfig = field(default_factory=SFTConfig)
    # Stage A.5 — on-policy distillation. ``opd_enabled=False`` by default
    # so v0.4 baseline pipelines (SFT→KTO→GRPO) stay unaffected; enable
    # only once the §4.4 baseline numbers are in (see ON_POLICY_DISTILLATION_ANALYSIS.md).
    opd: OPDConfig = field(default_factory=OPDConfig)
    opd_enabled: bool = False
    kto: KTOConfig = field(default_factory=KTOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    reward_threshold_for_sft: float = 7.0
    rm_n_epochs: int = 5
    rm_lr: float = 0.1


@dataclass
class PipelineReport:
    rm_path: str
    rm_accuracy: float
    sft_metrics: dict[str, Any]
    opd_metrics: Optional[dict[str, Any]] = None     # only present when opd_enabled
    kto_metrics: dict[str, Any] = field(default_factory=dict)
    grpo_metrics: dict[str, Any] = field(default_factory=dict)


class PipelineRunner:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        trajectory_path: Path | str,
        preferences_path: Path | str,
        env_factory: Callable[[EditingQualityRM], AgentEnvBase],
        policy: AgentPolicyBase,
        output_dir: Path | str,
        teacher_policy: Optional[AgentPolicyBase] = None,
    ) -> PipelineReport:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── v0.3: train EditingQualityRM ─────────────────────────────
        prefs = PreferenceDataset.from_file(preferences_path)
        rm_trainer = EditingQualityRMTrainer(
            lr=self.config.rm_lr, n_epochs=self.config.rm_n_epochs)
        rm = rm_trainer.fit(prefs)
        rm_path = output_dir / "editing_quality_rm.json"
        rm_trainer.save(rm_path)
        # In-train accuracy as proxy for hold-out (we'd split in real life).
        accuracy = (rm_trainer.stats.n_correct / max(1, rm_trainer.stats.n_seen))

        # ── v0.4 Stage A: SFT cold start ────────────────────────────
        traj = TrajectoryDataset.from_file(trajectory_path)
        sft_data = SFTDataset.from_trajectory(
            traj, reward_threshold=self.config.reward_threshold_for_sft)
        sft_metrics = SFTStage(self.config.sft).fit(sft_data, output_dir / "sft")

        # ── v0.4.1 Stage A.5: On-Policy Distillation (optional) ──────
        #   See docs/ON_POLICY_DISTILLATION_ANALYSIS.md for the rationale.
        #   Disabled by default — only flip ``config.opd_enabled=True`` once
        #   the baseline SFT→KTO→GRPO numbers exist (§7 of the analysis doc).
        opd_metrics_dict: Optional[dict[str, Any]] = None

        def _env_factory() -> AgentEnvBase:
            return env_factory(rm)

        if self.config.opd_enabled:
            opd_metrics = OPDStage(self.config.opd).fit(
                env_factory=_env_factory,
                student_policy=policy,
                teacher_policy=teacher_policy,
                output_dir=output_dir / "opd",
            )
            opd_metrics_dict = asdict(opd_metrics)

        # ── v0.4 Stage B: KTO preference align ──────────────────────
        kto_metrics = KTOStage(self.config.kto).fit(prefs, output_dir / "kto")

        # ── v0.4 Stage C: GRPO RL ───────────────────────────────────
        grpo_metrics = GRPOStage(self.config.grpo).fit(
            _env_factory, policy, output_dir / "grpo")

        report = PipelineReport(
            rm_path=str(rm_path), rm_accuracy=accuracy,
            sft_metrics=asdict(sft_metrics),
            opd_metrics=opd_metrics_dict,
            kto_metrics=asdict(kto_metrics),
            grpo_metrics=asdict(grpo_metrics),
        )
        (output_dir / "pipeline_report.json").write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8")
        return report


__all__ = ["PipelineRunner", "PipelineConfig", "PipelineReport"]
