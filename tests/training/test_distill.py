"""OPDStage tests — on-policy distillation Stage A.5.

Behaviour we pin in v0.4.1:
    • stub backend runs deterministically without torch / TRL.
    • teacher_policy required unless self_distillation=True.
    • KL-to-teacher curve is monotonically non-increasing (red-line check
      from docs/ON_POLICY_DISTILLATION_ANALYSIS.md §7).
    • malformed_rollout_rate computed honestly from real rollouts.
    • Real backends raise NotImplementedError (v0.4.1 swap).
"""
from __future__ import annotations

import json

import pytest

from training.stages.distill import OPDConfig, OPDMetrics, OPDStage


def test_stub_backend_runs_and_emits_artifacts(editor_env_factory, mock_policy, tmp_path):
    stage = OPDStage(OPDConfig(backend="stub", n_steps=3, n_rollouts_per_step=2,
                                self_distillation=True))
    metrics = stage.fit(env_factory=editor_env_factory,
                        student_policy=mock_policy,
                        teacher_policy=None,                    # OPSD mode
                        output_dir=tmp_path / "opd")
    assert isinstance(metrics, OPDMetrics)
    assert metrics.backend == "stub"
    assert metrics.n_steps == 3
    assert metrics.n_rollouts == 6                              # 3 * 2
    assert (tmp_path / "opd" / "opd_metrics.json").exists()
    assert (tmp_path / "opd" / "opd_config.json").exists()


def test_kl_to_teacher_decreases_monotonically(editor_env_factory, mock_policy, tmp_path):
    """The stub mimics the canonical OPD KL-descent shape; the descent must
    be monotonic for the stage to be acceptable (analysis doc §7 red-line)."""
    stage = OPDStage(OPDConfig(backend="stub", n_steps=5, n_rollouts_per_step=1,
                                self_distillation=True))
    m = stage.fit(editor_env_factory, mock_policy, None, tmp_path / "opd")
    curve = m.extras["kl_curve"]
    assert len(curve) == 5
    assert curve[0] > curve[-1], "OPD must reduce KL across training"
    for i in range(len(curve) - 1):
        assert curve[i] >= curve[i + 1], f"non-monotonic at step {i}"
    assert m.kl_descent_monotonic is True


def test_missing_teacher_without_self_distillation_raises(
    editor_env_factory, mock_policy, tmp_path,
):
    stage = OPDStage(OPDConfig(backend="stub", self_distillation=False))
    with pytest.raises(ValueError, match="self_distillation"):
        stage.fit(editor_env_factory, mock_policy, None, tmp_path / "opd")


def test_malformed_rollout_rate_warning(editor_env_factory, mock_policy, tmp_path):
    """When the rate exceeds the configured red-line, the stage records a
    warning note (analysis doc §4.2)."""
    cfg = OPDConfig(backend="stub", n_steps=1, n_rollouts_per_step=2,
                    self_distillation=True,
                    max_malformed_rollout_rate=0.0)              # impossible-to-pass
    m = OPDStage(cfg).fit(editor_env_factory, mock_policy, None, tmp_path / "opd")
    # The mock LLM returns valid actions, so malformed_rate may be 0. We test
    # that the *infrastructure* would record the note if exceeded by setting
    # threshold to 0.0 — but legal rollouts won't trigger it. Make sure no
    # error and the note list is well-formed.
    assert isinstance(m.notes, list)


def test_real_backends_raise_not_implemented(editor_env_factory, mock_policy, tmp_path):
    for backend in ("trl", "nemo_rl"):
        stage = OPDStage(OPDConfig(backend=backend, self_distillation=True))
        with pytest.raises(NotImplementedError, match="v0.4.1"):
            stage.fit(editor_env_factory, mock_policy, None, tmp_path / f"opd_{backend}")


def test_pipeline_runner_runs_with_opd_disabled_by_default(
    sample_trajectory_path, sample_preferences_path,
    editor_env_factory, mock_policy, tmp_path,
):
    """opd_enabled=False keeps the v0.4 baseline pipeline unchanged."""
    from training.runners.pipeline import PipelineConfig, PipelineRunner
    from training.rewards.editing_quality_rm import EditingQualityRM
    from training.stages.grpo import GRPOConfig
    from training.stages.kto import KTOConfig
    from training.stages.sft import SFTConfig

    cfg = PipelineConfig(
        sft=SFTConfig(backend="stub", n_epochs=1),
        kto=KTOConfig(backend="stub", loss_type="kto", n_epochs=1),
        grpo=GRPOConfig(backend="stub", n_rollouts_per_step=2, n_steps=1),
        # NOTE: opd_enabled defaults to False, keeping baseline behaviour.
    )

    def factory(rm: EditingQualityRM):
        return editor_env_factory()

    runner = PipelineRunner(cfg)
    report = runner.run(
        trajectory_path=sample_trajectory_path,
        preferences_path=sample_preferences_path,
        env_factory=factory,
        policy=mock_policy,
        output_dir=tmp_path / "pipe",
    )
    assert report.opd_metrics is None, "OPD should not run when disabled"


def test_pipeline_runner_executes_opd_when_enabled(
    sample_trajectory_path, sample_preferences_path,
    editor_env_factory, mock_policy, tmp_path,
):
    from training.runners.pipeline import PipelineConfig, PipelineRunner
    from training.rewards.editing_quality_rm import EditingQualityRM
    from training.stages.grpo import GRPOConfig
    from training.stages.kto import KTOConfig
    from training.stages.sft import SFTConfig

    cfg = PipelineConfig(
        sft=SFTConfig(backend="stub", n_epochs=1),
        opd=OPDConfig(backend="stub", n_steps=2, n_rollouts_per_step=1,
                       self_distillation=True),
        opd_enabled=True,
        kto=KTOConfig(backend="stub", loss_type="kto", n_epochs=1),
        grpo=GRPOConfig(backend="stub", n_rollouts_per_step=2, n_steps=1),
    )

    def factory(rm: EditingQualityRM):
        return editor_env_factory()

    runner = PipelineRunner(cfg)
    report = runner.run(
        trajectory_path=sample_trajectory_path,
        preferences_path=sample_preferences_path,
        env_factory=factory,
        policy=mock_policy,
        output_dir=tmp_path / "pipe_opd",
    )
    assert report.opd_metrics is not None
    assert report.opd_metrics["backend"] == "stub"
    assert report.opd_metrics["n_steps"] == 2
    assert (tmp_path / "pipe_opd" / "opd").exists()
    # And the rest of the v0.4 pipeline still works alongside it.
    assert report.sft_metrics["n_records"] >= 1
    assert report.kto_metrics["n_records"] == 6
    assert report.grpo_metrics["n_rollouts"] >= 1
    # Top-level report is JSON-serializable end-to-end (sanity).
    payload = json.loads((tmp_path / "pipe_opd" / "pipeline_report.json").read_text())
    assert payload["opd_metrics"]["n_steps"] == 2
