"""Runners + end-to-end pipeline integration test."""
from __future__ import annotations

import pytest

from training.runners.pipeline import PipelineConfig, PipelineRunner
from training.runners.rollout import RolloutRunner


def test_rollout_runner_produces_transitions(editor_env_factory, mock_policy, tmp_path):
    runner = RolloutRunner(max_turns_per_episode=3, context_window=3)
    res = runner.run(editor_env_factory, mock_policy, n_episodes=2)
    assert len(res.episode_returns) == 2
    assert len(res.transitions) >= 2
    dump = tmp_path / "rollouts.jsonl"
    runner.dump(res, dump)
    lines = dump.read_text().strip().splitlines()
    assert len(lines) == len(res.transitions)


@pytest.mark.integration
def test_end_to_end_pipeline_runner(
    sample_trajectory_path, sample_preferences_path,
    editor_env_factory, mock_policy, tmp_path,
):
    """v0.3 RM + v0.4 SFT + KTO + GRPO orchestration."""
    from training.rewards.editing_quality_rm import EditingQualityRM
    from training.stages.grpo import GRPOConfig
    from training.stages.sft import SFTConfig
    from training.stages.kto import KTOConfig

    cfg = PipelineConfig(
        sft=SFTConfig(backend="stub", n_epochs=1),
        kto=KTOConfig(backend="stub", loss_type="kto", n_epochs=1),
        grpo=GRPOConfig(backend="stub", n_rollouts_per_step=2, n_steps=1),
    )

    def env_factory_with_rm(rm: EditingQualityRM):
        # The pipeline calls us with a trained EditingQualityRM; for the smoke
        # test we ignore it and reuse the fixture-bound MockRewardModel env.
        return editor_env_factory()

    runner = PipelineRunner(cfg)
    report = runner.run(
        trajectory_path=sample_trajectory_path,
        preferences_path=sample_preferences_path,
        env_factory=env_factory_with_rm,
        policy=mock_policy,
        output_dir=tmp_path / "pipeline",
    )

    assert report.rm_accuracy > 0.5
    assert report.sft_metrics["n_records"] >= 1
    assert report.kto_metrics["n_records"] == 6
    assert report.grpo_metrics["n_rollouts"] >= 1
    assert (tmp_path / "pipeline" / "pipeline_report.json").exists()
    assert (tmp_path / "pipeline" / "editing_quality_rm.json").exists()
