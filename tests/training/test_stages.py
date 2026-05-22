"""Three-stage trainers — stub backend."""
from __future__ import annotations

import json

from training.data.preference_dataset import PreferenceDataset
from training.data.sft_dataset import SFTDataset
from training.data.trajectory_dataset import TrajectoryDataset
from training.stages.kto import KTOConfig, KTOStage
from training.stages.sft import SFTConfig, SFTStage


def test_sft_stage_stub(sample_trajectory_path, tmp_path):
    traj = TrajectoryDataset.from_file(sample_trajectory_path)
    ds = SFTDataset.from_trajectory(traj, reward_threshold=7.0)
    stage = SFTStage(SFTConfig(backend="stub", n_epochs=2))
    m = stage.fit(ds, tmp_path / "sft")
    assert m.backend == "stub"
    assert m.n_records >= 1
    assert (tmp_path / "sft" / "metrics.json").exists()
    assert (tmp_path / "sft" / "ckpt.json").exists()


def test_kto_stage_stub_with_default_loss(sample_preferences_path, tmp_path):
    prefs = PreferenceDataset.from_file(sample_preferences_path)
    stage = KTOStage(KTOConfig(backend="stub", loss_type="kto", n_epochs=1))
    m = stage.fit(prefs, tmp_path / "kto")
    assert m.n_records == 6        # 3 records × 2 (winner+loser) for kto view


def test_kto_stage_stub_with_dpo_view(sample_preferences_path, tmp_path):
    prefs = PreferenceDataset.from_file(sample_preferences_path)
    stage = KTOStage(KTOConfig(backend="stub", loss_type="dpo", n_epochs=1))
    m = stage.fit(prefs, tmp_path / "dpo")
    # dpo uses the raw triples — 3 records
    assert m.n_records == 3
    metrics = json.loads((tmp_path / "dpo" / "metrics.json").read_text())
    assert metrics["extras"]["loss_type"] == "dpo"


def test_grpo_stage_stub(editor_env_factory, mock_policy, tmp_path):
    from training.stages.grpo import GRPOConfig, GRPOStage
    stage = GRPOStage(GRPOConfig(backend="stub", n_rollouts_per_step=2, n_steps=2))
    m = stage.fit(editor_env_factory, mock_policy, tmp_path / "grpo")
    assert m.backend == "stub"
    assert m.n_rollouts == 4  # 2 steps * 2 rollouts each
    assert (tmp_path / "grpo" / "grpo_metrics.json").exists()
