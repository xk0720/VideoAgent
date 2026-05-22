"""Dataset loader tests."""
from __future__ import annotations

from training.data.preference_dataset import PreferenceDataset
from training.data.sft_dataset import SFTDataset
from training.data.trajectory_dataset import TrajectoryDataset


def test_trajectory_views(sample_trajectory_path):
    ds = TrajectoryDataset.from_file(sample_trajectory_path)
    assert len(ds.steps()) == 3
    assert len(ds.segment_summaries()) == 2
    high = ds.filter_high_reward(threshold=7.0)
    # segment 0 has reward 8.5, segment 1 has reward 4.0 → only seg 0 kept
    kept_segments = {r["action_input"].get("segment_idx")
                     for r in high.segment_summaries()}
    assert kept_segments == {0}


def test_sft_dataset_filters_by_reward(sample_trajectory_path):
    traj = TrajectoryDataset.from_file(sample_trajectory_path)
    sft = SFTDataset.from_trajectory(traj, reward_threshold=7.0, agent_name="editor")
    # Only steps from segment 0 should be retained.
    assert len(sft) >= 1
    for rec in sft:
        assert "prompt" in rec and "completion" in rec


def test_preference_dataset_triples(sample_preferences_path):
    prefs = PreferenceDataset.from_file(sample_preferences_path)
    assert len(prefs) == 3   # 3 records × 1 loser each
    for rec in prefs:
        assert "prompt" in rec and "chosen" in rec and "rejected" in rec
        assert "metric_scores" in rec["chosen"]


def test_preference_kto_view(sample_preferences_path):
    prefs = PreferenceDataset.from_file(sample_preferences_path)
    kto = prefs.kto_view()
    # one positive + one negative per record
    assert len(kto) == 6
    labels = [k["label"] for k in kto]
    assert sum(labels) == 3 and sum(1 for L in labels if not L) == 3
