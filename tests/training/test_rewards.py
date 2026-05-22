"""Reward modules: composite + EditingQualityRM + hindsight."""
from __future__ import annotations

from longvideoagent.models.reward.base import MockRewardModel
from longvideoagent.types import EditingSegment, SegmentGuidance

from training.data.preference_dataset import PreferenceDataset
from training.rewards.composite import CompositeReward
from training.rewards.editing_quality_rm import EditingQualityRMTrainer, EditingQualityRM
from training.rewards.hindsight import HindsightCriticRefiner


def _seg(scores, source="retrieval"):
    return EditingSegment(segment_idx=0, source=source, duration=1.0,
                          metric_scores=scores,
                          shot_trims=[(0.5, 1.5)] if source == "retrieval" else [])


def _g():
    return SegmentGuidance(segment_idx=0, parent_section_idx=0,
                           semantic_query="x", editing_heuristic="default",
                           rhythmic_pacing=[2, 2])


def test_composite_reward_returns_breakdown():
    rm = MockRewardModel(accept_threshold=5.0)
    cr = CompositeReward(rm, alpha=1.0, beta=0.5, gamma=0.0)
    res = cr(_seg({f"m{i}": 0.9 for i in range(1, 7)}), _g())
    assert 0.0 <= res.total <= 2.0
    assert res.rm_score > 0
    assert res.metric_mean > 0.8


def test_composite_reward_disagreement_zero_for_single_judge():
    rm = MockRewardModel()
    cr = CompositeReward(rm)
    res = cr(_seg({f"m{i}": 0.5 for i in range(1, 7)}), _g())
    assert res.disagreement == 0.0


def test_editing_quality_rm_trainer_runs_and_saves(sample_preferences_path, tmp_path):
    prefs = PreferenceDataset.from_file(sample_preferences_path)
    trainer = EditingQualityRMTrainer(lr=0.2, n_epochs=3)
    trained = trainer.fit(prefs)
    assert isinstance(trained, EditingQualityRM)
    out = tmp_path / "rm.json"
    trainer.save(out)
    assert out.exists()
    loaded = EditingQualityRM.load(out)
    assert len(loaded.weights) == 6
    # winners have higher metric_scores than losers — accuracy should be > 0.5
    assert trainer.stats.n_correct / max(1, trainer.stats.n_seen) > 0.5


def test_editing_quality_rm_drops_into_base_reward():
    rm = EditingQualityRM(weights=[0.3, 0.2, 0.1, 0.1, 0.2, 0.1], bias=0.0,
                          accept_threshold=0.5)
    high = _seg({f"m{i}": 0.9 for i in range(1, 7)})
    low = _seg({f"m{i}": 0.1 for i in range(1, 7)})
    r_high = rm.score(high, _g())
    r_low = rm.score(low, _g())
    assert r_high.score > r_low.score
    assert r_high.accepted
    assert not r_low.accepted


def test_hindsight_refiner_gamma_zero_leaves_steps_alone():
    refiner = HindsightCriticRefiner(gamma=0.0)
    credits = refiner.refine([1.0, 0.5, -0.2], terminal_reward=10.0)
    for c, r in zip(credits, [1.0, 0.5, -0.2]):
        assert abs(c.refined_reward - r) < 1e-9


def test_hindsight_refiner_gamma_one_pulls_final_step_to_terminal():
    """γ=1, k=N-1 ⇒ mix = (N/N) = 1.0 ⇒ refined = terminal exactly."""
    refiner = HindsightCriticRefiner(gamma=1.0)
    credits = refiner.refine([1.0, 0.5, -0.2], terminal_reward=10.0)
    # final step is pulled fully to terminal
    assert abs(credits[-1].refined_reward - 10.0) < 1e-9
    # later steps borrow more from terminal than earlier ones
    assert credits[0].refined_reward < credits[1].refined_reward < credits[2].refined_reward


def test_hindsight_refiner_critic_downweight():
    refiner = HindsightCriticRefiner(gamma=0.5, critic_downweight=0.1)
    credits = refiner.refine([1.0, 1.0, 1.0], terminal_reward=2.0,
                             critic_flags=[False, True, False])
    assert credits[0].weight == 1.0
    assert abs(credits[1].weight - 0.1) < 1e-9
    assert credits[2].weight == 1.0
