"""EnsembleRewardModel multi-judge tests."""
from __future__ import annotations

from longvideoagent.models.reward import (
    EnsembleResult, EnsembleRewardModel, MockRewardModel,
)
from longvideoagent.types import EditingSegment, SegmentGuidance


def _g():
    return SegmentGuidance(segment_idx=0, parent_section_idx=0,
                           semantic_query="x", editing_heuristic="default")


def _seg(scores):
    return EditingSegment(segment_idx=0, source="retrieval", duration=1.0,
                          metric_scores=scores)


def test_ensemble_returns_mean_and_disagreement():
    j_balanced = MockRewardModel(weights={"m1": 0.20, "m2": 0.15, "m3": 0.20,
                                          "m4": 0.15, "m5": 0.15, "m6": 0.15})
    j_semantic = MockRewardModel(weights={"m1": 0.80, "m2": 0.04, "m3": 0.04,
                                          "m4": 0.04, "m5": 0.04, "m6": 0.04})
    ens = EnsembleRewardModel([j_balanced, j_semantic])

    # candidate is m1-strong, others weak — judges should disagree.
    seg = _seg({"m1": 0.95, "m2": 0.1, "m3": 0.1, "m4": 0.1, "m5": 0.1, "m6": 0.1})
    res = ens.score(seg, _g())
    assert isinstance(res, EnsembleResult)
    assert len(res.per_judge_scores) == 2
    assert res.disagreement > 0.5    # m1-heavy judge gives much higher score


def test_quorum_blocks_low_consensus():
    j1 = MockRewardModel(accept_threshold=6.0)
    j2 = MockRewardModel(accept_threshold=6.0, weights={"m1": 1.0, "m2": 0.0, "m3": 0.0,
                                                         "m4": 0.0, "m5": 0.0, "m6": 0.0})
    j3 = MockRewardModel(accept_threshold=6.0, weights={"m3": 1.0, "m1": 0.0, "m2": 0.0,
                                                         "m4": 0.0, "m5": 0.0, "m6": 0.0})
    ens = EnsembleRewardModel([j1, j2, j3], accept_threshold=6.0)
    # only one judge accepts.
    seg = _seg({"m1": 0.95, "m2": 0.0, "m3": 0.0, "m4": 0.0, "m5": 0.0, "m6": 0.0})
    res = ens.score(seg, _g())
    # ensemble mean is below threshold AND quorum is not met → reject.
    assert not res.accepted


def test_active_learning_flag():
    j1 = MockRewardModel(weights={"m1": 1.0, "m2": 0.0, "m3": 0.0,
                                  "m4": 0.0, "m5": 0.0, "m6": 0.0})
    j2 = MockRewardModel(weights={"m3": 1.0, "m1": 0.0, "m2": 0.0,
                                  "m4": 0.0, "m5": 0.0, "m6": 0.0})
    ens = EnsembleRewardModel([j1, j2])
    seg = _seg({"m1": 0.95, "m2": 0.0, "m3": 0.0, "m4": 0.0, "m5": 0.0, "m6": 0.0})
    res = ens.score(seg, _g())
    assert res.is_active_learning_candidate(threshold=1.0)
