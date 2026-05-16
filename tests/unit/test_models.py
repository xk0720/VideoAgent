"""Model-wrapper unit tests (mock paths)."""
from __future__ import annotations

import json
from pathlib import Path

from longvideoagent.models.llm import MockLLMClient, build_llm_from_alias
from longvideoagent.models.reward import MockRewardModel
from longvideoagent.models.video_gen import MockVideoGenClient, build_video_gen_from_config
from longvideoagent.types import EditingSegment, SegmentGuidance


def test_mock_llm_stubs_distinct_shapes():
    cli = MockLLMClient(alias="director")
    r = cli.chat([{"role": "user", "content": "director — generate a semantic query"}])
    data = json.loads(r.text)
    assert "semantic_query" in data
    # Calling again should rotate the vocabulary to avoid duplicates.
    r2 = cli.chat([{"role": "user", "content": "director — generate a semantic query"}])
    data2 = json.loads(r2.text)
    assert data["semantic_query"] != data2["semantic_query"]


def test_mock_llm_judge_shape():
    cli = MockLLMClient(alias="validator")
    r = cli.chat([{"role": "user", "content": "judge / validator — score this"}])
    data = json.loads(r.text)
    assert {"score", "accepted"} <= set(data)


def test_build_llm_from_alias_returns_mock_when_mocks_on():
    cli = build_llm_from_alias("director", mocks_enabled=True)
    assert isinstance(cli, MockLLMClient)


def test_mock_video_gen_writes_file(tmp_path: Path):
    out = tmp_path / "x.mp4"
    p = MockVideoGenClient().generate("blue sky", duration=1.0, out_path=out)
    assert p.exists()
    assert p.stat().st_size > 0


def test_build_video_gen_factory_mock():
    c = build_video_gen_from_config("omniweaving", mocks_enabled=True)
    assert isinstance(c, MockVideoGenClient)


def test_mock_reward_model_uses_metric_scores():
    rm = MockRewardModel(accept_threshold=5.0)
    s_good = EditingSegment(segment_idx=0, source="retrieval", duration=1.0,
                            metric_scores={f"m{i}": 0.9 for i in range(1, 7)})
    s_bad = EditingSegment(segment_idx=1, source="retrieval", duration=1.0,
                           metric_scores={f"m{i}": 0.1 for i in range(1, 7)})
    g = SegmentGuidance(segment_idx=0, parent_section_idx=0,
                        semantic_query="x", editing_heuristic="default")
    r_good = rm.score(s_good, g)
    r_bad = rm.score(s_bad, g)
    assert r_good.score > r_bad.score
    assert r_good.accepted
    assert not r_bad.accepted
