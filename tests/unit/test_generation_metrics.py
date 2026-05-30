"""Tests for the dynamic metric_scores GenerationTool now emits."""
from __future__ import annotations

from pathlib import Path

from longvideoagent.models.video_gen import MockVideoGenClient
from longvideoagent.tools import GenerationTool
from longvideoagent.types import SegmentGuidance


def _g():
    return SegmentGuidance(
        segment_idx=0, parent_section_idx=0,
        semantic_query="a sunset shot", editing_heuristic="default",
        rhythmic_pacing=[4], cinematography_hints=["wide", "static"],
    )


def test_first_frame_anchor_lifts_m2(tmp_path: Path):
    tool = GenerationTool(MockVideoGenClient(), default_duration_s=1.0)
    no_anchor = tool.run(_g(), neighbor_context={"end_frame": None}, output_dir=tmp_path / "no")
    with_anchor = tool.run(_g(), neighbor_context={"end_frame": object()}, output_dir=tmp_path / "yes")
    assert with_anchor.metric_scores["m2"] > no_anchor.metric_scores["m2"]


def test_flow_condition_lifts_m3(tmp_path: Path):
    tool = GenerationTool(MockVideoGenClient(), default_duration_s=1.0)
    no_flow = tool.run(_g(), neighbor_context={}, output_dir=tmp_path / "no")
    with_flow = tool.run(_g(), neighbor_context={"end_flow": object()}, output_dir=tmp_path / "yes")
    assert with_flow.metric_scores["m3"] > no_flow.metric_scores["m3"]


def test_beat_sync_m5_responds_to_beats(tmp_path: Path):
    tool = GenerationTool(MockVideoGenClient(), default_duration_s=1.0)
    off_beat = tool.run(_g(), neighbor_context={
        "beats": [1.0, 2.0, 3.0], "expected_start_time_s": 1.5,
    }, output_dir=tmp_path / "off")
    on_beat = tool.run(_g(), neighbor_context={
        "beats": [1.0, 2.0, 3.0], "expected_start_time_s": 2.0,
    }, output_dir=tmp_path / "on")
    assert on_beat.metric_scores["m5"] > off_beat.metric_scores["m5"]


def test_previous_source_modulates_m2_m3(tmp_path: Path):
    """v0.2 honesty fix (docs/BASELINE_v0_2.md): a generated segment chained
    after another generated segment must score lower on m2/m3 than one
    chained after a real retrieval — because the anchor is itself synthetic.
    Without this, R→G and G→G look identical and the hybrid claim is
    untestable in mock mode (which is what the v0.2 baseline measurement
    needs to discriminate)."""
    tool = GenerationTool(MockVideoGenClient(), default_duration_s=1.0)
    anchors = {
        "end_frame": object(),                                # anchor inputs are present
        "end_flow": object(),
        "character_anchors": ["char_0"],
    }
    r_to_g = tool.run(_g(),
                      neighbor_context={**anchors, "previous_source": "retrieval"},
                      output_dir=tmp_path / "r_to_g")
    g_to_g = tool.run(_g(),
                      neighbor_context={**anchors, "previous_source": "generation"},
                      output_dir=tmp_path / "g_to_g")
    nothing_to_g = tool.run(_g(),
                            neighbor_context={**anchors, "previous_source": None},
                            output_dir=tmp_path / "none_to_g")
    assert r_to_g.metric_scores["m2"] > g_to_g.metric_scores["m2"] > nothing_to_g.metric_scores["m2"]
    assert r_to_g.metric_scores["m3"] > g_to_g.metric_scores["m3"] > nothing_to_g.metric_scores["m3"]


def test_character_refs_lift_m4(tmp_path: Path):
    """``has_refs`` was computed but unused in the v0.1 _estimate_metrics —
    the v0.2 honesty fix threads it through and lets it lift m4 (framing)."""
    tool = GenerationTool(MockVideoGenClient(), default_duration_s=1.0)
    with_refs = tool.run(_g(),
                         neighbor_context={"character_anchors": ["char_0", "char_1"]},
                         output_dir=tmp_path / "with")
    without_refs = tool.run(_g(),
                            neighbor_context={"character_anchors": []},
                            output_dir=tmp_path / "without")
    assert with_refs.metric_scores["m4"] > without_refs.metric_scores["m4"]
