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
