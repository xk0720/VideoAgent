"""End-to-end smoke test.

Runs the full preprocess → plan → compose pipeline against the auto-generated
``tests/fixtures/tiny_clip.mp4`` fixture in mock-everything mode, and verifies:
    1. The pipeline returns an EditingScript with >=1 segment.
    2. A real .mp4 lands at the expected output path.
    3. A non-empty trajectory.jsonl is written.
"""
from __future__ import annotations

import json

import pytest

from longvideoagent.pipeline.run import run_pipeline


pytestmark = pytest.mark.integration


def test_full_pipeline_with_tiny_clip(tmp_path, tiny_clip_path):
    cache_dir = tmp_path / "cache"
    output = tmp_path / "output.mp4"
    traj = tmp_path / "trajectory.jsonl"

    script = run_pipeline(
        source_videos=[tiny_clip_path],
        user_prompt="Make a short montage with high energy",
        output_path=output,
        music=None,
        cache_dir=cache_dir,
        trajectory_log_path=traj,
    )

    assert script.segments, "pipeline produced no segments"
    assert output.exists(), f"output mp4 missing at {output}"
    assert output.stat().st_size > 0
    assert traj.exists()
    lines = traj.read_text().splitlines()
    assert lines, "trajectory.jsonl empty"
    # The log must be valid JSONL.
    for ln in lines:
        rec = json.loads(ln)
        assert "agent_name" in rec
        assert "action" in rec
