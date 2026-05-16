"""Verify the trajectory log contains every artefact future RL training will need."""
from __future__ import annotations

import json

import pytest

from longvideoagent.pipeline.run import run_pipeline
from longvideoagent.utils.video_io import write_silent_color_clip


pytestmark = pytest.mark.integration


def test_trajectory_has_per_segment_summary(tmp_path):
    src = write_silent_color_clip(tmp_path / "s.mp4", 4.0, fps=24,
                                  width=160, height=120, color=(80, 100, 200))
    out = tmp_path / "out.mp4"
    traj = tmp_path / "traj.jsonl"
    script = run_pipeline(
        source_videos=[src], user_prompt="quick edit",
        output_path=out, cache_dir=tmp_path / "cache",
        trajectory_log_path=traj,
    )
    records = [json.loads(line) for line in traj.read_text().splitlines()]
    final_entries = [r for r in records if r["action"] == "segment_finalized"]
    # One summary per produced segment.
    assert len(final_entries) == len(script.segments)
    # Every summary must include the full metric vector + a validator score.
    for entry in final_entries:
        obs = entry["observation"]
        assert "metric_scores" in obs
        for k in ("m1", "m2", "m3", "m4", "m5", "m6"):
            assert k in obs["metric_scores"], f"missing {k} in {obs['metric_scores']}"
        assert "accepted" in obs
        assert entry["reward"] is not None
