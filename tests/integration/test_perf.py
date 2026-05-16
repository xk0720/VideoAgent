"""Performance smoke test.

Pins the v0.1 mock-pipeline runtime to a sane upper bound; the goal is to
catch O(N²) regressions and accidental real-model calls, not to be a
precise benchmark.
"""
from __future__ import annotations

import time

import pytest

from longvideoagent.pipeline.run import run_pipeline
from longvideoagent.utils.video_io import write_silent_color_clip


pytestmark = pytest.mark.integration


def test_end_to_end_under_5_seconds(tmp_path):
    src = write_silent_color_clip(tmp_path / "s.mp4", duration_s=4.0,
                                  fps=24, width=160, height=120, color=(20, 60, 200))
    cache = tmp_path / "cache"
    out = tmp_path / "out.mp4"
    traj = tmp_path / "traj.jsonl"

    t0 = time.perf_counter()
    script = run_pipeline(
        source_videos=[src],
        user_prompt="A short energetic edit",
        output_path=out,
        cache_dir=cache,
        trajectory_log_path=traj,
    )
    elapsed = time.perf_counter() - t0

    assert script.segments, "pipeline produced no segments"
    assert out.exists() and out.stat().st_size > 0
    # 5 s upper bound: the typical mock run is ~1.5s on this CI; if this
    # ever fails, look for accidental real-model calls or import-time slow paths.
    assert elapsed < 5.0, f"pipeline took {elapsed:.2f}s, expected < 5s"
