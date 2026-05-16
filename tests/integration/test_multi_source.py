"""Integration: confirm that multi-source input is actually exploited.

The script's retrieval beam must end up pulling shots from BOTH source
videos (with high probability) — otherwise the multi-source advantage
advertised in the design doc isn't being realised.
"""
from __future__ import annotations

import pytest

from longvideoagent.pipeline.run import run_pipeline
from longvideoagent.utils.video_io import write_silent_color_clip


pytestmark = pytest.mark.integration


def test_two_sources_are_both_referenced(tmp_path):
    a = write_silent_color_clip(tmp_path / "a.mp4", duration_s=4.0,
                                fps=24, width=160, height=120, color=(200, 30, 30))
    b = write_silent_color_clip(tmp_path / "b.mp4", duration_s=4.0,
                                fps=24, width=160, height=120, color=(30, 200, 30))

    out = tmp_path / "out.mp4"
    traj = tmp_path / "traj.jsonl"
    script = run_pipeline(
        source_videos=[a, b],
        user_prompt="A varied two-clip montage",
        output_path=out,
        cache_dir=tmp_path / "cache",
        trajectory_log_path=traj,
    )

    assert out.exists()
    sources_used: set[str] = set()
    for seg in script.segments:
        for src in seg.source_videos:
            sources_used.add(str(src))
    # Each source video should appear in the final timeline at least once.
    # (Beam search ranks across the full pool of shots from both videos, so
    # both being present is the expected outcome.)
    assert str(a) in sources_used, f"video A not referenced: {sources_used}"
    assert str(b) in sources_used, f"video B not referenced: {sources_used}"
