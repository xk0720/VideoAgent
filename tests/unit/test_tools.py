"""Tool-layer unit tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from longvideoagent.config import load_config
from longvideoagent.memory.builder import build_memory_from_shots
from longvideoagent.memory.retriever import MemoryRetriever
from longvideoagent.memory.store import MemoryStore
from longvideoagent.models.video_gen import MockVideoGenClient
from longvideoagent.perception import MusicAnalyzer
from longvideoagent.tools import AssemblyTool, GenerationTool, RetrievalTool
from longvideoagent.tools.metric_tool import (
    m1_prompt_relevance, m2_segment_consistency, m3_motion_continuity,
    m4_framing, m5_beat_sync, m6_energy_correspondence,
)
from longvideoagent.types import (
    CinematographyTags, EditingScript, EditingSegment, SegmentGuidance, Shot, ShotFeatures,
)


def test_metric_tool_basic():
    a = np.array([1.0, 0.0, 0.0], dtype="float32")
    b = np.array([1.0, 0.0, 0.0], dtype="float32")
    assert abs(m1_prompt_relevance(a, b) - 1.0) < 1e-3
    assert abs(m2_segment_consistency(None, a) - 1.0) < 1e-3
    assert 0.0 <= m3_motion_continuity(None, None) <= 1.0
    sal = np.ones((4, 4), dtype="float32"); sal /= sal.sum()
    assert 0.0 <= m4_framing(sal, sal) <= 1.0
    assert 0.0 <= m5_beat_sync(0.5, [0.5, 1.0]) <= 1.0
    assert 0.0 <= m6_energy_correspondence([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) <= 1.0


def _store_with_shots(tmp_path):
    store = MemoryStore(tmp_path)
    rng = np.random.default_rng(7)
    shots = []
    for i in range(6):
        emb = rng.standard_normal(64).astype("float32"); emb /= np.linalg.norm(emb) + 1e-9
        flow = rng.standard_normal((8, 8, 2)).astype("float32")
        sal = np.ones((8, 8), dtype="float32") / 64.0
        shots.append(Shot(shot_id=f"s{i}", source_video="/tmp/x.mp4",
                          start_time=i*2.0, end_time=i*2.0+2,
                          caption="", cinematography=CinematographyTags(),
                          features=ShotFeatures(
                              clip_embedding=emb, start_flow=flow, end_flow=flow,
                              start_saliency=sal, end_saliency=sal, avg_flow_magnitude=0.5)))
    nm = build_memory_from_shots(shots, store)
    nm.music_profile = MusicAnalyzer(load_config().preprocess, mock=True).analyze(None)
    return store, nm


def test_retrieval_tool_returns_segment(tmp_path):
    store, nm = _store_with_shots(tmp_path)
    retriever = MemoryRetriever(store, embed_dim=64)
    rt = RetrievalTool(retriever, beam_width=2, top_k_pool=6)
    g = SegmentGuidance(segment_idx=0, parent_section_idx=0,
                        semantic_query="abc", editing_heuristic="default",
                        rhythmic_pacing=[2, 2])
    seg = rt.run(g, nm)
    assert seg is not None
    assert seg.source == "retrieval"
    assert len(seg.shot_ids) == 2
    assert {"m1", "m2", "m3", "m4", "m5", "m6"} <= set(seg.metric_scores)
    store.close()


def test_generation_tool_writes_file(tmp_path):
    g = SegmentGuidance(segment_idx=0, parent_section_idx=0,
                        semantic_query="abc", editing_heuristic="default")
    tool = GenerationTool(MockVideoGenClient(), default_duration_s=1.0)
    seg = tool.run(g, output_dir=tmp_path)
    assert seg.source == "generation"
    assert Path(seg.gen_video_path).exists()


def test_assembly_tool_emits_mp4(tmp_path, tiny_clip_path):
    # Build a simple script with two retrieval segments referencing the fixture clip.
    seg1 = EditingSegment(segment_idx=0, source="retrieval", duration=1.0,
                          shot_ids=["s0"], shot_trims=[(0.0, 1.0)],
                          source_videos=[str(tiny_clip_path)])
    seg2 = EditingSegment(segment_idx=1, source="retrieval", duration=1.0,
                          shot_ids=["s1"], shot_trims=[(1.0, 2.0)],
                          source_videos=[str(tiny_clip_path)])
    script = EditingScript(segments=[seg1, seg2], total_duration=2.0)
    out = tmp_path / "out.mp4"
    AssemblyTool().run(script, out)
    assert out.exists()
    assert out.stat().st_size > 0
