"""Tests for RetrievalTool._slide_trim sliding-window enumeration."""
from __future__ import annotations

from longvideoagent.memory.store import MemoryStore
from longvideoagent.memory.retriever import MemoryRetriever
from longvideoagent.tools.retrieval_tool import RetrievalTool, _Beam
from longvideoagent.types import CinematographyTags, Shot, ShotFeatures

import numpy as np


def _make_shot(start: float, end: float, mag: float = 0.5) -> Shot:
    emb = np.random.default_rng(0).standard_normal(32).astype("float32")
    return Shot(
        shot_id="s", source_video="/tmp/x.mp4",
        start_time=start, end_time=end,
        caption="", cinematography=CinematographyTags(),
        features=ShotFeatures(clip_embedding=emb, avg_flow_magnitude=mag),
    )


def test_slide_trim_short_shot_returns_full_shot(tmp_path):
    store = MemoryStore(tmp_path)
    retr = MemoryRetriever(store, embed_dim=32)
    rt = RetrievalTool(retr, sliding_stride=0.5)
    shot = _make_shot(0.0, 1.0)
    window = rt._slide_trim(shot, slot_duration=2.0)
    assert window == (0.0, 1.0)
    store.close()


def test_slide_trim_enumerates_multiple_candidates(tmp_path):
    store = MemoryStore(tmp_path)
    retr = MemoryRetriever(store, embed_dim=32)
    rt = RetrievalTool(retr, sliding_stride=0.5)
    shot = _make_shot(0.0, 5.0, mag=0.5)
    window = rt._slide_trim(shot, slot_duration=2.0, beats=[1.0, 2.0, 3.0, 4.0])
    # The window must be inside the shot bounds and have the right length.
    a, b = window
    assert 0.0 <= a <= 3.0
    assert abs((b - a) - 2.0) < 1e-3


def test_slide_trim_motion_continuity_prefers_similar_flow(tmp_path):
    store = MemoryStore(tmp_path)
    retr = MemoryRetriever(store, embed_dim=32)
    rt = RetrievalTool(retr, sliding_stride=0.5)
    # Build a beam whose tail has avg_flow_magnitude=0.5; the candidate shot
    # also has 0.5 so the motion term should be 1.0 (the best possible).
    tail = ShotFeatures(clip_embedding=np.zeros(32, dtype="float32"),
                        avg_flow_magnitude=0.5)
    beam = _Beam(tail_features=tail)
    shot = _make_shot(0.0, 5.0, mag=0.5)
    # Choose a beat list that prefers t=2.0 (which is one valid window start).
    window = rt._slide_trim(shot, slot_duration=2.0, beam=beam, beats=[2.0])
    assert window[0] == 2.0
    store.close()
