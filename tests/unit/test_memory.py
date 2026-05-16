"""MemoryStore basic CRUD + retrieval tests."""
from __future__ import annotations

import numpy as np

from longvideoagent.memory.builder import build_memory_from_shots
from longvideoagent.memory.retriever import MemoryRetriever
from longvideoagent.memory.store import MemoryStore
from longvideoagent.types import CinematographyTags, Shot, ShotFeatures


def _make_shot(i: int) -> Shot:
    rng = np.random.default_rng(i)
    emb = rng.standard_normal(64).astype("float32")
    emb /= np.linalg.norm(emb) + 1e-9
    return Shot(
        shot_id=f"s{i:03d}", source_video=f"/tmp/v{i}.mp4",
        start_time=float(i), end_time=float(i + 2),
        caption=f"shot {i}", cinematography=CinematographyTags(),
        features=ShotFeatures(clip_embedding=emb, avg_flow_magnitude=0.1 * i),
    )


def test_store_roundtrip(tmp_path):
    store = MemoryStore(tmp_path)
    shots = [_make_shot(i) for i in range(4)]
    build_memory_from_shots(shots, store)
    nm = store.load_full_memory(load_features=True)
    assert len(nm.shots) == 4
    assert nm.shots["s000"].caption == "shot 0"
    assert nm.shots["s000"].features is not None
    assert nm.shots["s000"].features.clip_embedding.shape == (64,)
    assert len(nm.events) >= 1
    assert len(nm.stories) >= 1
    store.close()


def test_retriever_basic(tmp_path):
    store = MemoryStore(tmp_path)
    shots = [_make_shot(i) for i in range(8)]
    build_memory_from_shots(shots, store)
    retr = MemoryRetriever(store, embed_dim=64)
    hits = retr.retrieve_by_query("anything", top_k=5)
    assert 1 <= len(hits) <= 5
    feas = retr.estimate_feasibility("anything")
    assert 0.0 <= feas <= 1.0
    store.close()
