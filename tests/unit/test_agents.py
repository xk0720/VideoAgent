"""Agent unit tests under the mock LLM."""
from __future__ import annotations

import numpy as np

from longvideoagent.agents import (
    DirectorAgent, OrchestratorAgent, ScreenwriterAgent, ValidatorAgent,
)
from longvideoagent.memory.builder import build_memory_from_shots
from longvideoagent.memory.retriever import MemoryRetriever
from longvideoagent.memory.store import MemoryStore
from longvideoagent.models.llm import MockLLMClient
from longvideoagent.models.reward import MockRewardModel
from longvideoagent.perception import MusicAnalyzer
from longvideoagent.config import load_config
from longvideoagent.types import CinematographyTags, Shot, ShotFeatures


def _populate_store(tmp_path):
    store = MemoryStore(tmp_path)
    rng = np.random.default_rng(0)
    shots = []
    for i in range(4):
        emb = rng.standard_normal(64).astype("float32"); emb /= np.linalg.norm(emb) + 1e-9
        shots.append(Shot(shot_id=f"s{i}", source_video="/tmp/x.mp4",
                          start_time=i*2.0, end_time=i*2.0+2,
                          caption=f"shot {i}", cinematography=CinematographyTags(),
                          features=ShotFeatures(clip_embedding=emb, avg_flow_magnitude=0.1*i)))
    nm = build_memory_from_shots(shots, store)
    nm.music_profile = MusicAnalyzer(load_config().preprocess, mock=True).analyze(None)
    return store, nm


def test_screenwriter_emits_plan(tmp_path):
    store, nm = _populate_store(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"))
    out = sw.run({"memory": nm, "user_prompt": "energetic montage"})
    assert "global_plan" in out
    assert len(out["global_plan"].section_plans) >= 1
    store.close()


def test_director_emits_guidances(tmp_path):
    store, nm = _populate_store(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"))
    plan = sw.run({"memory": nm, "user_prompt": "p"})["global_plan"]
    retr = MemoryRetriever(store, embed_dim=64)
    dr = DirectorAgent(MockLLMClient(alias="director"), retr)
    out = dr.run({"memory": nm, "global_plan": plan})
    assert "segment_guidances" in out
    assert len(out["segment_guidances"]) >= 1
    g = out["segment_guidances"][0]
    assert g.semantic_query
    assert 0.0 <= g.retrieval_feasibility <= 1.0
    store.close()


def test_orchestrator_validates(tmp_path):
    store, nm = _populate_store(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"))
    plan = sw.run({"memory": nm, "user_prompt": "p"})["global_plan"]
    retr = MemoryRetriever(store, embed_dim=64)
    dr = DirectorAgent(MockLLMClient(alias="director"), retr)
    guidances = dr.run({"memory": nm, "global_plan": plan})["segment_guidances"]
    ortr = OrchestratorAgent(MockLLMClient(alias="orchestrator"), retr)
    out = ortr.run({"memory": nm, "global_plan": plan, "segment_guidances": guidances})
    assert "validation" in out
    assert isinstance(out["validation"]["feedback"], list)
    store.close()


def test_validator_scoring():
    from longvideoagent.types import EditingSegment, SegmentGuidance
    seg = EditingSegment(segment_idx=0, source="retrieval", duration=2.0,
                         metric_scores={"m1": 0.9, "m2": 0.8, "m3": 0.7,
                                        "m4": 0.6, "m5": 0.7, "m6": 0.7})
    g = SegmentGuidance(segment_idx=0, parent_section_idx=0,
                        semantic_query="x", editing_heuristic="default")
    v = ValidatorAgent(MockRewardModel(accept_threshold=5.0))
    r = v.score(seg, g)
    assert 1.0 <= r.score <= 10.0
    assert r.accepted
