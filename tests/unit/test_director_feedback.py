"""DirectorAgent consumes Orchestrator feedback on re-plan iterations."""
from __future__ import annotations

import numpy as np

from longvideoagent.agents import DirectorAgent, ScreenwriterAgent
from longvideoagent.memory.builder import build_memory_from_shots
from longvideoagent.memory.retriever import MemoryRetriever
from longvideoagent.memory.store import MemoryStore
from longvideoagent.models.llm import MockLLMClient
from longvideoagent.perception import MusicAnalyzer
from longvideoagent.config import load_config
from longvideoagent.types import CinematographyTags, Shot, ShotFeatures


def _populate(tmp_path):
    store = MemoryStore(tmp_path)
    rng = np.random.default_rng(0)
    shots = []
    for i in range(4):
        emb = rng.standard_normal(64).astype("float32"); emb /= np.linalg.norm(emb) + 1e-9
        shots.append(Shot(shot_id=f"s{i}", source_video="/tmp/x.mp4",
                          start_time=i*2.0, end_time=i*2.0+2,
                          caption="", cinematography=CinematographyTags(),
                          features=ShotFeatures(clip_embedding=emb, avg_flow_magnitude=0.1*i)))
    nm = build_memory_from_shots(shots, store)
    nm.music_profile = MusicAnalyzer(load_config().preprocess, mock=True).analyze(None)
    return store, nm


def test_feedback_reaches_director(tmp_path):
    """The Director's observation log must show it consumed feedback when
    a previous validation cycle produced any."""
    store, nm = _populate(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"))
    plan = sw.run({"memory": nm, "user_prompt": "p"})["global_plan"]
    retr = MemoryRetriever(store, embed_dim=64)

    # Capture log calls.
    captured: list[dict] = []
    class _SpyLogger:
        def log(self, **kw):
            captured.append(kw)

    dr = DirectorAgent(MockLLMClient(alias="director"), retr,
                       trajectory_logger=_SpyLogger())                # type: ignore[arg-type]
    out = dr.run({
        "memory": nm,
        "global_plan": plan,
        "validation": {"passed": False,
                       "feedback": ["Segment 0: too generic", "Segment 1: duplicate"]},
    })
    assert out["segment_guidances"]
    # The trajectory entry should reflect feedback was consumed.
    last = [c for c in captured if c["action"] == "emit_segment_guidances"][-1]
    assert last["observation"]["feedback_consumed"] == 2
    store.close()


def test_no_feedback_means_zero_consumed(tmp_path):
    store, nm = _populate(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"))
    plan = sw.run({"memory": nm, "user_prompt": "p"})["global_plan"]
    retr = MemoryRetriever(store, embed_dim=64)

    captured: list[dict] = []
    class _SpyLogger:
        def log(self, **kw):
            captured.append(kw)

    dr = DirectorAgent(MockLLMClient(alias="director"), retr,
                       trajectory_logger=_SpyLogger())                # type: ignore[arg-type]
    dr.run({"memory": nm, "global_plan": plan})
    last = [c for c in captured if c["action"] == "emit_segment_guidances"][-1]
    assert last["observation"]["feedback_consumed"] == 0
    store.close()
