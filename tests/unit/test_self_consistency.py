"""ScreenwriterAgent self-consistency tests."""
from __future__ import annotations

import numpy as np

from longvideoagent.agents import ScreenwriterAgent
from longvideoagent.memory.builder import build_memory_from_shots
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
                          features=ShotFeatures(clip_embedding=emb, avg_flow_magnitude=0.0)))
    nm = build_memory_from_shots(shots, store)
    nm.music_profile = MusicAnalyzer(load_config().preprocess, mock=True).analyze(None)
    return store, nm


def test_self_consistency_k_eq_1_is_backwards_compatible(tmp_path):
    store, nm = _populate(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"), self_consistency_k=1)
    out = sw.run({"memory": nm, "user_prompt": "x"})
    assert len(out["global_plan"].section_plans) >= 1
    store.close()


def test_self_consistency_k_gt_1_still_emits_one_plan(tmp_path):
    store, nm = _populate(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"), self_consistency_k=3)
    out = sw.run({"memory": nm, "user_prompt": "x"})
    plan = out["global_plan"]
    assert len(plan.section_plans) >= 1
    # The aggregated rationale must mention sample count.
    assert any("self-consistency" in sp.rationale for sp in plan.section_plans)
    store.close()


def test_invalid_k_raises():
    import pytest
    with pytest.raises(ValueError):
        ScreenwriterAgent(MockLLMClient(alias="screenwriter"), self_consistency_k=0)


def test_lessons_get_appended_to_prompt(tmp_path):
    """Smoke: when lessons are passed in state, the prompt grows."""
    from longvideoagent.memory.lessons import Lesson
    store, nm = _populate(tmp_path)
    sw = ScreenwriterAgent(MockLLMClient(alias="screenwriter"))

    captured: dict = {}
    original_chat = sw.llm.chat
    def spy(messages, **kw):
        captured["last_prompt"] = messages[-1]["content"]
        return original_chat(messages, **kw)
    sw.llm.chat = spy                                            # type: ignore[method-assign]

    sw.run({"memory": nm, "user_prompt": "x", "lessons_for_screenwriter": [
        Lesson(lesson_id="a", created_at=0.0, trigger="x", scope="screenwriter",
               lesson="Avoid generic queries."),
    ]})
    assert "Lessons from previous runs" in captured["last_prompt"]
    assert "Avoid generic queries." in captured["last_prompt"]
    store.close()
