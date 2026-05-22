"""Fixtures shared by training/* tests."""
from __future__ import annotations

# Make ``training.*`` importable. The project keeps ``training/`` as a
# sibling of ``src/`` (not inside it), so pytest needs the project root on
# ``sys.path``. We do it here rather than via pyproject ``pythonpath``
# because the latter wasn't being honoured in some pytest invocations.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import json
from typing import Any

import numpy as np
import pytest

from longvideoagent.config import load_config
from longvideoagent.memory.builder import build_memory_from_shots
from longvideoagent.memory.retriever import MemoryRetriever
from longvideoagent.memory.store import MemoryStore
from longvideoagent.models.llm import MockLLMClient
from longvideoagent.models.reward.base import MockRewardModel
from longvideoagent.models.video_gen import MockVideoGenClient
from longvideoagent.perception import MusicAnalyzer
from longvideoagent.tools import GenerationTool, RetrievalTool
from longvideoagent.types import (
    CinematographyTags, SegmentGuidance, Shot, ShotFeatures,
)


@pytest.fixture
def populated_store(tmp_path):
    store = MemoryStore(tmp_path / "cache")
    rng = np.random.default_rng(0)
    shots = []
    for i in range(6):
        emb = rng.standard_normal(64).astype("float32")
        emb /= np.linalg.norm(emb) + 1e-9
        flow = rng.standard_normal((8, 8, 2)).astype("float32")
        sal = np.ones((8, 8), dtype="float32") / 64.0
        shots.append(Shot(
            shot_id=f"s{i}", source_video="/tmp/x.mp4",
            start_time=i * 2.0, end_time=i * 2.0 + 2.0,
            caption="", cinematography=CinematographyTags(),
            features=ShotFeatures(
                clip_embedding=emb, start_flow=flow, end_flow=flow,
                start_saliency=sal, end_saliency=sal, avg_flow_magnitude=0.3),
        ))
    nm = build_memory_from_shots(shots, store)
    nm.music_profile = MusicAnalyzer(load_config().preprocess, mock=True).analyze(None)
    return store, nm


@pytest.fixture
def sample_guidances():
    return [
        SegmentGuidance(segment_idx=0, parent_section_idx=0,
                        semantic_query="opening shot",
                        editing_heuristic="default", rhythmic_pacing=[2, 2]),
        SegmentGuidance(segment_idx=1, parent_section_idx=1,
                        semantic_query="action moment",
                        editing_heuristic="motion_continuity",
                        rhythmic_pacing=[1, 1, 1]),
    ]


@pytest.fixture
def sample_trajectory_path(tmp_path) -> Path:
    """Write a minimal trajectory.jsonl with some high- and low-reward segments."""
    records: list[dict[str, Any]] = [
        {"timestamp": 1.0, "agent_name": "editor", "action": "retrieve",
         "action_input": {"segment_idx": 0, "step": 1},
         "observation": {"n_candidates": 1}, "reward": None,
         "state_snapshot": {}, "extra": {}},
        {"timestamp": 2.0, "agent_name": "editor", "action": "segment_finalized",
         "action_input": {"segment_idx": 0, "semantic_query": "opening shot",
                           "heuristic": "default"},
         "observation": {"source": "retrieval", "accepted": True,
                          "metric_scores": {f"m{i}": 0.8 for i in range(1, 7)} | {"validator": 8.5},
                          "n_shot_ids": 2, "duration": 2.0},
         "reward": 8.5, "state_snapshot": {}, "extra": {}},
        {"timestamp": 3.0, "agent_name": "editor", "action": "segment_finalized",
         "action_input": {"segment_idx": 1, "semantic_query": "action moment",
                           "heuristic": "motion_continuity"},
         "observation": {"source": "generation", "accepted": False,
                          "metric_scores": {f"m{i}": 0.3 for i in range(1, 7)} | {"validator": 4.0},
                          "n_shot_ids": 0, "duration": 2.0},
         "reward": 4.0, "state_snapshot": {}, "extra": {}},
    ]
    p = tmp_path / "trajectory.jsonl"
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def sample_preferences_path(tmp_path) -> Path:
    """Write a minimal preferences.jsonl with 3 (winner, loser) records."""
    records = []
    for i in range(3):
        records.append({
            "ts": float(i), "run_id": "test",
            "segment_idx": i,
            "guidance": {"segment_idx": i,
                          "semantic_query": f"shot {i}",
                          "editing_heuristic": "default",
                          "rhythmic_pacing": [2, 2],
                          "retrieval_feasibility": 0.8},
            "winner": {"source": "retrieval",
                        "metric_scores": {f"m{j}": 0.8 for j in range(1, 7)},
                        "shot_ids": [f"s{i}"], "shot_trims": [[0.0, 1.0]],
                        "gen_prompt": None, "accepted_by_validator": True,
                        "validator_reasons": []},
            "losers": [{"source": "generation",
                         "metric_scores": {f"m{j}": 0.3 for j in range(1, 7)},
                         "shot_ids": [], "shot_trims": [],
                         "gen_prompt": "fake", "accepted_by_validator": False,
                         "validator_reasons": []}],
            "judge": "MockRewardModel",
        })
    p = tmp_path / "preferences.jsonl"
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def editor_env_factory(populated_store, sample_guidances):
    store, nm = populated_store
    retriever = MemoryRetriever(store, embed_dim=64)
    cfg = load_config()
    rt = RetrievalTool(retriever,
                       beam_width=cfg.compose.retrieval.beam_width,
                       top_k_pool=8)
    gt = GenerationTool(MockVideoGenClient(), default_duration_s=1.0)
    rm = MockRewardModel(accept_threshold=5.0)

    def _factory():
        from training.env.editor_env import EditorEnv
        return EditorEnv(memory=nm, guidances=sample_guidances,
                         retrieval_tool=rt, generation_tool=gt,
                         reward_model=rm, max_steps=4)
    yield _factory
    store.close()


@pytest.fixture
def mock_policy():
    from training.policy.editor_policy import EditorAgentPolicy
    return EditorAgentPolicy(MockLLMClient(alias="editor"))
