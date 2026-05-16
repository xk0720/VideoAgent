"""CriticAgent rule-triggered Lesson emission."""
from __future__ import annotations

import json
from pathlib import Path

from longvideoagent.agents.critic import CriticAgent
from longvideoagent.memory.lessons import LessonBook


def _write_trajectory(path: Path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_low_reward_trigger(tmp_path: Path):
    traj = tmp_path / "t.jsonl"
    _write_trajectory(traj, [{
        "timestamp": 1.0, "agent_name": "editor", "action": "segment_finalized",
        "action_input": {"segment_idx": 0, "semantic_query": "vague", "heuristic": "default"},
        "observation": {"metric_scores": {"m1": 0.4, "m2": 0.4, "m3": 0.4,
                                          "m4": 0.4, "m5": 0.4, "m6": 0.4,
                                          "validator": 4.0}},
        "reward": 4.0,
    }])
    book = LessonBook(tmp_path / "lessons.jsonl")
    out = CriticAgent(book, low_reward_threshold=6.0).review(traj, user_prompt="abc")
    assert any(L.trigger == "low_reward" for L in out)


def test_duplicate_query_trigger(tmp_path: Path):
    traj = tmp_path / "t.jsonl"
    _write_trajectory(traj, [
        {"timestamp": 1.0, "agent_name": "editor", "action": "segment_finalized",
         "action_input": {"segment_idx": 0, "semantic_query": "same q", "heuristic": "x"},
         "observation": {"metric_scores": {}}, "reward": 8.0},
        {"timestamp": 2.0, "agent_name": "editor", "action": "segment_finalized",
         "action_input": {"segment_idx": 1, "semantic_query": "same q", "heuristic": "x"},
         "observation": {"metric_scores": {}}, "reward": 8.0},
    ])
    book = LessonBook(tmp_path / "lessons.jsonl")
    out = CriticAgent(book).review(traj)
    assert any(L.trigger == "duplicate_query" for L in out)


def test_disagreement_trigger(tmp_path: Path):
    traj = tmp_path / "t.jsonl"
    _write_trajectory(traj, [{
        "timestamp": 1.0, "agent_name": "editor", "action": "segment_finalized",
        "action_input": {"segment_idx": 0, "semantic_query": "x", "heuristic": "d"},
        "observation": {"metric_scores": {f"m{i}": 0.9 for i in range(1, 7)} | {"validator": 3.0}},
        "reward": 3.0,
    }])
    book = LessonBook(tmp_path / "lessons.jsonl")
    out = CriticAgent(book, disagreement_threshold=1.0).review(traj)
    # judge=3.0, metric_mean = 0.9*10*6/6 = 9.0, diff=6 > 1.0 ⇒ trigger
    assert any(L.trigger == "disagreement" for L in out)


def test_empty_trajectory_yields_no_lessons(tmp_path: Path):
    traj = tmp_path / "t.jsonl"
    traj.write_text("")
    book = LessonBook(tmp_path / "lessons.jsonl")
    assert CriticAgent(book).review(traj) == []
