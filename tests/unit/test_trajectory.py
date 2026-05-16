"""TrajectoryLogger smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from longvideoagent.utils.trajectory import open_trajectory


def test_basic_log_roundtrip(tmp_path: Path):
    p = tmp_path / "trj.jsonl"
    with open_trajectory(p) as log:
        log.log("agent_a", action="step1", action_input={"x": 1}, observation={"ok": True})
        log.log("agent_b", action="step2", reward=0.5)
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["agent_name"] == "agent_a"
    assert rec0["action"] == "step1"
    rec1 = json.loads(lines[1])
    assert rec1["reward"] == 0.5


def test_redacts_large_ndarray(tmp_path: Path):
    p = tmp_path / "trj.jsonl"
    big = np.zeros(1000, dtype="float32")
    with open_trajectory(p, redact_large_tensors=True) as log:
        log.log("agent", action="x", action_input={"feat": big})
    rec = json.loads(p.read_text().splitlines()[0])
    assert rec["action_input"]["feat"]["__ndarray__"] is True
    assert rec["action_input"]["feat"]["shape"] == [1000]


def test_keeps_small_ndarray(tmp_path: Path):
    p = tmp_path / "trj.jsonl"
    small = np.array([1, 2, 3], dtype="float32")
    with open_trajectory(p, redact_large_tensors=True) as log:
        log.log("agent", action="x", action_input={"feat": small})
    rec = json.loads(p.read_text().splitlines()[0])
    assert rec["action_input"]["feat"] == [1.0, 2.0, 3.0]


def test_truncate_mode_overwrites(tmp_path: Path):
    p = tmp_path / "trj.jsonl"
    with open_trajectory(p) as log:
        log.log("a", action="first")
    with open_trajectory(p) as log:                            # default mode='w'
        log.log("a", action="second")
    lines = p.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["action"] == "second"


def test_append_mode_accumulates(tmp_path: Path):
    p = tmp_path / "trj.jsonl"
    with open_trajectory(p) as log:
        log.log("a", action="first")
    with open_trajectory(p, mode="a") as log:
        log.log("a", action="second")
    lines = p.read_text().splitlines()
    assert len(lines) == 2


def test_invalid_mode_raises(tmp_path: Path):
    import pytest
    with pytest.raises(ValueError):
        open_trajectory(tmp_path / "x.jsonl", mode="r").__enter__()
