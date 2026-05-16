"""CLI smoke tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from longvideoagent.scripts_impl import (
    build_memory_main,
    eval_main,
    preprocess_main,
    viz_trajectory_main,
)


def test_preprocess_cli(monkeypatch, tmp_path: Path, tiny_clip_path: Path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv",
                        ["lva-preprocess", "--source", str(tiny_clip_path),
                         "--cache-dir", str(cache)])
    assert preprocess_main() == 0
    assert (cache / "memory.sqlite").exists()


def test_build_memory_cli(monkeypatch, capsys, tmp_path: Path, tiny_clip_path: Path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["lva-preprocess", "--source", str(tiny_clip_path),
                                       "--cache-dir", str(cache)])
    preprocess_main()
    monkeypatch.setattr(sys, "argv", ["lva-build-memory", "--cache-dir", str(cache)])
    assert build_memory_main() == 0
    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["n_shots"] >= 1


def test_eval_cli_stub(monkeypatch, tmp_path: Path):
    out_dir = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["lva-eval", "--benchmark", "mashup-bench",
                                       "--output-dir", str(out_dir)])
    assert eval_main() == 0
    assert (out_dir / "summary.json").exists()


def test_viz_trajectory(monkeypatch, capsys, tmp_path: Path):
    log = tmp_path / "t.jsonl"
    log.write_text(json.dumps({
        "timestamp": 1.0, "agent_name": "a", "action": "x",
        "action_input": {}, "observation": {}, "reward": 0.5,
    }) + "\n")
    monkeypatch.setattr(sys, "argv", ["lva-viz", "--log", str(log)])
    assert viz_trajectory_main() == 0
