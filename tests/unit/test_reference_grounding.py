"""Reference-grounding regression test.

Per AUDIT_REPORT.md (Round 10), every core source file must keep its
top-docstring citations. If someone later strips a reference (e.g. while
refactoring), this test fails — making reference drift impossible to land
without a deliberate change to this file.
"""
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

# (relative path, must-contain substrings — at least one prior-work cite per file)
EXPECTED_CITATIONS: list[tuple[str, list[str]]] = [
    ("src/longvideoagent/tools/retrieval_tool.py",       ["DIRECT"]),
    ("src/longvideoagent/tools/metric_tool.py",          ["DIRECT"]),
    ("src/longvideoagent/agents/screenwriter.py",        ["DIRECT", "Self-Consistency", "rStar",
                                                          "Multi-Agent Evolve", "LongVideoAgent"]),
    ("src/longvideoagent/agents/director.py",            ["DIRECT"]),
    ("src/longvideoagent/agents/orchestrator.py",        ["CineAgents"]),
    ("src/longvideoagent/agents/editor.py",              ["ReAct", "GLANCE", "Sima 1.0",
                                                          "LongVideoAgent"]),
    ("src/longvideoagent/agents/validator.py",           ["G-Eval", "JudgeLM", "Tülu"]),
    ("src/longvideoagent/agents/critic.py",              ["Reflexion", "Trace", "rStar", "AFlow",
                                                          "Multi-Agent Evolve", "SELAUR",
                                                          "Trajectory-Informed Memory",
                                                          "AgentPRM"]),
    ("src/longvideoagent/memory/lessons.py",             ["Reflexion", "Trace", "AFlow",
                                                          "Trajectory-Informed Memory",
                                                          "Self-Evolving Agents"]),
    ("src/longvideoagent/models/reward/mllm_judge.py",   ["Qwen2.5-VL", "Tülu", "Skywork",
                                                          "JudgeLM", "AgentPRM", "ThinkPRM",
                                                          "VRPRM"]),
    ("src/longvideoagent/models/reward/ensemble.py",     ["Multi-Agent Debate", "DyLAN",
                                                          "JudgeLM", "Multi-Agent Evolve",
                                                          "SELAUR"]),
    ("src/longvideoagent/utils/preferences.py",          ["DPO", "IPO", "KTO", "SimPO", "GRPO",
                                                          "DGPO"]),
    ("src/longvideoagent/models/video_gen/omniweaving.py", ["HunyuanVideo", "CogVideoX"]),
    ("src/longvideoagent/models/video_gen/api_client.py", ["Veo 2"]),
    ("src/longvideoagent/perception/captioner.py",       ["Qwen3-VL", "InternVL3"]),
    # OPD (Round 14, branch rl-integration) ───────────────────────────
    ("training/stages/distill.py",                       ["GKD", "2306.13649",
                                                          "Thinking Machines",
                                                          "OPSD", "2601.18734",
                                                          "REOPOLD", "2603.11137",
                                                          "Rethinking OPD", "2604.13016"]),
]


@pytest.mark.parametrize("relpath,needles", EXPECTED_CITATIONS)
def test_top_docstring_cites(relpath: str, needles: list[str]):
    path = ROOT / relpath
    assert path.exists(), f"{relpath} missing"
    head = "\n".join(path.read_text().splitlines()[:80])
    missing = [n for n in needles if n not in head]
    assert not missing, (
        f"{relpath}: top docstring is missing required citation(s) {missing}. "
        f"If you removed a reference on purpose, update tests/unit/test_reference_grounding.py."
    )
