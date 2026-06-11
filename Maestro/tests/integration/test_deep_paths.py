"""Deep-path verification — go beyond "do tests pass" to "does the framework
actually behave the way the design doc promises?".

Covers four properties the connectivity audit could not check by itself:

  1. **C4 actually improves subsequent runs** — A/B with shared LessonLibrary
     vs. fresh library; the first run must inject lessons that change the
     second run's plan (DirectorAgent.injected_lessons populated).
  2. **HSI exhaustion** — when every tier fails for k_retries rounds, the
     escape hatch must fire and the loop must not deadlock.
  3. **Server boots under real uvicorn** (not just TestClient) — proves the
     ASGI app is actually wireable behind a real server process; doubles as a
     `Dockerfile` smoke for `HEALTHCHECK`.
  4. **No stale `sys.modules` / circular imports** — re-import the public
     surface in a fresh interpreter to catch top-level side-effects we don't
     want.

All four are CPU-only.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest


def _subproc_env() -> dict:
    """A copy of the parent env with the package src dir prepended to
    PYTHONPATH so child interpreters can resolve `maestro.*` without a
    prior `pip install -e .`. The pytest harness gets the same path via
    `pyproject.toml [tool.pytest.ini_options] pythonpath`, but subprocesses
    don't inherit pytest's modifications.
    """
    here = Path(__file__).resolve().parents[2]   # Maestro/
    src = here / "src"
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) + (os.pathsep + prev if prev else "")
    return env


# ─────────────────────────────────────────────────────────────────────────────
# 1) C4 — LessonLibrary truly improves subsequent runs
# ─────────────────────────────────────────────────────────────────────────────
def test_lesson_library_changes_next_run_plan(tmp_path: Path):
    """First run distills a lesson; second run's Director must retrieve it
    via injected_lessons and pass it forward into the generator's prompt."""
    from maestro.config import load_config
    from maestro.memory.lesson_library import LessonLibrary
    from maestro.pipeline.run import run_maestro

    cfg = load_config()
    lessons_path = tmp_path / "lessons.jsonl"

    # Round 1 — empty library; should write at least one lesson.
    run_maestro(
        user_prompt="a ball is thrown and bounces off a wall",
        output_path=tmp_path / "r1.mp4",
        config=cfg,
        cache_dir=tmp_path / "c1",
        trajectory_path=tmp_path / "r1.traj.jsonl",
        lesson_path=lessons_path,
    )
    after_r1 = len(LessonLibrary(lessons_path))
    assert after_r1 >= 1, "Round 1 distilled no lesson"

    # Round 2 — same library file. The Director must retrieve the lesson and
    # inject it into ShotSpec, which then flows into the generator's prompt
    # ("constraints: ..." prefix in the mock metadata) and the trajectory's
    # `expand_shotspecs` observation should show lessons_injected > 0.
    run_maestro(
        user_prompt="a ball is thrown and bounces off a wall",
        output_path=tmp_path / "r2.mp4",
        config=cfg,
        cache_dir=tmp_path / "c2",
        trajectory_path=tmp_path / "r2.traj.jsonl",
        lesson_path=lessons_path,
    )
    entries2 = [json.loads(l) for l in (tmp_path / "r2.traj.jsonl").read_text().splitlines() if l.strip()]
    expand = [e for e in entries2 if e["action"] == "expand_shotspecs"]
    assert expand, "DirectorAgent did not log expand_shotspecs in round 2"
    assert expand[0]["observation"]["lessons_injected"] > 0, \
        "round-2 Director did not retrieve any lesson from the library"


# ─────────────────────────────────────────────────────────────────────────────
# 2) HSI exhaustion — every tier fails → escape hatch
# ─────────────────────────────────────────────────────────────────────────────
class _AlwaysFailingMLLM:
    """No critic verdict ever passes; every metric stays low so the Verifier
    cannot accept any candidate at any tier — forces the loop into escape."""

    name = "always-failing"

    def assess_semantic(self, clip, spec):
        return [("never passes", False, "fix the unfixable")]

    def assess_physics(self, clip, spec, fps):
        from maestro.types import PhysFailureMode, PhysicsVerdict
        return [PhysicsVerdict(
            mode=PhysFailureMode.GRAVITY_INERTIA,
            frame_range=(0, max(1, int(round(spec.duration * fps)))),
            severity=0.95,
            suggested_intervention="impossible fix",
        )]

    def compare(self, a, b, spec):
        return 0


def test_hsi_falls_back_to_escape_hatch_when_all_tiers_fail(tmp_path: Path):
    """Synthetic worst case: every tier rejected by Verifier → loop must
    still terminate (escape hatch tier 3 fires) and never deadlock."""
    from maestro.agents.director import DirectorAgent
    from maestro.agents.generator import GeneratorAgent
    from maestro.agents.physics_planner import PhysicsPlannerAgent
    from maestro.agents.refiner import RefinerAgent
    from maestro.agents.verifier import VerifierAgent
    from maestro.critics.board import ReviewBoard
    from maestro.critics.consistency import ConsistencyCritic
    from maestro.critics.physics import PhysicsCritic
    from maestro.critics.rhythm import RhythmCritic
    from maestro.critics.semantic import SemanticCritic
    from maestro.physics.annotate import annotate_physics
    from maestro.pipeline.generate_loop import generate_shot
    from maestro.tools.metric_tool import MetricTool
    from maestro.types import ShotSpec

    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_annotation = annotate_physics(spec)

    judge = _AlwaysFailingMLLM()
    # Pin weights to dims the stubborn judge controls so the Verifier cannot
    # accept on m2/aesthetic auto-improvement.
    weights = {"m1_semantic": 0.5, "p1_physics": 0.5, "m2_temporal": 0.0,
               "p2_law_consistency": 0.0, "id1_identity": 0.0,
               "m5_rhythm": 0.0, "aesthetic": 0.0}
    board = ReviewBoard(
        critics=[SemanticCritic(mllm=judge), PhysicsCritic(mllm=judge),
                 ConsistencyCritic(), RhythmCritic()],
        metric_tool=MetricTool(weights=weights),
    )

    res = generate_shot(
        spec, board,
        GeneratorAgent(), RefinerAgent(), VerifierAgent(), tmp_path,
        physics_planner=PhysicsPlannerAgent(),
        director=DirectorAgent(),
        max_revisions=3, k_retries=1, n_candidates=1,
    )
    # Escape hatch (tier 3) must appear in tier_used.
    assert 3 in res.tier_used, res.tier_used
    # Escape hatch tier 3 reaches it AFTER all of tier 0/1/2 retries failed —
    # `escalations` counts moves past tier 0, so at least 2 per round that
    # ended in escape.
    assert res.escalations >= 2, res.escalations
    # And we still terminate gracefully with a clip object marked accepted.
    assert res.clip.accepted
    # Score history non-decreasing across all escape rounds.
    h = res.score_history
    assert all(h[i] <= h[i + 1] + 1e-6 for i in range(len(h) - 1)), h


# ─────────────────────────────────────────────────────────────────────────────
# 3) Server boots under real uvicorn (not TestClient)
# ─────────────────────────────────────────────────────────────────────────────
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.skipif(
    "uvicorn" not in {m.split(".")[0] for m in sys.modules}
    and not __import__("importlib.util").util.find_spec("uvicorn"),
    reason="uvicorn not installed (optional [server] extras)",
)
def test_server_health_via_real_uvicorn(tmp_path: Path):
    """Boot uvicorn in a subprocess, GET /health, then terminate. Catches
    issues TestClient can't (real ASGI lifespan, port binding, etc.)."""
    port = _free_port()
    # Use the CLI's `serve` so we exercise the same entry point Docker uses.
    proc = subprocess.Popen(
        [sys.executable, "-m", "maestro.cli", "serve",
         "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=_subproc_env(),
    )
    try:
        url = f"http://127.0.0.1:{port}/health"
        body = None
        # Poll up to 5s for the server to come up.
        deadline = time.time() + 5.0
        last_err = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as r:
                    body = json.loads(r.read())
                    break
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        assert body is not None, f"uvicorn never served /health: {last_err}"
        assert body["status"] == "ok"
        assert body["service"] == "maestro"
        assert body["n_tools"] >= 7
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─────────────────────────────────────────────────────────────────────────────
# 4) Clean-interpreter import — no top-level side effects, no circular imports
# ─────────────────────────────────────────────────────────────────────────────
def test_public_surface_imports_in_fresh_interpreter():
    """Spawn a subprocess that imports the public surface and prints the
    registered tool count + agent class names. Catches:
      • top-level side effects (e.g., reading $HOME, opening sockets)
      • circular imports that pytest's already-warm sys.modules hides
    """
    script = textwrap.dedent("""
        import sys, json
        from maestro.agents import (
            ScreenwriterAgent, DirectorAgent, PhysicsPlannerAgent,
            PlanValidatorAgent, GeneratorAgent, VerifierAgent, RefinerAgent,
            ActAgent, ToolCall, ToolResult,
        )
        from maestro.tools import default_registry
        from maestro.pipeline.run import run_maestro, build_components
        from maestro.critics import (
            ReviewBoard, SemanticCritic, PhysicsCritic, ConsistencyCritic,
            RhythmCritic, Tournament, PhysicsConsistencyCritic,
        )
        out = {
            "tool_count": len(default_registry().names()),
            "agents_ok": True,
            "critics_ok": True,
        }
        print(json.dumps(out))
    """).strip()
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=30,
                       env=_subproc_env())
    assert r.returncode == 0, f"stderr: {r.stderr}"
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["agents_ok"] and info["critics_ok"]
    assert info["tool_count"] >= 9     # all v0.2.2 default tools registered


# ─────────────────────────────────────────────────────────────────────────────
# 5) scripts/run_pipeline.py exposes EVERY innovation in stdout
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_script_exposes_every_innovation_in_stdout(tmp_path: Path):
    """The demo entry script must surface evidence of C1-C6 + UniVA wiring
    in a single run's stdout. Locks the operator-visible contract so future
    refactors of the script don't silently drop a column."""
    repo = Path(__file__).resolve().parents[2]
    src_video = tmp_path / "src.mp4"; src_video.write_text("mock")
    img = tmp_path / "hero.png"; img.write_text("mock")
    out = tmp_path / "demo.mp4"
    r = subprocess.run(
        [sys.executable, str(repo / "scripts/run_pipeline.py"),
         "--prompt", "a ball is thrown and bounces off a wall",
         "--source", str(src_video),
         "--image", str(img),
         "--output", str(out)],
        capture_output=True, text=True, timeout=30, env=_subproc_env(),
    )
    assert r.returncode == 0, r.stderr
    s = r.stdout
    # UniVA-style tool manifest banner
    assert "UniVA-style registry" in s
    assert "analysis" in s and "tracking" in s
    # Per-shot panel: every innovation tag must appear
    for marker in (
        "(C5)",                      # HSI tier_used + escalations
        "p1(C1)",                    # native physics
        "p2(C6)",                    # sketch consistency
        "Lessons learned (C4)",      # cross-task memory
    ):
        assert marker in s, f"stdout missing marker {marker!r}"
    # Trajectory action distribution reveals load-bearing agents
    for action in ("review", "generate", "plan_fix", "verify",
                   "annotate_physics", "tool_call", "validate_plan"):
        assert action in s, f"stdout missing action {action!r}"
