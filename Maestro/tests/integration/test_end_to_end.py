import json
from pathlib import Path

from maestro.config import load_config
from maestro.pipeline.run import run_maestro


def test_end_to_end_produces_outputs(tmp_path: Path):
    # fixtures: a tiny placeholder source video, image and music (mock perception)
    src = tmp_path / "clip.mp4"; src.write_text("mock video bytes")
    img = tmp_path / "hero.png"; img.write_text("mock image bytes")
    music = tmp_path / "track.mp3"; music.write_text("mock audio bytes")

    out = tmp_path / "out" / "demo.mp4"
    config = load_config()  # default.yaml

    result = run_maestro(
        user_prompt="a ball is thrown and bounces; a person runs through a city",
        output_path=out,
        source_videos=[src],
        images=[img],
        music=music,
        config=config,
        cache_dir=tmp_path / "cache",
        trajectory_path=out.with_suffix(".trajectory.jsonl"),
        lesson_path=tmp_path / "lessons.jsonl",
    )

    # 1. produced a video + report + trajectory
    assert out.exists()
    assert result["report_path"].exists()
    traj = out.with_suffix(".trajectory.jsonl")
    assert traj.exists() and traj.read_text().strip()

    # 2. self-improve loop is visible in the trajectory
    entries = [json.loads(l) for l in traj.read_text().splitlines() if l.strip()]
    actions = [e["action"] for e in entries]
    assert "generate" in actions
    assert "verify" in actions          # Verifier ran
    assert "annotate_physics" in actions  # physics module ran
    assert "validate_plan" in actions   # plan-level Validate->Correct loop ran
    # v0.2.2: UniVA-style ActAgent must have routed analysis tools through the
    # registry during Stage 0 (build_asset_memory). Pre-fix this was orphaned;
    # this assertion locks it in.
    assert "tool_call" in actions, \
        "ActAgent never invoked any tool — Stage 0 wiring regressed"
    tool_calls = [e for e in entries if e["action"] == "tool_call"]
    invoked = {e["action_input"]["name"] for e in tool_calls}
    assert "video_probe" in invoked          # analysis
    assert "caption" in invoked              # analysis
    assert "detect_objects" in invoked       # tracking
    assert all(e["observation"]["ok"] for e in tool_calls), \
        "some tool_call failed in production trajectory"

    # 3. report shows per-shot revisions + converged + score history
    report = result["report"]
    assert report["n_shots"] >= 1
    for shot in report["shots"]:
        assert "score_history" in shot
        assert "p1_physics" in shot["final_metrics"]
        assert "p2_law_consistency" in shot["final_metrics"]   # C6 wired
        assert "tier_used" in shot                                # C5 wired
        assert "escalations" in shot


def test_config_override_changes_behavior(tmp_path: Path):
    out = tmp_path / "o.mp4"
    cfg = load_config(overrides={"plan": {"n_shots": 1, "max_shots": 1}})
    result = run_maestro("a dog jumps over a fence", out, config=cfg,
                         cache_dir=tmp_path / "c")
    assert result["report"]["n_shots"] == 1
