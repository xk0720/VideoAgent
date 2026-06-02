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
    actions = [json.loads(l)["action"] for l in traj.read_text().splitlines() if l.strip()]
    assert "generate" in actions
    assert "verify" in actions          # Verifier ran
    assert "build_sketch" in actions    # physics module ran
    assert "validate_plan" in actions   # plan-level Validate->Correct loop ran

    # 3. report shows per-shot revisions + converged + score history
    report = result["report"]
    assert report["n_shots"] >= 1
    for shot in report["shots"]:
        assert "score_history" in shot
        assert "p1_physics" in shot["final_metrics"]


def test_config_override_changes_behavior(tmp_path: Path):
    out = tmp_path / "o.mp4"
    cfg = load_config(overrides={"plan": {"n_shots": 1, "max_shots": 1}})
    result = run_maestro("a dog jumps over a fence", out, config=cfg,
                         cache_dir=tmp_path / "c")
    assert result["report"]["n_shots"] == 1
