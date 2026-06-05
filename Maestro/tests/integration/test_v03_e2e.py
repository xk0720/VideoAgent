"""v0.3 end-to-end — verify the C7 distillation + C8 cross-run memory really
fire when `run_maestro` executes the whole pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from maestro.config import load_config
from maestro.memory.multi_layer import MultiLayerMemory
from maestro.pipeline.run import run_maestro


def test_v03_run_distills_skill_and_persists_episode(tmp_path: Path):
    """A normal mock-pipeline run (which the test_loop tests already prove
    converges at Tier 0) MUST result in:
      - at least one skill distilled (perf_score > 0)
      - the episodic trace appended
      - report exposes matched_skill_id / distilled_skill_id columns
    """
    out = tmp_path / "out.mp4"
    cfg = load_config()
    result = run_maestro(
        user_prompt="a ball is thrown and bounces off a wall",
        output_path=out,
        config=cfg,
        cache_dir=tmp_path / "cache",
        trajectory_path=tmp_path / "out.traj.jsonl",
        memory_dir=tmp_path / "memory",
    )
    report = result["report"]
    # C7 — at least one skill should be born from this run.
    assert report["skills_learned"] >= 1, report
    # Every shot has the new columns.
    for shot in report["shots"]:
        assert "matched_skill_id" in shot
        assert "distilled_skill_id" in shot
        assert "distilled_lesson_id" in shot
    # At least one shot actually distilled a skill (id non-empty).
    distilled_any = any(s["distilled_skill_id"] for s in report["shots"])
    assert distilled_any, [s["distilled_skill_id"] for s in report["shots"]]
    # C8 — episodic trace appended to memory/episodes.jsonl.
    ep_path = tmp_path / "memory" / "episodes.jsonl"
    assert ep_path.exists()
    lines = [l for l in ep_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["user_prompt"] == "a ball is thrown and bounces off a wall"
    assert rec["skills_distilled"]


def test_v03_skill_is_retrieved_on_second_run(tmp_path: Path):
    """End-to-end the loop closes: run #1 distills a skill, run #2 RETRIEVES
    it during planning (visible in `report.shots[*].matched_skill_id`)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    mem_dir = tmp_path / "memory"
    cfg = load_config()

    # Round 1 — fresh memory; we should distill at least one skill.
    r1 = run_maestro(
        user_prompt="a ball is thrown and bounces off a wall",
        output_path=out_dir / "r1.mp4", config=cfg,
        cache_dir=tmp_path / "c1",
        trajectory_path=out_dir / "r1.traj.jsonl",
        memory_dir=mem_dir,
    )
    assert r1["report"]["skills_learned"] >= 1

    # Round 2 — same memory_dir; planning must pick up a matching skill.
    r2 = run_maestro(
        user_prompt="a ball is thrown and bounces off a wall",
        output_path=out_dir / "r2.mp4", config=cfg,
        cache_dir=tmp_path / "c2",
        trajectory_path=out_dir / "r2.traj.jsonl",
        memory_dir=mem_dir,
    )
    matched_any = any(s["matched_skill_id"] for s in r2["report"]["shots"])
    assert matched_any, [s["matched_skill_id"] for s in r2["report"]["shots"]]


def test_v03_episodic_retrieval_finds_similar_prior_run(tmp_path: Path):
    """After one run, MLM.episodes.similar_tasks must return that run for a
    semantically-close new prompt — closes Tier-1 replay loop."""
    mem_dir = tmp_path / "memory"
    cfg = load_config()
    run_maestro(
        user_prompt="a ball is thrown and bounces off a wall",
        output_path=tmp_path / "r1.mp4", config=cfg,
        cache_dir=tmp_path / "c1",
        trajectory_path=tmp_path / "r1.traj.jsonl",
        memory_dir=mem_dir,
    )
    mlm = MultiLayerMemory.open(base_dir=mem_dir, user_id="default")
    hits = mlm.episodes.similar_tasks(
        "a ball is thrown into the air", top_k=1, threshold=0.05,
    )
    assert hits
    assert "ball" in hits[0].user_prompt.lower()
