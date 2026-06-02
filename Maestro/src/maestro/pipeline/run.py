"""End-to-end driver. Builds components from config, runs Stage 0-3, writes
output video + metric report + trajectory log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..agents.director import DirectorAgent
from ..agents.generator import GeneratorAgent
from ..agents.physics_planner import PhysicsPlannerAgent
from ..agents.plan_validator import PlanValidatorAgent
from ..agents.refiner import RefinerAgent
from ..agents.screenwriter import ScreenwriterAgent
from ..agents.verifier import VerifierAgent
from ..config import Config
from ..critics.board import ReviewBoard
from ..critics.consistency import ConsistencyCritic
from ..critics.physics import PhysicsCritic
from ..critics.physics_consistency import PhysicsConsistencyCritic
from ..critics.rhythm import RhythmCritic
from ..critics.semantic import SemanticCritic
from ..critics.tournament import Tournament
from ..logging_utils import get_logger
from ..memory.lesson_library import LessonLibrary
from ..models import build_image_edit, build_llm, build_mllm, build_video_gen
from ..tools.metric_tool import MetricTool
from ..tools.retrieval_tool import RetrievalTool
from ..trajectory import TrajectoryLogger
from .assemble import assemble
from .generate_loop import generate_shot
from .plan import plan_shots
from .understand import build_asset_memory

log = get_logger(__name__)


@dataclass
class MaestroComponents:
    screenwriter: ScreenwriterAgent
    director: DirectorAgent
    physics_planner: PhysicsPlannerAgent
    plan_validator: PlanValidatorAgent
    generator: GeneratorAgent
    refiner: RefinerAgent
    verifier: VerifierAgent
    board: ReviewBoard
    tournament: Tournament
    lesson_library: LessonLibrary
    image_edit: object


def build_components(
    config: Optional[Config] = None,
    trajectory: Optional[TrajectoryLogger] = None,
    lesson_path: Optional[Path] = None,
) -> MaestroComponents:
    cfg = config.data if config else {}
    llm = build_llm(cfg.get("models", {}).get("llm"))
    mllm = build_mllm(cfg.get("models", {}).get("mllm"))
    video_gen = build_video_gen(cfg.get("models", {}).get("video_gen"))
    image_edit = build_image_edit(cfg.get("models", {}).get("image_edit"))

    metric_tool = MetricTool(cfg.get("metrics", {}).get("weights"))
    board = ReviewBoard(
        critics=[
            SemanticCritic(mllm=mllm, logger=trajectory),
            PhysicsCritic(mllm=mllm, logger=trajectory),
            PhysicsConsistencyCritic(logger=trajectory),  # C6: closed-loop sketch verify
            ConsistencyCritic(logger=trajectory),
            RhythmCritic(logger=trajectory),
        ],
        metric_tool=metric_tool,
    )
    plan_cfg = cfg.get("plan", {})
    return MaestroComponents(
        screenwriter=ScreenwriterAgent(llm=llm, config=plan_cfg, logger=trajectory),
        director=DirectorAgent(llm=llm, config=plan_cfg, logger=trajectory),
        physics_planner=PhysicsPlannerAgent(llm=llm, logger=trajectory),
        plan_validator=PlanValidatorAgent(llm=llm, logger=trajectory),
        generator=GeneratorAgent(video_gen=video_gen, logger=trajectory),
        refiner=RefinerAgent(llm=llm, logger=trajectory),
        verifier=VerifierAgent(llm=llm, logger=trajectory),
        board=board,
        tournament=Tournament(judge=mllm),
        lesson_library=LessonLibrary(lesson_path),
        image_edit=image_edit,
    )


def run_maestro(
    user_prompt: str,
    output_path: Path,
    source_videos: Optional[list[Path]] = None,
    images: Optional[list[Path]] = None,
    music: Optional[Path] = None,
    config: Optional[Config] = None,
    cache_dir: Optional[Path] = None,
    trajectory_path: Optional[Path] = None,
    lesson_path: Optional[Path] = None,
) -> dict:
    cfg = config.data if config else {}
    output_path = Path(output_path)
    cache_dir = Path(cache_dir or output_path.parent / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    fps = int(cfg.get("compose", {}).get("fps", 8))
    gen_cfg = cfg.get("compose", {})
    trajectory = TrajectoryLogger(trajectory_path)

    comp = build_components(config, trajectory, lesson_path)

    # Stage 0
    asset_memory = build_asset_memory(source_videos, images, music, cache_dir, cfg)
    log.info("AssetMemory: %s", asset_memory.summarize())
    retrieval = RetrievalTool(asset_memory)

    # Stage 1 — plan + Validate->Correct loop
    specs = plan_shots(
        user_prompt, asset_memory,
        comp.screenwriter, comp.director, comp.physics_planner,
        cache_dir, lesson_library=comp.lesson_library,
        plan_validator=comp.plan_validator,
        max_plan_iters=int(cfg.get("plan", {}).get("max_plan_iters", 3)),
        fps=fps,
    )

    # Stage 2 — self-improve loop per shot
    results = []
    clips = []
    for spec in specs:
        res = generate_shot(
            spec, comp.board, comp.generator, comp.refiner, comp.verifier,
            cache_dir, asset_memory=asset_memory, lesson_library=comp.lesson_library,
            image_edit=comp.image_edit, tournament=comp.tournament, retrieval=retrieval,
            physics_planner=comp.physics_planner,  # HSI Tier-1 (C5)
            director=comp.director,                 # HSI Tier-2 (C5)
            fps=fps,
            n_candidates=int(gen_cfg.get("n_candidates", 2)),
            max_revisions=int(gen_cfg.get("max_revisions", 5)),
            k_retries=int(gen_cfg.get("k_retries", 2)),
        )
        results.append(res)
        clips.append(res.clip)

    # Stage 3
    shot_dur = float(cfg.get("plan", {}).get("shot_duration", 3.0))
    script = assemble(clips, output_path, music, shot_dur)

    report = {
        "user_prompt": user_prompt,
        "n_shots": len(specs),
        "output_path": str(output_path),
        "lessons_learned": len(comp.lesson_library),
        "shots": [
            {
                "shot_idx": r.clip.shot_idx,
                "revisions_used": r.revisions_used,
                "converged": r.converged,
                "gen_calls": r.gen_calls,
                "score_history": [round(s, 4) for s in r.score_history],
                "final_metrics": r.clip.metric_scores,
                "skipped_items": r.clip.skipped_items,
                "tier_used": r.tier_used,        # HSI tier per revision (C5)
                "escalations": r.escalations,
            }
            for r in results
        ],
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Done. Output=%s Report=%s", output_path, report_path)
    return {
        "output_path": output_path,
        "report_path": report_path,
        "report": report,
        "script": script,
        "trajectory": trajectory,
        "results": results,
    }
