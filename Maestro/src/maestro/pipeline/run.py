"""End-to-end driver. Builds components from config, runs Stage 0-3, writes
output video + metric report + trajectory log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..agents.act import ActAgent
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
from ..memory.entity_store import make_identity, propose_transitions_from_spec
from ..memory.lesson_library import LessonLibrary
from ..memory.multi_layer import MultiLayerMemory
from ..memory.skill_admission import SkillAdmission
from ..memory.skill_library import PHYSICS_REVIEW_TIERS
from ..memory.write_gate import VerificationWriteGate
from ..models import build_image_edit, build_llm, build_mllm, build_video_gen
from ..models.world_reward import build_world_reward
from ..physics.tracks import build_track_extractor
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
    act: ActAgent                            # UniVA Plan→Act executor (v0.2.2)
    board: ReviewBoard
    tournament: Tournament
    lesson_library: LessonLibrary
    image_edit: object
    mlm: Optional[MultiLayerMemory] = None   # C8 — wired by run_maestro


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

    metric_tool = MetricTool(
        cfg.get("metrics", {}).get("weights"),
        world_reward=build_world_reward(cfg.get("models", {}).get("world_reward")),
    )
    # C6 oracle track extractor: mock by default; 'cotracker'/'tapir' recovers
    # observed motion from REAL generated frames (lazy torch).
    track_extractor = build_track_extractor(cfg.get("models", {}).get("track_extractor"))
    board = ReviewBoard(
        critics=[
            SemanticCritic(mllm=mllm, logger=trajectory),
            PhysicsCritic(mllm=mllm, logger=trajectory),
            PhysicsConsistencyCritic(
                violation_threshold=cfg.get("physics", {}).get("violation_threshold", 0.4),
                logger=trajectory, extractor=track_extractor,
            ),  # C6
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
        act=ActAgent(llm=llm, logger=trajectory),  # routes tool_call via registry
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
    memory_dir: Optional[Path] = None,
) -> dict:
    import time
    import uuid

    cfg = config.data if config else {}
    output_path = Path(output_path)
    cache_dir = Path(cache_dir or output_path.parent / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    fps = int(cfg.get("compose", {}).get("fps", 8))
    gen_cfg = cfg.get("compose", {})
    mem_cfg = cfg.get("memory", {})
    trajectory = TrajectoryLogger(trajectory_path)

    comp = build_components(config, trajectory, lesson_path)

    # C8 — Multi-Layer Memory façade (v0.3). The lesson library that
    # `build_components` already created is REPLACED by MLM's so all five
    # tiers persist to a consistent base_dir.
    mem_base = Path(memory_dir) if memory_dir else (
        output_path.parent / "memory" if output_path.parent.name else Path("./memory")
    )
    mlm = MultiLayerMemory.open(
        base_dir=mem_base,
        user_id=mem_cfg.get("user_id", "default"),
        lesson_path=lesson_path,                       # honor v0.2.2 caller's path
        enable_skills=bool(mem_cfg.get("enable_skills", True)),
        enable_entities=bool(mem_cfg.get("enable_entities", True)),
        enable_preferences=bool(mem_cfg.get("enable_preferences", True)),
        enable_episodes=bool(mem_cfg.get("enable_episodes", True)),
        # Skill CI (unified skill abstraction): distilled entries must pass
        # admission before persisting. Disable via memory.enable_skill_admission
        # to recover the legacy v0.3 insert-unconditionally behavior.
        skill_admission=(
            SkillAdmission()
            if bool(mem_cfg.get("enable_skill_admission", True)) else None
        ),
    )
    comp.mlm = mlm
    comp.lesson_library = mlm.lessons              # keep single source of truth

    # v0.4 dual-register entity memory: let the ConsistencyCritic read the
    # committed state registers (the store does not exist yet when
    # build_components constructs the board, so it is wired here).
    if mlm.enabled["entities"]:
        for critic in comp.board.critics:
            if isinstance(critic, ConsistencyCritic):
                critic.entity_store = mlm.entities

    # Register the C6 physics verification tiers as REVIEW skills so router
    # choices are recordable as skill usage (idempotent skill_id — safe to
    # re-register across runs sharing one memory_dir).
    if mlm.enabled["skills"]:
        for tier in PHYSICS_REVIEW_TIERS:
            mlm.skills.register_review_skill(f"physics_review_{tier}", tier)

    # Stage 0 — material understanding routes through the UniVA-style tool
    # registry via the ActAgent so the trajectory captures the Plan→Act handoff
    # (`tool_call` events for probe / caption / detect_objects). Entity tier 4
    # is consulted so identities reused across runs are deduplicated.
    asset_memory = build_asset_memory(
        source_videos, images, music, cache_dir, cfg, act_agent=comp.act,
        entity_store=mlm.entities if mlm.enabled["entities"] else None,
        task_id=str(uuid.uuid4())[:8],
    )
    log.info("AssetMemory: %s", asset_memory.summarize())
    retrieval = RetrievalTool(asset_memory)

    # Stage 1 — plan + Validate->Correct loop + Skill retrieval (C7).
    specs = plan_shots(
        user_prompt, asset_memory,
        comp.screenwriter, comp.director, comp.physics_planner,
        cache_dir, lesson_library=mlm.lessons,
        skill_library=mlm.skills if mlm.enabled["skills"] else None,
        plan_validator=comp.plan_validator,
        max_plan_iters=int(cfg.get("plan", {}).get("max_plan_iters", 3)),
        fps=fps,
    )

    # v0.4 — dual-register entity memory + verification-gated writes
    # (memory/entity_store.py, memory/write_gate.py; survey Angles 1+2).
    # Register the IMMUTABLE identity half up front: one frozen register per
    # entity named by the plan (physics annotation entities) or anchored by
    # an uploaded image (identity anchors carry reference paths). Idempotent
    # — same name always maps to the same entity_id, across shots AND runs.
    write_gate = VerificationWriteGate()
    transitions_committed = 0
    transitions_rejected = 0
    if mlm.enabled["entities"]:
        for spec in specs:
            if spec.physics_annotation:
                for pe in spec.physics_annotation.entities:
                    mlm.entities.register(make_identity(
                        pe.name,
                        description=f"planned entity (motion_class={pe.motion_class})",
                    ))
        for anchor in asset_memory.identity_anchors.values():
            name = anchor.name or anchor.identity_id
            mlm.entities.register(make_identity(
                name,
                reference_paths=[anchor.source] if anchor.source else [],
                description=anchor.description,
            ))

    # Stage 2 — self-improve loop per shot
    task_id = f"t{int(time.time())}"
    results = []
    clips = []
    for spec in specs:
        res = generate_shot(
            spec, comp.board, comp.generator, comp.refiner, comp.verifier,
            cache_dir, asset_memory=asset_memory, lesson_library=mlm.lessons,
            image_edit=comp.image_edit, tournament=comp.tournament, retrieval=retrieval,
            physics_planner=comp.physics_planner,  # HSI Tier-1 (C5)
            director=comp.director,                 # HSI Tier-2 (C5)
            skill_library=mlm.skills if mlm.enabled["skills"] else None,
            task_id=task_id,
            fps=fps,
            n_candidates=int(gen_cfg.get("n_candidates", 2)),
            max_revisions=int(gen_cfg.get("max_revisions", 5)),
            k_retries=int(gen_cfg.get("k_retries", 2)),
        )
        results.append(res)
        clips.append(res.clip)

        # Verification-gated entity-state writes: transitions proposed from
        # this shot's plan are committed ONLY if the gate confirms them in
        # the clip the HSI loop just ACCEPTED ("commit only what rendered").
        # propose_transitions_from_spec is the deterministic mock stand-in
        # for the Director authoring transitions at planning time.
        if mlm.enabled["entities"]:
            for t in propose_transitions_from_spec(spec, mlm.entities):
                mlm.entities.propose(t)
            gated = mlm.entities.commit_gated(res.clip, spec, write_gate)
            transitions_committed += gated["committed"]
            transitions_rejected += gated["rejected"]

    # Stage 3
    shot_dur = float(cfg.get("plan", {}).get("shot_duration", 3.0))
    script = assemble(clips, output_path, music, shot_dur)

    distilled_skill_ids = [r.distilled_skill_id for r in results if r.distilled_skill_id]
    distilled_lesson_ids = [r.distilled_lesson_id for r in results if r.distilled_lesson_id]
    report = {
        "user_prompt": user_prompt,
        "n_shots": len(specs),
        "output_path": str(output_path),
        "lessons_learned": len(mlm.lessons),
        # "learned" = distilled creation skills only; registered review/memory
        # skills are declared capabilities, not learned ones.
        "skills_learned": len(mlm.skills.by_class("creation")),   # C7
        "skills_registered": len(mlm.skills) - len(mlm.skills.by_class("creation")),
        "entities_persisted": len(mlm.entities), # C8 (v0.3 legacy register)
        # v0.4 dual-register entity memory + verification-gated writes:
        "entities": len(mlm.entities.identities),
        "transitions_committed": transitions_committed,
        "transitions_rejected": transitions_rejected,
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
                "matched_skill_id": (              # C7 — what we retrieved from library
                    specs[r.clip.shot_idx].matched_skill.skill_id
                    if specs[r.clip.shot_idx].matched_skill else ""
                ),
                "distilled_skill_id": r.distilled_skill_id,    # C7 — what we LEARNED
                "distilled_lesson_id": r.distilled_lesson_id,  # C4 lesson id
            }
            for r in results
        ],
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # C8 Tier-1 — persist the episodic trace for this run so future planning
    # can retrieve similar past tasks.
    if mlm.enabled["episodes"]:
        mlm.episodes.append(
            task_id=task_id,
            user_prompt=user_prompt,
            trajectory_path=str(trajectory_path) if trajectory_path else "",
            report=report,
            lessons_distilled=distilled_lesson_ids,
            skills_distilled=distilled_skill_ids,
        )

    log.info("Done. Output=%s Report=%s", output_path, report_path)
    return {
        "output_path": output_path,
        "report_path": report_path,
        "report": report,
        "script": script,
        "trajectory": trajectory,
        "results": results,
        "mlm": mlm,                  # exposed for tests / downstream apps
    }
