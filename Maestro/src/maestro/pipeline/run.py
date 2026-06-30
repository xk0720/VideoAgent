"""End-to-end driver. Builds components from config, runs Stage 0-3, writes
output video + metric report + trajectory log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..agents.act import ActAgent
from ..agents.capability_router import CapabilityRouter
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
from .generate_loop import generate_shot, generate_shot_orchestrated
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
    llm: object = None                       # the brain (orchestrator repair_mode)
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

    # Bind the AudioGenTool to the CONFIGURED audio client (last-write-wins
    # over the registry's mock default) so `models.audio_gen: wavespeed` is not
    # a hollow claim — the tool is REAL when invoked. NB: generated audio is not
    # yet a deterministic pipeline STAGE (a clip's sound still comes only from
    # the user's --music track muxed at assembly); this makes foley/TTS real for
    # any ActAgent tool_call, which is where the future sound-director stage hooks.
    audio_spec = cfg.get("models", {}).get("audio_gen")
    if audio_spec and str(
        audio_spec.get("name") if isinstance(audio_spec, dict) else audio_spec
    ).startswith("mock") is False:
        from ..models.audio_gen_backends import build_audio_gen
        from ..tools.audio_gen import AudioGenTool
        from ..tools.base import default_registry
        default_registry().register(AudioGenTool(client=build_audio_gen(audio_spec)))

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
        llm=llm,
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
    def _opt_float(key: str):
        return float(mem_cfg[key]) if key in mem_cfg else None

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
        # Skill lifecycle tuning (configs/default.yaml memory.skill_*).
        skill_distill_severity_threshold=_opt_float("skill_distill_severity_threshold"),
        skill_perf_ema_alpha=_opt_float("skill_perf_ema_alpha"),
        skill_eviction_floor=_opt_float("skill_eviction_floor"),
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
    #
    # ORDER MATTERS: identity ANCHORS register FIRST. They carry verified
    # reference paths/descriptions; a bare plan entity with the same name
    # (e.g. anchor 'hero.png' vs prompt noun 'hero') must never claim the id
    # first and shadow the anchor's refs. EntityStore.register additionally
    # backfills empty fields on collision (registration-time enrichment), so
    # the refs survive either order — but anchors-first keeps the log honest.
    #
    # NOTE on run scoping: identity ids are name-keyed, so generic-noun ids
    # ('hero', 'dog') deliberately SHARE state across runs — that is the
    # recurring-character feature. The annotator's 'subject' FALLBACK entity
    # (emitted when no noun cue matched) is pure noise under that rule and is
    # excluded from registration (and therefore from proposals).
    run_id = uuid.uuid4().hex[:12]
    write_gate = VerificationWriteGate()
    transitions_committed = 0
    transitions_rejected = 0
    if mlm.enabled["entities"]:
        for anchor in asset_memory.identity_anchors.values():
            name = anchor.name or anchor.identity_id
            mlm.entities.register(make_identity(
                name,
                reference_paths=[anchor.source] if anchor.source else [],
                description=anchor.description,
            ))
        for spec in specs:
            if spec.physics_annotation:
                for pe in spec.physics_annotation.entities:
                    if pe.name == "subject":   # annotator fallback — not an entity
                        continue
                    mlm.entities.register(make_identity(
                        pe.name,
                        description=f"planned entity (motion_class={pe.motion_class})",
                    ))

    # Stage 2 — self-improve loop per shot
    task_id = f"t{int(time.time())}"
    # F14: report "learned" counters are THIS-RUN deltas — cumulative library
    # sizes inflate on a shared memory_dir. Snapshot before the shot loop.
    skills_before = len(mlm.skills.by_class("creation")) if mlm.enabled["skills"] else 0
    lessons_before = len(mlm.lessons)
    results = []
    clips = []
    # Phase-2 — capability routing (skill-driven). At the top of each shot, the
    # matched_skill (from plan_shots) and asset_memory are both in hand, so this
    # is the seam where WHICH generation capability the shot needs is decided:
    # a matched skill's RECORDED capability wins (the skill decides the model);
    # otherwise a deterministic cold-start heuristic. The decision is logged to
    # the trajectory ("route_capability") and stored on the spec for the
    # GeneratorAgent to dispatch on. Choice is constrained to what THIS backend
    # actually offers — never a silent capability claim.
    router = CapabilityRouter()
    available_caps = comp.generator.video_gen.capabilities()
    # Repair mode: "hsi" (default — the rigid HSI tier ladder, generate_shot) or
    # "orchestrator" (the LLM brain function-calls over the tool registry,
    # generate_shot_orchestrated). Default keeps every existing test unchanged.
    repair_mode = str(gen_cfg.get("repair_mode", "hsi")).lower()
    orchestrator = None
    orchestrator_skills = mlm.skills if mlm.enabled["skills"] else None
    if repair_mode == "orchestrator":
        from ..agents.orchestrator import OrchestratorAgent
        orchestrator = OrchestratorAgent(
            llm=comp.llm, generator=comp.generator, refiner=comp.refiner,
            image_edit=comp.image_edit, retrieval=retrieval,
            skill_library=orchestrator_skills,   # RETRIEVE-FIRST learned repair workflows
            max_turns=int(gen_cfg.get("max_revisions", 5)), logger=trajectory,
        )
    for spec in specs:
        decision = router.route(spec, asset_memory, available_caps)
        spec.gen_capability = decision.capability
        spec.gen_params = dict(decision.params)
        trajectory.append(
            agent_name="CapabilityRouter",
            action="route_capability",
            action_input={"shot_idx": spec.shot_idx,
                          "available": sorted(available_caps)},
            observation={"capability": decision.capability,
                         "source": decision.source,
                         "reason": decision.reason,
                         "downgraded_from": decision.downgraded_from},
        )
        if orchestrator is not None:
            res = generate_shot_orchestrated(
                spec, comp.board, comp.generator, comp.refiner, comp.verifier,
                cache_dir, orchestrator,
                asset_memory=asset_memory, lesson_library=mlm.lessons,
                image_edit=comp.image_edit, tournament=comp.tournament,
                retrieval=retrieval, skill_library=orchestrator_skills, fps=fps,
                n_candidates=int(gen_cfg.get("n_candidates", 2)),
                max_turns=int(gen_cfg.get("max_revisions", 5)),
            )
        else:
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
                post_accept_strictness=float(
                    cfg.get("physics", {}).get("post_accept_strictness", 1.0)
                ),
            )
        results.append(res)
        clips.append(res.clip)

        # Verification-gated entity-state writes: transitions proposed from
        # this shot's plan are committed ONLY if the gate confirms them in
        # the clip the HSI loop just ACCEPTED ("commit only what rendered")
        # AND the loop converged on it (accepted = "the clip we ship", not
        # "defect-free"). Proposals and the gate's pending-selection are
        # scoped to THIS run's run_id, so a transition rejected in an earlier
        # run sharing the memory_dir stays re-proposable and report counters
        # count only this run's decisions. propose_transitions_from_spec is
        # the deterministic mock stand-in for the Director authoring
        # transitions at planning time.
        if mlm.enabled["entities"]:
            for t in propose_transitions_from_spec(spec, mlm.entities, run_id=run_id):
                mlm.entities.propose(t)
            gated = mlm.entities.commit_gated(
                res.clip, spec, write_gate,
                converged=res.converged, run_id=run_id,
            )
            transitions_committed += gated["committed"]
            transitions_rejected += gated["rejected"]

    # Stage 3
    shot_dur = float(cfg.get("plan", {}).get("shot_duration", 3.0))
    script = assemble(clips, output_path, music, shot_dur)

    # F12d: SkillOps lifecycle pass once per run — decay/evict stale skills.
    skills_evicted = mlm.skills.age_and_evict() if mlm.enabled["skills"] else 0

    distilled_skill_ids = [r.distilled_skill_id for r in results if r.distilled_skill_id]
    distilled_lesson_ids = [r.distilled_lesson_id for r in results if r.distilled_lesson_id]
    skills_total = len(mlm.skills.by_class("creation")) if mlm.enabled["skills"] else 0
    report = {
        "user_prompt": user_prompt,
        "n_shots": len(specs),
        "output_path": str(output_path),
        # "*_learned" = THIS RUN's deltas (a shared memory_dir would otherwise
        # inflate them); "*_total" = cumulative library sizes.
        "lessons_learned": max(0, len(mlm.lessons) - lessons_before),
        "lessons_total": len(mlm.lessons),
        # "learned" = distilled creation skills only; registered review/memory
        # skills are declared capabilities, not learned ones.
        "skills_learned": max(0, skills_total - skills_before),   # C7
        "skills_total": skills_total,
        "skills_evicted": skills_evicted,
        "skills_registered": (
            (len(mlm.skills) - skills_total) if mlm.enabled["skills"] else 0
        ),
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
