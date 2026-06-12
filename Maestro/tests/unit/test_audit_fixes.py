"""Regression tests for the audit-confirmed self-improve-loop + memory fixes.

One test (or small group) per fix:
  F2  prompt/strictness lifecycle (tier-2 mutation never leaks into tier-1
      re-annotation; strictness resets on loop exit)
  F3  escape-hatch accounting (typed loop-level skips; distill refuses
      hatched episodes; lesson distillation subtracts skips)
  F4  mirrored checklist items flip with their verdicts; honest `accepted`
  F5  md5-stable skill names (PYTHONHASHSEED-independent)
  F6  injected lesson ids flow into distilled skills (coupling no longer a no-op)
  F7  write gate: clip-body-only evidence, converged required, hatched
      renders rejected
  F8  identity-anchor enrichment on name collision
  F9  run-scoped proposal dedup / pending-commit selection
  F11 atomic persistence (tmp+rename)
  F12 legacy re-distill threshold update, stable embedding, lifecycle wiring
  F13 config threading (severity threshold blocks distill end-to-end)
  F14 report counters are this-run deltas
  F15 verifier tie-break does not double-count physics defects
  F16 keyframe filenames include the seed
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from maestro.agents.director import DirectorAgent
from maestro.agents.generator import GeneratorAgent
from maestro.agents.physics_planner import PhysicsPlannerAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.screenwriter import ScreenwriterAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.consistency import ConsistencyCritic
from maestro.critics.physics import PhysicsCritic
from maestro.critics.physics_consistency import PhysicsConsistencyCritic
from maestro.critics.rhythm import RhythmCritic
from maestro.critics.semantic import SemanticCritic
from maestro.memory.entity_store import (
    EntityStore,
    make_identity,
    propose_transitions_from_spec,
)
from maestro.memory.lesson_library import LessonLibrary
from maestro.memory.skill_library import SkillLibrary
from maestro.memory.write_gate import VerificationWriteGate
from maestro.physics.annotate import annotate_physics
from maestro.pipeline.generate_loop import (
    _distill_lesson,
    _escape_hatch,
    _post_accept_audit,
    _stable_skill_name,
    generate_shot,
)
from maestro.pipeline.plan import plan_shots
from maestro.tools.metric_tool import MetricTool
from maestro.types import (
    AssetMemory,
    CandidateClip,
    ChecklistItem,
    CinematographyTags,
    PhysFailureMode,
    PhysicsVerdict,
    ShotSpec,
    StateTransition,
)


def _board():
    return ReviewBoard([
        SemanticCritic(), PhysicsCritic(), PhysicsConsistencyCritic(),
        ConsistencyCritic(), RhythmCritic(),
    ])


def _spec(prompt: str, shot_idx: int = 0, duration: float = 1.0) -> ShotSpec:
    spec = ShotSpec(shot_idx=shot_idx, duration=duration, prompt=prompt)
    spec.physics_annotation = annotate_physics(spec)
    return spec


def _accepted_clip(tmp_path: Path, shot_idx: int, body: str,
                   accepted: bool = True) -> CandidateClip:
    vp = tmp_path / f"shot{shot_idx:03d}_gate.mp4"
    vp.write_text(f"MOCK VIDEO\nprompt={body}\n", encoding="utf-8")
    return CandidateClip(shot_idx=shot_idx, video_path=vp, accepted=accepted)


# ─────────────────────────────────────────────────────────────────────────────
# F2 — prompt/strictness lifecycle
# ─────────────────────────────────────────────────────────────────────────────
def test_tier2_then_tier1_preserves_entity_set():
    """A Tier-2 prompt mutation ('| plan-fix: <hint>') must not leak into the
    Tier-1 re-annotation: replanning from the snapshotted base prompt keeps
    the entity set and expected modes identical to the original annotation."""
    spec = _spec("a ball is thrown")
    orig_entities = {e.name for e in spec.physics_annotation.entities}
    orig_modes = set(spec.physics_annotation.expected_modes)
    base_prompt = spec.prompt

    # Tier-2 hint text deliberately names an entity cue ('cup') and a fluid
    # cue ('splash') that a naive keyword re-scan would absorb.
    DirectorAgent().refine_spec(spec, "avoid splash artifacts near the cup")
    assert "plan-fix:" in spec.prompt

    PhysicsPlannerAgent().replan(spec, None, base_prompt=base_prompt)
    assert {e.name for e in spec.physics_annotation.entities} == orig_entities
    assert set(spec.physics_annotation.expected_modes) == orig_modes
    assert spec.physics_annotation.strictness == 1.0

    # Control: WITHOUT the snapshot the hint text leaks (this is the bug F2
    # fixed) — proves the test would catch a regression.
    PhysicsPlannerAgent().replan(spec, None)
    assert {e.name for e in spec.physics_annotation.entities} != orig_entities


# ─────────────────────────────────────────────────────────────────────────────
# F3 / F4 — escape-hatch accounting + mirror flip
# ─────────────────────────────────────────────────────────────────────────────
def test_escape_hatch_flips_mirrored_item_and_returns_typed_mode():
    clip = CandidateClip(shot_idx=0, video_path=Path("x.mp4"))
    clip.physics_verdicts = [PhysicsVerdict(
        mode=PhysFailureMode.GRAVITY_INERTIA, frame_range=(0, 8),
        severity=0.9, suggested_intervention="fix gravity",
    )]
    clip.checklist.items.append(ChecklistItem(
        question="gravity plausible?", kind="physics", passed=False,
        fix_instruction="fix gravity", mode="gravity_inertia",
    ))
    clip.checklist.items.append(ChecklistItem(
        question="semantic ok?", kind="semantic", passed=False,
    ))
    mode = _escape_hatch(clip)
    assert mode == PhysFailureMode.GRAVITY_INERTIA
    assert clip.physics_verdicts == []
    # The mirrored physics item flipped in lock-step (p1/p2 and m1 in sync)…
    assert all(i.passed for i in clip.checklist.items if i.kind == "physics")
    # …while unrelated items stay failed.
    assert not all(i.passed for i in clip.checklist.items)
    assert "physics:gravity_inertia" in clip.skipped_items


def test_escape_hatch_checklist_skip_reports_typed_mode():
    """Checklist-only physics skips (no verdict left) report their typed mode
    too — the old f'{kind}:{question[:40]}' record could never be matched
    against mode values."""
    clip = CandidateClip(shot_idx=0, video_path=Path("x.mp4"))
    clip.checklist.items.append(ChecklistItem(
        question="fluid plausible?", kind="physics", passed=False, mode="fluid",
    ))
    mode = _escape_hatch(clip)
    assert mode == PhysFailureMode.FLUID
    assert clip.checklist.items[0].passed


def test_board_keys_physics_items_to_their_verdict_modes(tmp_path: Path):
    spec = _spec("a ball is thrown")
    board = _board()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    board.review(clip, spec, None, fps=8)
    failed_phys = [i for i in clip.checklist.items
                   if i.kind == "physics" and not i.passed]
    assert failed_phys
    verdict_modes = {v.mode.value for v in clip.physics_verdicts}
    assert all(i.mode for i in failed_phys)
    assert {i.mode for i in failed_phys} <= verdict_modes


def test_distill_lesson_subtracts_loop_level_skips():
    """A mode that vanished from the final review because it was HATCHED is
    not 'resolved'. With no annotation fallback available, no lesson at all
    may be distilled from a fully-hatched episode."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="x")   # no annotation
    final = CandidateClip(shot_idx=0, video_path=Path("x.mp4"))
    initial = {PhysFailureMode.GRAVITY_INERTIA}
    assert _distill_lesson(spec, final, initial,
                           skipped_modes={PhysFailureMode.GRAVITY_INERTIA}) is None
    got = _distill_lesson(spec, final, initial, skipped_modes=set())
    assert got is not None and got[0] == PhysFailureMode.GRAVITY_INERTIA


class _AlwaysFailingMLLM:
    """No verdict ever passes — forces the loop into the escape hatch."""

    name = "always-failing"

    def assess_semantic(self, clip, spec):
        return [("never passes", False, "fix the unfixable")]

    def assess_physics(self, clip, spec, fps):
        return [PhysicsVerdict(
            mode=PhysFailureMode.GRAVITY_INERTIA,
            frame_range=(0, max(1, int(round(spec.duration * fps)))),
            severity=0.95,
            suggested_intervention="impossible fix",
        )]

    def compare(self, a, b, spec):
        return 0


def test_hatched_episode_never_distills_skill(tmp_path: Path):
    """F3b end-to-end: the escape hatch makes the final board pass (so
    `converged` flips True), but the loop-level `escape_hatched` flag must
    veto skill distillation."""
    spec = _spec("a ball is thrown")
    judge = _AlwaysFailingMLLM()
    weights = {"m1_semantic": 0.5, "p1_physics": 0.5}
    board = ReviewBoard(
        critics=[SemanticCritic(mllm=judge), PhysicsCritic(mllm=judge)],
        metric_tool=MetricTool(weights=weights),
    )
    skills = SkillLibrary(tmp_path / "skills.jsonl")
    res = generate_shot(
        spec, board, GeneratorAgent(), RefinerAgent(), VerifierAgent(),
        tmp_path, skill_library=skills,
        max_revisions=4, k_retries=1, n_candidates=1,
    )
    assert res.escape_hatched
    assert "gravity_inertia" in res.skipped_modes
    assert res.distilled_skill_id == ""
    assert len(skills.by_class("creation")) == 0
    assert res.clip.accepted        # still "the clip we ship" (assembly needs it)


# ─────────────────────────────────────────────────────────────────────────────
# F1 (watchdog half) — post-acceptance audit is log-only
# ─────────────────────────────────────────────────────────────────────────────
def test_post_accept_audit_is_log_only(tmp_path: Path):
    spec = _spec("a ball falls to the ground", duration=2.0)
    board = ReviewBoard([PhysicsConsistencyCritic()])
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    board.review(clip, spec, None, fps=8)
    verdicts_before = list(clip.physics_verdicts)
    scores_before = dict(clip.metric_scores)
    _post_accept_audit(accepted=clip, spec=spec, board=board,
                       asset_memory=None, fps=8, strictness=3.0)
    # The accepted clip's review state is untouched (the audit ran on a copy)
    # and the annotation strictness is restored.
    assert clip.physics_verdicts == verdicts_before
    assert clip.metric_scores == scores_before
    assert spec.physics_annotation.strictness == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# F5 — md5-stable skill names
# ─────────────────────────────────────────────────────────────────────────────
def test_skill_name_is_md5_derived_and_deterministic():
    spec = _spec("a ball is thrown and bounces off a wall")
    name = _stable_skill_name(spec)
    digest = hashlib.md5(spec.prompt.encode("utf-8")).hexdigest()[:5]
    # md5-derived: independent of PYTHONHASHSEED, stable across processes.
    assert name.endswith(f"__{digest}")
    assert _stable_skill_name(spec) == name
    assert name.startswith("skill_")


# ─────────────────────────────────────────────────────────────────────────────
# F6 — lesson coupling actually flows into distilled skills
# ─────────────────────────────────────────────────────────────────────────────
def test_injected_lesson_ids_flow_into_distilled_skill(tmp_path: Path):
    lessons = LessonLibrary(tmp_path / "lessons.jsonl")
    lesson = lessons.add(
        "a ball is thrown and bounces off a wall",
        "enforce gravity arc + bounce response",
        PhysFailureMode.GRAVITY_INERTIA,
    )
    specs = plan_shots(
        "a ball is thrown and bounces off a wall", AssetMemory(),
        ScreenwriterAgent(config={"n_shots": 1, "max_shots": 1}),
        DirectorAgent(config={"shot_duration": 1.0}),
        PhysicsPlannerAgent(), tmp_path, lesson_library=lessons,
    )
    spec = specs[0]
    assert lesson.lesson_id in spec.injected_lesson_ids   # ids populated

    skills = SkillLibrary(tmp_path / "skills.jsonl")
    res = generate_shot(
        spec, _board(), GeneratorAgent(), RefinerAgent(), VerifierAgent(),
        tmp_path, lesson_library=lessons, skill_library=skills,
        max_revisions=6, k_retries=2,
    )
    assert res.distilled_skill_id
    skill = next(s for s in skills.skills
                 if s.skill_id == res.distilled_skill_id)
    assert lesson.lesson_id in skill.coupled_lesson_ids


# ─────────────────────────────────────────────────────────────────────────────
# F7 — write gate honesty
# ─────────────────────────────────────────────────────────────────────────────
def test_gate_requires_converged(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    store.propose(StateTransition(
        entity_id=ident.entity_id, shot_idx=1, field="condition",
        old="", new="wet", cause="test"))
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    clip = _accepted_clip(tmp_path, 1, spec.prompt)
    out = store.commit_gated(clip, spec, VerificationWriteGate(),
                             converged=False)
    assert out == {"committed": 0, "rejected": 1}
    (t,) = store.history(ident.entity_id)
    assert "converge" in t.evidence


def test_gate_rejects_hatched_render(tmp_path: Path):
    gate = VerificationWriteGate()
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    t = StateTransition(entity_id="E1", shot_idx=1, field="condition",
                        old="", new="wet", cause="test")
    for skip in ("physics:gravity_inertia", "consistency:identity drift?"):
        clip = _accepted_clip(tmp_path, 1, spec.prompt)
        clip.skipped_items.append(skip)
        ok, evidence = gate.confirm(t, clip, spec, converged=True)
        assert not ok
        assert "escape hatch" in evidence


def test_gate_evidence_is_clip_body_only(tmp_path: Path):
    """The prompt itself is no longer evidence: a transition whose value
    appears only in spec.prompt (not in the rendered artifact) is rejected —
    the old prompt-echo tautology would have committed it."""
    gate = VerificationWriteGate()
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    t = StateTransition(entity_id="E1", shot_idx=1, field="condition",
                        old="", new="wet", cause="test")
    clip = _accepted_clip(tmp_path, 1, "the hero in bright sunshine")
    ok, evidence = gate.confirm(t, clip, spec, converged=True)
    assert not ok
    assert "absent" in evidence
    # And a body that DOES show it commits.
    clip2 = _accepted_clip(tmp_path, 2, "the hero is wet")
    spec2 = ShotSpec(shot_idx=2, duration=2.0, prompt="the hero is wet")
    t2 = StateTransition(entity_id="E1", shot_idx=2, field="condition",
                         old="", new="wet", cause="test")
    ok2, _ = gate.confirm(t2, clip2, spec2, converged=True)
    assert ok2


# ─────────────────────────────────────────────────────────────────────────────
# F8 — identity-anchor enrichment on collision
# ─────────────────────────────────────────────────────────────────────────────
def test_register_enriches_empty_fields_on_name_collision(tmp_path: Path):
    """anchor 'hero.png' + prompt noun 'hero': whichever order they register
    in, the anchor's reference paths must survive."""
    path = tmp_path / "entities.jsonl"
    store = EntityStore(path)
    bare = store.register(make_identity("hero", description="planned entity"))
    assert bare.reference_paths == []
    enriched = store.register(make_identity("hero", ["/refs/hero.png"]))
    assert enriched.entity_id == bare.entity_id
    assert enriched.reference_paths == ["/refs/hero.png"]   # backfilled
    assert enriched.description == "planned entity"          # non-empty kept
    # Non-empty fields are frozen: a third registration cannot overwrite.
    again = store.register(make_identity("hero", ["/refs/OTHER.png"], "imposter"))
    assert again.reference_paths == ["/refs/hero.png"]
    assert again.description == "planned entity"
    # Persisted.
    reloaded = EntityStore(path)
    assert reloaded.identities[bare.entity_id].reference_paths == ["/refs/hero.png"]


# ─────────────────────────────────────────────────────────────────────────────
# F9 — run-scoped entity log
# ─────────────────────────────────────────────────────────────────────────────
def test_rejected_transition_is_reproposable_in_next_run(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    gate = VerificationWriteGate()
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")

    # Run 1: proposal rejected (loop did not converge).
    (t1,) = propose_transitions_from_spec(spec, store, run_id="run1")
    assert t1.run_id == "run1"
    store.propose(t1)
    bad_clip = _accepted_clip(tmp_path, 1, spec.prompt, accepted=True)
    out1 = store.commit_gated(bad_clip, spec, gate,
                              converged=False, run_id="run1")
    assert out1 == {"committed": 0, "rejected": 1}

    # Run 2: the SAME transition is re-proposable (run-scoped dedup) …
    proposals2 = propose_transitions_from_spec(spec, store, run_id="run2")
    assert [(t.field, t.new) for t in proposals2] == [("condition", "wet")]
    store.propose(proposals2[0])
    # … and a foreign pending proposal is NOT decided by run 2's gate pass.
    store.propose(StateTransition(
        entity_id=ident.entity_id, shot_idx=1, field="emotion",
        old="", new="happy", cause="other run", run_id="run3"))
    good_clip = _accepted_clip(tmp_path, 1, spec.prompt)
    out2 = store.commit_gated(good_clip, spec, gate,
                              converged=True, run_id="run2")
    assert out2 == {"committed": 1, "rejected": 0}
    statuses = {(t.run_id, t.status) for t in store.history(ident.entity_id)}
    assert statuses == {("run1", "rejected"), ("run2", "committed"),
                        ("run3", "proposed")}
    # State/history stay cross-run: the committed value is visible globally.
    assert store.current_state(ident.entity_id)["attributes"] == {"condition": "wet"}


# ─────────────────────────────────────────────────────────────────────────────
# F11 — atomic persistence
# ─────────────────────────────────────────────────────────────────────────────
def test_persistence_uses_tmp_then_replace(tmp_path: Path):
    spec = _spec("a ball is thrown")
    skills_path = tmp_path / "skills.jsonl"
    lib = SkillLibrary(skills_path)
    lib.distill("r", spec.prompt, spec.physics_annotation,
                CinematographyTags(), {}, weighted_total=0.9)
    assert skills_path.exists()
    assert not skills_path.with_suffix(".tmp").exists()   # renamed, not left over
    assert len(SkillLibrary(skills_path)) == 1

    entities_path = tmp_path / "entities.jsonl"
    store = EntityStore(entities_path)
    store.register(make_identity("hero"))
    assert entities_path.exists()
    assert not entities_path.with_suffix(".tmp").exists()
    assert len(EntityStore(entities_path).identities) == 1


# ─────────────────────────────────────────────────────────────────────────────
# F12 — version/evolve honesty + lifecycle wiring
# ─────────────────────────────────────────────────────────────────────────────
def test_legacy_redistill_updates_thresholds(tmp_path: Path):
    """F12a: the no-admission re-distill path must update
    acceptance_thresholds exactly like the admission path does."""
    spec = _spec("a ball is thrown")
    lib = SkillLibrary(tmp_path / "skills.jsonl")          # legacy: no admission
    s1 = lib.distill("r", spec.prompt, spec.physics_annotation,
                     CinematographyTags(), {"weighted_total": 0.5},
                     weighted_total=0.6)
    assert s1.acceptance_thresholds == {"weighted_total": 0.5}
    s2 = lib.distill("r", spec.prompt, spec.physics_annotation,
                     CinematographyTags(), {"weighted_total": 0.9},
                     weighted_total=0.9)
    assert s2.version == 2
    assert s2.acceptance_thresholds == {"weighted_total": 0.9}


def test_distill_embedding_matches_reload(tmp_path: Path):
    """F12c: distill and _load embed the same text (name + triggers), so
    retrieval ranking is identical across a process restart."""
    spec = _spec("a ball is thrown and bounces off a wall")
    path = tmp_path / "skills.jsonl"
    lib = SkillLibrary(path)
    s = lib.distill("r", spec.prompt, spec.physics_annotation,
                    CinematographyTags(), {}, weighted_total=0.9)
    reloaded = SkillLibrary(path).skills[0]
    assert np.allclose(s.embedding, reloaded.embedding)


def test_matched_skill_outcome_recorded(tmp_path: Path):
    """F12d: a shot that ran with a matched skill feeds its episode outcome
    back into the skill's perf EMA via record_outcome."""
    spec = _spec("a ball is thrown and bounces off a wall")
    skills = SkillLibrary(tmp_path / "skills.jsonl")
    matched = skills.distill("incumbent", spec.prompt, spec.physics_annotation,
                             CinematographyTags(), {}, weighted_total=0.0)
    matched.perf_score = 0.0
    spec.matched_skill = matched
    res = generate_shot(
        spec, _board(), GeneratorAgent(), RefinerAgent(), VerifierAgent(),
        tmp_path, skill_library=skills, max_revisions=6, k_retries=2,
    )
    final_total = res.clip.metric_scores["weighted_total"]
    expected = skills.perf_ema_alpha * final_total
    assert abs(skills._by_id[matched.skill_id].perf_score - expected) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# F13 / F14 — config threading + this-run report deltas
# ─────────────────────────────────────────────────────────────────────────────
def test_run_report_counts_this_run_deltas(tmp_path: Path):
    """F14: '*_learned' are THIS-RUN deltas; cumulative sizes live in
    '*_total'. A second run on the same memory_dir re-distills (version bump,
    not a new entry) so its delta is 0 while the total persists."""
    from maestro.config import load_config
    from maestro.pipeline.run import run_maestro

    cfg = load_config()
    mem = tmp_path / "memory"
    r1 = run_maestro("a ball is thrown and bounces off a wall",
                     tmp_path / "r1.mp4", config=cfg,
                     cache_dir=tmp_path / "c1", memory_dir=mem)["report"]
    assert r1["skills_learned"] >= 1
    assert r1["skills_total"] >= r1["skills_learned"]
    assert r1["lessons_total"] >= r1["lessons_learned"] >= 1
    assert "skills_evicted" in r1                  # F12d: lifecycle pass ran

    r2 = run_maestro("a ball is thrown and bounces off a wall",
                     tmp_path / "r2.mp4", config=cfg,
                     cache_dir=tmp_path / "c2", memory_dir=mem)["report"]
    assert r2["skills_learned"] == 0               # no NEW skill this run
    assert r2["skills_total"] >= 1                 # cumulative survives


def test_config_severity_threshold_blocks_distill(tmp_path: Path):
    """F13b: memory.skill_distill_severity_threshold is actually threaded
    into should_distill — an impossible bar yields zero distilled skills."""
    from maestro.config import load_config
    from maestro.pipeline.run import run_maestro

    cfg = load_config(overrides={
        "memory": {"skill_distill_severity_threshold": 0.99},
    })
    r = run_maestro("a ball is thrown and bounces off a wall",
                    tmp_path / "out.mp4", config=cfg,
                    cache_dir=tmp_path / "cache",
                    memory_dir=tmp_path / "memory")["report"]
    assert r["skills_learned"] == 0
    assert all(not s["distilled_skill_id"] for s in r["shots"])


# ─────────────────────────────────────────────────────────────────────────────
# F15 — verifier tie-break de-duplication
# ─────────────────────────────────────────────────────────────────────────────
def test_verifier_tiebreak_does_not_double_count_physics():
    """One physics defect = one verdict + one mirrored checklist item. Under
    the old counting a candidate with 1 physics defect (raw count 2) could
    not beat an equal-total best with 2 semantic defects (raw count 2)."""
    best = CandidateClip(shot_idx=0, video_path=Path("a.mp4"))
    best.metric_scores = {"weighted_total": 0.5}
    best.checklist.items += [
        ChecklistItem(question="s1", kind="semantic", passed=False),
        ChecklistItem(question="s2", kind="semantic", passed=False),
    ]
    cand = CandidateClip(shot_idx=0, video_path=Path("b.mp4"))
    cand.metric_scores = {"weighted_total": 0.5}
    cand.physics_verdicts = [PhysicsVerdict(
        mode=PhysFailureMode.COLLISION, frame_range=(0, 4),
        severity=0.5, suggested_intervention="fix")]
    cand.checklist.items.append(ChecklistItem(
        question="mirror", kind="physics", passed=False,
        fix_instruction="fix", mode="collision"))
    assert VerifierAgent().is_better(cand, best)    # 1 defect < 2 defects


# ─────────────────────────────────────────────────────────────────────────────
# F16 — keyframe aliasing
# ─────────────────────────────────────────────────────────────────────────────
def test_keyframes_include_seed_so_candidates_do_not_alias(tmp_path: Path):
    spec = _spec("a ball is thrown")
    g = GeneratorAgent()
    c0 = g.run(spec, tmp_path, revision=1, seed=0, fps=8)
    body0 = c0.keyframes[0].read_text(encoding="utf-8")
    c1 = g.run(spec, tmp_path, revision=1, seed=1, fps=8)
    assert set(map(str, c0.keyframes)).isdisjoint(set(map(str, c1.keyframes)))
    # seed-0's keyframes were not overwritten by seed-1's generation
    assert c0.keyframes[0].read_text(encoding="utf-8") == body0
    assert "_s0_" in c0.keyframes[0].name and "_s1_" in c1.keyframes[0].name
