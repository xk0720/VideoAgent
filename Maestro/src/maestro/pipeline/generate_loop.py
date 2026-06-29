"""Stage 2 — Generation + Hierarchical Self-Improve Loop (C2/C3/C4/C5).

The soul of Maestro. Per shot:

  1. generate N candidates -> bidirectional-tournament-select best (E3)
  2. Review Board (multi-agent critics) + metric suite               (C3)
  3. Hierarchical revision loop (C5, NEW v0.2):

       Tier 0  refiner.plan  →  image_edit + first_frame + extra_prompt
                                (cheapest: keyframe-level LOCAL edit, M3-style)
       Tier 1  physics_planner.replan + verdict-derived hints
                                (re-annotate from the ORIGINAL prompt and
                                 regenerate with anti-violation prompt hints
                                 built from the worst measured/judged verdict —
                                 the verification bar does NOT move)
       Tier 2  director.refine_spec(hint)
                                (widen scope: slower / wider cinematography +
                                 prompt rewrite — VISTA-style but bounded)
       Tier 3  escape_hatch     (drop the worst remaining defect)

     We try every tier with k_retries candidates, and only escalate after the
     previous tier fails to produce a Verifier-accepted improvement. Verifier
     enforces strict monotonic improvement at every tier — no regressions slip
     through. After every accepted candidate we *reset to Tier 0* so the next
     revision starts cheap again (cost-amortized adaptive scope).

     WHY TIER 1 NEVER TIGHTENS STRICTNESS: tightening the bar on a FAILING
     shot inverts the repair incentive — the stricter critic threshold makes
     same-quality candidates accrue MORE verdicts, p2 drops, and the monotonic
     Verifier rejects exactly the regenerations that would repair the defect.
     Strictness-tightening therefore lives only in the POST-ACCEPTANCE
     hardening pass (`physics.post_accept_strictness`): once a candidate is
     accepted at any tier ≥ 1, one extra review at the tighter bar runs as a
     log-only quality watchdog (it can never reject the accepted clip).

  4. distill a Lesson from the failure mode that was actually resolved (C4)

NOTE on `accepted`: `best.accepted = True` means "the clip we ship" (assembly
depends on it) — NOT "defect-free". Honest quality status travels separately:
`converged` (review board fully passed) and the escape-hatch record
(`escape_hatched`, `skipped_modes`, `clip.skipped_items`).

WHY HSI: VISTA always replans the whole segment (expensive); M3 only does local
patches (limited scope on static images). HSI = adaptive scope escalation, the
gap neither prior work fills.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agents.director import DirectorAgent
from ..agents.generator import GeneratorAgent
from ..agents.physics_planner import PhysicsPlannerAgent
from ..agents.refiner import RefinerAgent
from ..agents.verifier import VerifierAgent
from ..critics.board import ReviewBoard
from ..critics.tournament import Tournament
from ..logging_utils import get_logger
from ..memory.lesson_library import LessonLibrary
from ..memory.skill_library import SkillLibrary
from ..models.image_edit import BaseImageEditClient, MockImageEditClient
from ..physics.failure_modes import suggest_intervention
from ..tools.retrieval_tool import RetrievalTool
from ..types import CandidateClip, PhysFailureMode, ShotSpec

log = get_logger(__name__)


@dataclass
class SelfImproveResult:
    clip: CandidateClip
    score_history: list[float] = field(default_factory=list)
    revisions_used: int = 0
    converged: bool = False
    gen_calls: int = 0
    tier_used: list[int] = field(default_factory=list)   # tier per revision (C5)
    escalations: int = 0                                  # # times we went past Tier 0
    initial_severity_max: float = 0.0                     # snapshot for C7 distill
    distilled_skill_id: str = ""                          # C7 — empty if no distill
    distilled_lesson_id: str = ""                         # C4 — what was saved
    escape_hatched: bool = False                          # any Tier-3 skip this episode
    skipped_modes: list[str] = field(default_factory=list)  # typed physics skips


def _tournament_select(candidates: list[CandidateClip]) -> CandidateClip:
    """Fallback when no Tournament judge is provided (deterministic argmax)."""
    return max(candidates, key=lambda c: c.metric_scores.get("weighted_total", 0.0))


def _escape_hatch(clip: CandidateClip) -> Optional[PhysFailureMode]:
    """Drop the single worst remaining defect; return its physics mode if known.

    When a physics verdict is dropped, its MIRRORED failed checklist item
    (keyed by ReviewBoard._key_physics_items) is flipped too, so p1/p2 and m1
    never desynchronize. Checklist-item skips of physics kind report their
    typed mode the same way, so loop-level skip accounting sees them.
    """
    if clip.physics_verdicts:
        worst = max(clip.physics_verdicts, key=lambda v: v.severity)
        clip.physics_verdicts.remove(worst)
        clip.skipped_items.append(f"physics:{worst.mode.value}")
        # Flip the mirrored checklist item in lock-step with the verdict.
        for item in clip.checklist.failed_items:
            if item.kind == "physics" and item.mode == worst.mode.value:
                item.passed = True
                break
        return worst.mode
    failed = clip.checklist.failed_items
    if failed:
        item = failed[0]
        item.passed = True
        clip.skipped_items.append(f"{item.kind}:{item.question[:40]}")
        if item.kind == "physics" and item.mode:
            try:
                return PhysFailureMode(item.mode)
            except ValueError:  # unknown key — still skipped, just untyped
                return None
    return None


def _worst_hint(clip: CandidateClip) -> str:
    """Pick the worst defect's intervention as the hint that drives Tier-1/2 replans."""
    if clip.physics_verdicts:
        worst = max(clip.physics_verdicts, key=lambda v: v.severity)
        return worst.suggested_intervention
    for item in clip.checklist.failed_items:
        if item.fix_instruction:
            return item.fix_instruction
    return ""


def _tier1_hint(best: CandidateClip) -> str:
    """Verdict-derived anti-violation hint for Tier 1: the worst verdict's own
    intervention (_worst_hint) combined with the failure-mode library's
    generic intervention (suggest_intervention) when it adds anything."""
    hint = _worst_hint(best)
    if best.physics_verdicts:
        worst = max(best.physics_verdicts, key=lambda v: v.severity)
        lib_hint = suggest_intervention(worst.mode)
        if lib_hint and lib_hint not in hint:
            hint = f"{hint} | {lib_hint}" if hint else lib_hint
    return hint or (
        "one continuous passive trajectory per moving object; "
        "no mid-air direction changes"
    )


def _stable_skill_name(spec: ShotSpec) -> str:
    """Stable, human-readable skill name. The suffix is md5-derived from the
    prompt so the SAME prompt yields the SAME skill_id across processes —
    builtin hash() is PYTHONHASHSEED-salted and would duplicate skills in a
    shared memory_dir (and version bumps would never fire cross-run)."""
    modes_tag = "+".join(
        m.value for m in (spec.physics_annotation.expected_modes
                          if spec.physics_annotation else [])
    ) or "generic"
    digest = hashlib.md5(spec.prompt.encode("utf-8")).hexdigest()[:5]
    return f"skill_{modes_tag}__{digest}"


def _distill_lesson(
    spec: ShotSpec,
    final: CandidateClip,
    initial_modes: set[PhysFailureMode],
    skipped_modes: set[PhysFailureMode],
) -> Optional[tuple[PhysFailureMode, str]]:
    """Pick the failure mode that was ACTUALLY resolved during the loop.

    A mode is "resolved" if it appeared in the INITIAL review (initial_modes)
    but not in the FINAL review and was not escape-hatched at ANY point in the
    loop (`skipped_modes` is the loop-level typed accumulator — the final
    clip's own skip records are not enough, because an accepted candidate
    replaces the clip and drops earlier skip records). We distill from one of
    those true wins. Fall back to the annotation's expected_modes[0] only if
    no fix was actually achieved.
    """
    final_modes = {v.mode for v in final.physics_verdicts}
    resolved = [
        m for m in initial_modes if m not in final_modes and m not in skipped_modes
    ]
    if resolved:
        m = resolved[0]
        return m, suggest_intervention(m)
    if spec.physics_annotation and spec.physics_annotation.expected_modes:
        m = spec.physics_annotation.expected_modes[0]
        return m, suggest_intervention(m)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HSI tier implementations — each returns (accepted_candidate | None, gen_calls)
# ─────────────────────────────────────────────────────────────────────────────
def _tier0_local_edit(
    *,
    best: CandidateClip,
    spec: ShotSpec,
    cache_dir: Path,
    r: int,
    k_retries: int,
    generator: GeneratorAgent,
    refiner: RefinerAgent,
    verifier: VerifierAgent,
    board: ReviewBoard,
    image_edit: BaseImageEditClient,
    asset_memory,
    ref_images,
    fps: int,
) -> tuple[Optional[CandidateClip], int]:
    """Keyframe-level local edit (M3-style). The cheapest tier."""
    plan = refiner.plan(best)
    gen_calls = 0
    for k in range(k_retries):
        first_frame = None
        idx = plan["edit_keyframe_idx"]
        if idx is not None and 0 <= idx < len(best.keyframes):
            first_frame = image_edit.edit(
                best.keyframes[idx], plan["edit_instruction"],
                cache_dir / f"shot{spec.shot_idx:03d}_r{r}_t0_editkf.txt",
            )
        cand = generator.run(
            spec, cache_dir, revision=r, seed=k,
            extra_prompt=plan["extra_prompt"], first_frame=first_frame,
            reference_images=ref_images, fps=fps,
        )
        gen_calls += 1
        board.review(cand, spec, asset_memory, fps)
        if verifier.is_better(cand, best):
            return cand, gen_calls
    return None, gen_calls


def _tier1_replan_physics(
    *,
    best: CandidateClip,
    spec: ShotSpec,
    base_prompt: str,
    cache_dir: Path,
    r: int,
    k_retries: int,
    generator: GeneratorAgent,
    verifier: VerifierAgent,
    board: ReviewBoard,
    physics_planner: PhysicsPlannerAgent,
    asset_memory,
    ref_images,
    fps: int,
    retrieval: Optional[RetrievalTool] = None,
) -> tuple[Optional[CandidateClip], int]:
    """Tier 1 — the Review→Execute bridge: a RepairRouter picks ONE tool-backed
    repair ACTION from the worst review verdict, gated by what the backend +
    uploaded assets actually support, then executes it. This is richer than
    UniVA's single fixed prompt-regenerate (which UniVA re-decides via an
    ephemeral Act-LLM every run) — yet it DEGRADES to exactly that regenerate
    when no richer tool/asset applies, so the mock pipeline is unchanged.

    Every action still goes through board.review + verifier.is_better: the
    monotonic-improvement contract is UNCHANGED across all actions. The
    "regenerate_hint" branch is the original Tier-1 code verbatim (re-annotate
    from the ORIGINAL prompt snapshot at an UNCHANGED bar — never from the
    Tier-2-mutated `spec.prompt`; strictness stays 1.0, see module docstring)."""
    from ..agents.repair_router import RepairRouter

    caps = generator.video_gen.capabilities()
    has_source_shots = bool(asset_memory and asset_memory.video_shots)
    action = RepairRouter().choose(
        best, spec, capabilities=caps, has_source_shots=has_source_shots
    )
    log.info("shot %d r%d tier-1 repair action=%s (%s)",
             spec.shot_idx, r, action.action, action.reason)

    gen_calls = 0

    if action.action == "edit_clip":
        out = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_t1_edit.mp4"
        video_path = generator.video_gen.edit_video(
            prompt=action.hint, video_path=best.video_path,
            out_path=out, backend="runway",
        )
        cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path, revision=r)
        gen_calls += 1
        board.review(cand, spec, asset_memory, fps)
        if verifier.is_better(cand, best):
            return cand, gen_calls
        return None, gen_calls

    if action.action == "retrieve_replace" and retrieval is not None:
        shot_ids = retrieval.retrieve_source_shots(query=spec.prompt)
        for sid in shot_ids:
            shot = retrieval.memory.video_shots.get(sid)
            if shot is None or not shot.source_video:
                continue
            src = Path(shot.source_video)
            if not src.exists():
                continue
            cand = CandidateClip(shot_idx=spec.shot_idx, video_path=src, revision=r)
            gen_calls += 1
            board.review(cand, spec, asset_memory, fps)
            if verifier.is_better(cand, best):
                return cand, gen_calls
        return None, gen_calls

    if action.action == "extend_clip":
        out = cache_dir / f"shot{spec.shot_idx:03d}_r{r}_t1_extend.mp4"
        video_path = generator.video_gen.extend(
            prompt=action.hint, video_path=best.video_path, out_path=out,
            duration=max(1, int(round(spec.duration))),
        )
        cand = CandidateClip(shot_idx=spec.shot_idx, video_path=video_path, revision=r)
        gen_calls += 1
        board.review(cand, spec, asset_memory, fps)
        if verifier.is_better(cand, best):
            return cand, gen_calls
        return None, gen_calls

    # action.action == "regenerate_hint" (or any chosen-but-unreachable action):
    # the ORIGINAL Tier-1 behaviour, verbatim.
    physics_planner.replan(spec, cache_dir, fps=fps, strictness=1.0,
                           base_prompt=base_prompt)
    hint = action.hint or _tier1_hint(best)
    for k in range(k_retries):
        cand = generator.run(
            spec, cache_dir, revision=r, seed=100 + k,  # offset seeds so files differ
            extra_prompt=hint,
            reference_images=ref_images, fps=fps,
        )
        gen_calls += 1
        board.review(cand, spec, asset_memory, fps)
        if verifier.is_better(cand, best):
            return cand, gen_calls
    return None, gen_calls


def _tier2_replan_spec(
    *,
    best: CandidateClip,
    spec: ShotSpec,
    cache_dir: Path,
    r: int,
    k_retries: int,
    generator: GeneratorAgent,
    verifier: VerifierAgent,
    board: ReviewBoard,
    director: DirectorAgent,
    asset_memory,
    ref_images,
    fps: int,
) -> tuple[Optional[CandidateClip], int]:
    """Spec-level replan: rewrite cinematography + prompt (bounded VISTA-style)."""
    director.refine_spec(spec, _worst_hint(best))
    gen_calls = 0
    for k in range(k_retries):
        cand = generator.run(
            spec, cache_dir, revision=r, seed=200 + k,
            reference_images=ref_images, fps=fps,
        )
        gen_calls += 1
        board.review(cand, spec, asset_memory, fps)
        if verifier.is_better(cand, best):
            return cand, gen_calls
    return None, gen_calls


def _post_accept_audit(
    *,
    accepted: CandidateClip,
    spec: ShotSpec,
    board: ReviewBoard,
    asset_memory,
    fps: int,
    strictness: float,
) -> None:
    """Post-acceptance hardening (quality watchdog) — LOG-ONLY by design.

    Tightening the verification bar on a FAILING shot inverts the repair
    incentive (stricter bar → more verdicts → lower p2 → the monotonic
    Verifier rejects the candidates that fix the defect). Tightening on an
    ACCEPTED candidate is sound: the decision is already made, so the tighter
    pass can only surface borderline physics for the operator — it never
    rejects. Runs on a deep copy so the shipped clip's review state is
    untouched; the annotation strictness is restored afterwards.
    """
    ann = spec.physics_annotation
    if ann is None or strictness <= 1.0:
        return
    audit_clip = copy.deepcopy(accepted)
    prev = ann.strictness
    ann.strictness = strictness
    try:
        board.review(audit_clip, spec, asset_memory, fps)
    finally:
        ann.strictness = prev
    borderline = [
        (v.mode.value, round(v.severity, 3)) for v in audit_clip.physics_verdicts
    ]
    if borderline:
        log.info(
            "post-accept hardening shot %d (strictness=%.2f): %d borderline "
            "physics verdict(s) %s — log-only, the accepted clip stands",
            spec.shot_idx, strictness, len(borderline), borderline,
        )


# ─────────────────────────────────────────────────────────────────────────────
def generate_shot(
    spec: ShotSpec,
    board: ReviewBoard,
    generator: GeneratorAgent,
    refiner: RefinerAgent,
    verifier: VerifierAgent,
    cache_dir: Path,
    asset_memory=None,
    lesson_library: Optional[LessonLibrary] = None,
    image_edit: Optional[BaseImageEditClient] = None,
    tournament: Optional[Tournament] = None,
    retrieval: Optional[RetrievalTool] = None,
    physics_planner: Optional[PhysicsPlannerAgent] = None,   # enables Tier 1 (C5)
    director: Optional[DirectorAgent] = None,                # enables Tier 2 (C5)
    skill_library: Optional[SkillLibrary] = None,            # C7 (v0.3)
    task_id: str = "",                                       # for episodic tagging
    fps: int = 8,
    n_candidates: int = 2,
    max_revisions: int = 5,
    k_retries: int = 2,
    post_accept_strictness: float = 1.0,   # >1.0 enables the log-only watchdog
) -> SelfImproveResult:
    image_edit = image_edit or MockImageEditClient()
    cache_dir = Path(cache_dir)
    gen_calls = 0

    # Snapshot the ORIGINAL prompt before any tier can mutate it: Tier 2's
    # refine_spec appends '| plan-fix: <hint>' in place, and Tier-1 replans
    # must re-annotate from this snapshot, not from hint text.
    base_prompt = spec.prompt

    # Identity/style anchor images for cross-shot consistency (E1).
    ref_images = retrieval.retrieve_identity_refs(spec.identity_refs) if retrieval else None

    # 1. Initial candidates → bidirectional tournament.
    candidates = []
    for s in range(n_candidates):
        c = generator.run(spec, cache_dir, revision=0, seed=s,
                          reference_images=ref_images, fps=fps)
        gen_calls += 1
        board.review(c, spec, asset_memory, fps)
        candidates.append(c)
    best = tournament.select(candidates, spec) if tournament else _tournament_select(candidates)

    # Unified skill abstraction: the measurement verification tier is a
    # registered review skill (pipeline/run.py); ledger its usage when the
    # PhysicsConsistencyCritic actually produced MEASURED verdicts
    # (source="law_verifier"). Honest limitation: the loop only sees measured
    # reports that crossed the critic's violation threshold — sub-threshold
    # coverage lives in the trajectory log (we do not refactor the critic).
    if skill_library is not None and any(
        v.source == "law_verifier" for c in candidates for v in c.physics_verdicts
    ):
        measurement_skill = skill_library.find_review_skill("measurement")
        if measurement_skill is not None:
            skill_library.mark_used(measurement_skill.skill_id)

    # Snapshot the initial physics-failure-mode set so lesson distillation can
    # later report which modes were ACTUALLY resolved (C4 fix). The worst
    # initial severity also gates C7 skill distillation: only "real" recipes
    # (severity ≥ threshold) become skills, not no-op convergences.
    initial_modes: set[PhysFailureMode] = {v.mode for v in best.physics_verdicts}
    initial_severity_max = max(
        (v.severity for v in best.physics_verdicts), default=0.0,
    )

    history = [best.metric_scores.get("weighted_total", 0.0)]
    tier_used: list[int] = []
    escalations = 0
    # Loop-level escape-hatch accounting (F3): skip records must live on the
    # LOOP, not the clip — a later accepted candidate replaces the clip and
    # would silently drop the skip records.
    skipped_modes: set[PhysFailureMode] = set()
    escape_hatched = False

    # 2. Hierarchical revision loop (C5).
    revisions_used = 0
    converged = board.all_passed(best)
    for r in range(1, max_revisions + 1):
        if board.all_passed(best):
            converged = True
            break
        revisions_used = r

        accepted_cand: Optional[CandidateClip] = None
        tier_for_round = 0

        # Tier 0 — local keyframe edit (cheapest, always tried first).
        accepted_cand, calls = _tier0_local_edit(
            best=best, spec=spec, cache_dir=cache_dir, r=r, k_retries=k_retries,
            generator=generator, refiner=refiner, verifier=verifier, board=board,
            image_edit=image_edit, asset_memory=asset_memory,
            ref_images=ref_images, fps=fps,
        )
        gen_calls += calls

        # Tier 1 — regenerate with verdict-derived hints (bar unchanged).
        if accepted_cand is None and physics_planner is not None:
            tier_for_round = 1
            escalations += 1
            accepted_cand, calls = _tier1_replan_physics(
                best=best, spec=spec, base_prompt=base_prompt,
                cache_dir=cache_dir, r=r, k_retries=k_retries,
                generator=generator, verifier=verifier, board=board,
                physics_planner=physics_planner, asset_memory=asset_memory,
                ref_images=ref_images, fps=fps, retrieval=retrieval,
            )
            gen_calls += calls

        # Tier 2 — rewrite the ShotSpec (cinematography + prompt).
        if accepted_cand is None and director is not None:
            tier_for_round = 2
            escalations += 1
            accepted_cand, calls = _tier2_replan_spec(
                best=best, spec=spec, cache_dir=cache_dir, r=r, k_retries=k_retries,
                generator=generator, verifier=verifier, board=board,
                director=director, asset_memory=asset_memory,
                ref_images=ref_images, fps=fps,
            )
            gen_calls += calls

        if accepted_cand is not None:
            best = accepted_cand
            history.append(best.metric_scores.get("weighted_total", 0.0))
            tier_used.append(tier_for_round)
            # Post-acceptance hardening: only after an escalated acceptance,
            # only log-only (see _post_accept_audit for why never on failure).
            if tier_for_round >= 1:
                _post_accept_audit(
                    accepted=best, spec=spec, board=board,
                    asset_memory=asset_memory, fps=fps,
                    strictness=post_accept_strictness,
                )
        else:
            # Tier 3 — escape hatch. Drop the worst defect, then refresh metrics
            # so the Verifier's NEXT round compares against an honest total.
            mode = _escape_hatch(best)
            escape_hatched = True
            if mode is not None:
                skipped_modes.add(mode)
            board.recompute_metrics(best, spec, asset_memory, fps)
            history.append(best.metric_scores.get("weighted_total", 0.0))
            tier_used.append(3)

    converged = converged or board.all_passed(best)
    # `accepted` = "the clip we ship" (assembly consumes it), NOT "defect-free".
    # Honest status is `converged` + the escape-hatch record above.
    best.accepted = True

    # Strictness always returns to 1.0 when the loop exits (accepted or
    # budget-exhausted) — no replan/audit state may leak into the next shot
    # or the next run sharing this spec object.
    if spec.physics_annotation is not None:
        spec.physics_annotation.strictness = 1.0

    # Skill lifecycle (F12d): if planning matched a library skill for this
    # shot, feed the episode outcome back into its perf EMA.
    if skill_library is not None and spec.matched_skill is not None:
        skill_library.record_outcome(
            spec.matched_skill.skill_id,
            best.metric_scores.get("weighted_total", 0.0),
        )

    # 3. Distill a lesson based on the mode that was ACTUALLY resolved (C4 fix).
    distilled_lesson_id = ""
    if lesson_library is not None:
        result = _distill_lesson(spec, best, initial_modes, skipped_modes)
        if result is not None:
            mode, fix = result
            lesson = lesson_library.add(
                trigger=spec.prompt, fix=fix, failure_mode=mode,
                born_task_id=task_id,
            )
            distilled_lesson_id = lesson.lesson_id

    # 4. Distill a Skill (C7, v0.3) — see RESEARCH_MEMORY_SKILL.md §4.1 (b).
    # A skill is born only when HSI converged at Tier 0 with a non-trivial
    # initial physics challenge — i.e., the cheapest tier handled real physics
    # — and the escape hatch never fired (a hatched episode may have shipped
    # hidden defects; it is not admissible recipe evidence).
    distilled_skill_id = ""
    if (
        skill_library is not None
        and spec.physics_annotation is not None
        and skill_library.should_distill(
            escalations=escalations,
            converged=converged,
            initial_severity_max=initial_severity_max,
            escape_hatched=escape_hatched,
        )
    ):
        skill_name = _stable_skill_name(spec)
        coupled = list(spec.injected_lesson_ids)
        # Admission evidence — the REAL episode signals (metric suite total +
        # verifier-driven convergence/escalation record + which physics modes
        # the loop actually resolved), consumed by SkillAdmission ("skill CI").
        # Escape-hatched modes are NOT resolved modes (F3c).
        final_modes = {v.mode for v in best.physics_verdicts}
        evidence = {
            "weighted_total": best.metric_scores.get("weighted_total", 0.0),
            "escalations": escalations,
            "resolved_modes": sorted(
                m.value for m in initial_modes - final_modes - skipped_modes
            ),
            "converged": converged,
        }
        skill = skill_library.distill(
            name=skill_name,
            spec_prompt=spec.prompt,
            annotation=spec.physics_annotation,
            cinematography=spec.cinematography,
            thresholds={
                "weighted_total":
                    max(0.0, best.metric_scores.get("weighted_total", 0.0) * 0.9),
                "p1_physics":
                    max(0.0, best.metric_scores.get("p1_physics", 1.0) * 0.9),
            },
            coupled_lesson_ids=([distilled_lesson_id] if distilled_lesson_id else [])
                              + coupled,
            weighted_total=best.metric_scores.get("weighted_total", 0.0),
            evidence=evidence,
            # Phase-2: the distilled creation skill RECORDS the capability +
            # params that succeeded for this shot, so the next similar shot
            # reuses the routing decision (CapabilityRouter step (a)).
            gen_capability=spec.gen_capability,
            gen_params=dict(spec.gen_params),
        )
        # distill returns None when admission ("skill CI") rejects the entry.
        distilled_skill_id = skill.skill_id if skill is not None else ""

    return SelfImproveResult(
        clip=best, score_history=history, revisions_used=revisions_used,
        converged=converged, gen_calls=gen_calls,
        tier_used=tier_used, escalations=escalations,
        initial_severity_max=initial_severity_max,
        distilled_skill_id=distilled_skill_id,
        distilled_lesson_id=distilled_lesson_id,
        escape_hatched=escape_hatched,
        skipped_modes=sorted(m.value for m in skipped_modes),
    )
