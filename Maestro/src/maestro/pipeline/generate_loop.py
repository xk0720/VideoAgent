"""Stage 2 — Generation + Hierarchical Self-Improve Loop (C2/C3/C4/C5).

The soul of Maestro. Per shot:

  1. generate N candidates -> bidirectional-tournament-select best (E3)
  2. Review Board (multi-agent critics) + metric suite               (C3)
  3. Hierarchical revision loop (C5, NEW v0.2):

       Tier 0  refiner.plan  →  image_edit + first_frame + extra_prompt
                                (cheapest: keyframe-level LOCAL edit, M3-style)
       Tier 1  physics_planner.replan(strictness)
                                (rebuild the sketch with gentler velocities so
                                 the conditional generator gets an easier target)
       Tier 2  director.refine_spec(hint)
                                (widen scope: slower / wider cinematography +
                                 prompt rewrite — VISTA-style but bounded)
       Tier 3  escape_hatch     (drop the worst remaining defect)

     We try every tier with k_retries candidates, and only escalate after the
     previous tier fails to produce a Verifier-accepted improvement. Verifier
     enforces strict monotonic improvement at every tier — no regressions slip
     through. After every accepted candidate we *reset to Tier 0* so the next
     revision starts cheap again (cost-amortized adaptive scope).

  4. distill a Lesson from the failure mode that was actually resolved (C4)

WHY HSI: VISTA always replans the whole segment (expensive); M3 only does local
patches (limited scope on static images). HSI = adaptive scope escalation, the
gap neither prior work fills.
"""
from __future__ import annotations

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
from ..memory.lesson_library import LessonLibrary
from ..models.image_edit import BaseImageEditClient, MockImageEditClient
from ..physics.failure_modes import suggest_intervention
from ..tools.retrieval_tool import RetrievalTool
from ..types import CandidateClip, PhysFailureMode, ShotSpec


@dataclass
class SelfImproveResult:
    clip: CandidateClip
    score_history: list[float] = field(default_factory=list)
    revisions_used: int = 0
    converged: bool = False
    gen_calls: int = 0
    tier_used: list[int] = field(default_factory=list)   # tier per revision (C5)
    escalations: int = 0                                  # # times we went past Tier 0


def _tournament_select(candidates: list[CandidateClip]) -> CandidateClip:
    """Fallback when no Tournament judge is provided (deterministic argmax)."""
    return max(candidates, key=lambda c: c.metric_scores.get("weighted_total", 0.0))


def _escape_hatch(clip: CandidateClip) -> Optional[PhysFailureMode]:
    """Drop the single worst remaining defect; return its mode if it was a physics one."""
    if clip.physics_verdicts:
        worst = max(clip.physics_verdicts, key=lambda v: v.severity)
        clip.physics_verdicts.remove(worst)
        clip.skipped_items.append(f"physics:{worst.mode.value}")
        return worst.mode
    failed = clip.checklist.failed_items
    if failed:
        failed[0].passed = True
        clip.skipped_items.append(f"{failed[0].kind}:{failed[0].question[:40]}")
    return None


def _worst_hint(clip: CandidateClip) -> str:
    """Pick the worst defect's intervention as the hint that drives Tier-2 replan."""
    if clip.physics_verdicts:
        worst = max(clip.physics_verdicts, key=lambda v: v.severity)
        return worst.suggested_intervention
    for item in clip.checklist.failed_items:
        if item.fix_instruction:
            return item.fix_instruction
    return ""


def _distill_lesson(
    spec: ShotSpec,
    final: CandidateClip,
    initial_modes: set[PhysFailureMode],
) -> Optional[tuple[PhysFailureMode, str]]:
    """Pick the failure mode that was ACTUALLY resolved during the loop.

    Old logic always used `spec.physics_sketch.expected_modes[0]` regardless of
    whether anything was fixed. New: a mode is "resolved" if it appeared in the
    INITIAL review (initial_modes) but not in the FINAL review and was not
    escape-hatched. We distill from one of those true wins. Fall back to the
    sketch's expected_modes[0] only if no fix was actually achieved.
    """
    final_modes = {v.mode for v in final.physics_verdicts}
    skipped = {
        item.split(":", 1)[1] for item in final.skipped_items if item.startswith("physics:")
    }
    resolved = [
        m for m in initial_modes if m not in final_modes and m.value not in skipped
    ]
    if resolved:
        m = resolved[0]
        return m, suggest_intervention(m)
    if spec.physics_sketch and spec.physics_sketch.expected_modes:
        m = spec.physics_sketch.expected_modes[0]
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
) -> tuple[Optional[CandidateClip], int]:
    """Replan the physics sketch with stricter constraints."""
    physics_planner.replan(spec, cache_dir, fps=fps, strictness=0.55)
    gen_calls = 0
    for k in range(k_retries):
        cand = generator.run(
            spec, cache_dir, revision=r, seed=100 + k,  # offset seeds so files differ
            extra_prompt="tighter physics sketch (slower trajectory)",
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
    fps: int = 8,
    n_candidates: int = 2,
    max_revisions: int = 5,
    k_retries: int = 2,
) -> SelfImproveResult:
    image_edit = image_edit or MockImageEditClient()
    cache_dir = Path(cache_dir)
    gen_calls = 0

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

    # Snapshot the initial physics-failure-mode set so lesson distillation can
    # later report which modes were ACTUALLY resolved (C4 fix).
    initial_modes: set[PhysFailureMode] = {v.mode for v in best.physics_verdicts}

    history = [best.metric_scores.get("weighted_total", 0.0)]
    tier_used: list[int] = []
    escalations = 0

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

        # Tier 1 — replan the physics sketch.
        if accepted_cand is None and physics_planner is not None:
            tier_for_round = 1
            escalations += 1
            accepted_cand, calls = _tier1_replan_physics(
                best=best, spec=spec, cache_dir=cache_dir, r=r, k_retries=k_retries,
                generator=generator, verifier=verifier, board=board,
                physics_planner=physics_planner, asset_memory=asset_memory,
                ref_images=ref_images, fps=fps,
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
        else:
            # Tier 3 — escape hatch. Drop the worst defect, then refresh metrics
            # so the Verifier's NEXT round compares against an honest total.
            _escape_hatch(best)
            board.recompute_metrics(best, spec, asset_memory, fps)
            history.append(best.metric_scores.get("weighted_total", 0.0))
            tier_used.append(3)

    converged = converged or board.all_passed(best)
    best.accepted = True

    # 3. Distill a lesson based on the mode that was ACTUALLY resolved (C4 fix).
    if lesson_library is not None:
        result = _distill_lesson(spec, best, initial_modes)
        if result is not None:
            mode, fix = result
            lesson_library.add(trigger=spec.prompt, fix=fix, failure_mode=mode)

    return SelfImproveResult(
        clip=best, score_history=history, revisions_used=revisions_used,
        converged=converged, gen_calls=gen_calls,
        tier_used=tier_used, escalations=escalations,
    )
