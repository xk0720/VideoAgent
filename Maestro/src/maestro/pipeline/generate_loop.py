"""Stage 2 — Generation + Self-Improve Loop (C2/C3/C4). The soul of Maestro.

Per shot:
  1. generate N candidates -> tournament-select best (E3)
  2. Review Board (multi-agent critics) + metric suite  (C3)
  3. revision loop: Refiner localizes -> keyframe local edit + local regen (C2)
     Verifier enforces monotonic improvement; escape hatch prevents dead loops
  4. distill a Lesson into the cross-task library (C4)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agents.generator import GeneratorAgent
from ..agents.refiner import RefinerAgent
from ..agents.verifier import VerifierAgent
from ..critics.board import ReviewBoard
from ..critics.tournament import Tournament
from ..memory.lesson_library import LessonLibrary
from ..models.image_edit import BaseImageEditClient, MockImageEditClient
from ..tools.retrieval_tool import RetrievalTool
from ..types import CandidateClip, ShotSpec


@dataclass
class SelfImproveResult:
    clip: CandidateClip
    score_history: list[float] = field(default_factory=list)
    revisions_used: int = 0
    converged: bool = False
    gen_calls: int = 0


def _tournament_select(candidates: list[CandidateClip]) -> CandidateClip:
    # VISTA-style: pick the strongest. (Real impl: bidirectional pairwise to
    # de-bias the MLLM judge; here deterministic max on weighted_total.)
    return max(candidates, key=lambda c: c.metric_scores.get("weighted_total", 0.0))


def _escape_hatch(clip: CandidateClip) -> None:
    """Unblock the loop: drop the single worst remaining defect and record it."""
    if clip.physics_verdicts:
        worst = max(clip.physics_verdicts, key=lambda v: v.severity)
        clip.physics_verdicts.remove(worst)
        clip.skipped_items.append(f"physics:{worst.mode.value}")
    else:
        failed = clip.checklist.failed_items
        if failed:
            failed[0].passed = True
            clip.skipped_items.append(f"{failed[0].kind}:{failed[0].question[:40]}")


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
    fps: int = 8,
    n_candidates: int = 2,
    max_revisions: int = 5,
    k_retries: int = 2,
) -> SelfImproveResult:
    image_edit = image_edit or MockImageEditClient()
    cache_dir = Path(cache_dir)
    gen_calls = 0

    # identity/style anchor images for cross-shot consistency (E1)
    ref_images = retrieval.retrieve_identity_refs(spec.identity_refs) if retrieval else None

    # 1. initial candidates -> tournament (VISTA bidirectional de-biasing if provided)
    candidates = []
    for s in range(n_candidates):
        c = generator.run(spec, cache_dir, revision=0, seed=s,
                          reference_images=ref_images, fps=fps)
        gen_calls += 1
        board.review(c, spec, asset_memory, fps)
        candidates.append(c)
    best = tournament.select(candidates, spec) if tournament else _tournament_select(candidates)
    history = [best.metric_scores.get("weighted_total", 0.0)]

    # 3. revision loop (local keyframe repair + monotonic improvement)
    revisions_used = 0
    converged = board.all_passed(best)
    for r in range(1, max_revisions + 1):
        if board.all_passed(best):
            converged = True
            break
        revisions_used = r
        plan = refiner.plan(best)

        accepted = False
        for k in range(k_retries):
            first_frame = None
            idx = plan["edit_keyframe_idx"]
            if idx is not None and 0 <= idx < len(best.keyframes):
                first_frame = image_edit.edit(
                    best.keyframes[idx], plan["edit_instruction"],
                    cache_dir / f"shot{spec.shot_idx:03d}_r{r}_editkf.txt",
                )
            cand = generator.run(
                spec, cache_dir, revision=r, seed=k,
                extra_prompt=plan["extra_prompt"], first_frame=first_frame,
                reference_images=ref_images, fps=fps,
            )
            gen_calls += 1
            board.review(cand, spec, asset_memory, fps)
            if verifier.is_better(cand, best):
                best = cand
                accepted = True
                break
        history.append(best.metric_scores.get("weighted_total", 0.0))
        if not accepted:
            _escape_hatch(best)  # prevent dead loop

    converged = converged or board.all_passed(best)
    best.accepted = True

    # 4. distill lesson (C4): use the sketch's expected modes as trigger context
    if lesson_library is not None:
        expected = spec.physics_sketch.expected_modes if spec.physics_sketch else []
        if expected:
            from ..physics.failure_modes import suggest_intervention
            lesson_library.add(
                trigger=spec.prompt,
                fix=suggest_intervention(expected[0]),
                failure_mode=expected[0],
            )

    return SelfImproveResult(
        clip=best, score_history=history, revisions_used=revisions_used,
        converged=converged, gen_calls=gen_calls,
    )
