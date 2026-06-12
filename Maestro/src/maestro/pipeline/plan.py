"""Stage 1 — Planning. Screenwriter -> Director -> PhysicsPlanner, then an
optional Validate->Correct loop (PlanValidatorAgent) before generation, and
finally an optional Skill-retrieval pass that attaches a matching `Skill`
from the library to each ShotSpec (C7, v0.3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..agents.director import DirectorAgent
from ..agents.physics_planner import PhysicsPlannerAgent
from ..agents.plan_validator import PlanValidatorAgent
from ..agents.screenwriter import ScreenwriterAgent
from ..memory.lesson_library import LessonLibrary
from ..memory.skill_library import SkillLibrary
from ..types import AssetMemory, ShotSpec


def _attach_skills(
    specs: list[ShotSpec],
    skill_library: SkillLibrary,
    lesson_library: Optional[LessonLibrary] = None,
) -> None:
    """For each spec with a physics annotation, retrieve a Skill keyed on
    its expected_modes signature. When a skill matches:

      • set `spec.matched_skill` (downstream agents can inspect),
      • adopt the skill's cinematography preset (Director's choice may be
        too generic; an experienced skill carries a learned preference),
      • auto-inject the skill's `coupled_lesson_ids` (C7 §4.1 (c)).

    Skill matching is non-destructive to the prompt itself, so HSI's existing
    revision logic still functions identically.
    """
    for spec in specs:
        if spec.physics_annotation is None:
            continue
        modes = list(spec.physics_annotation.expected_modes)
        if not modes:
            continue
        hits = skill_library.retrieve(spec.prompt, modes, top_k=1)
        if not hits:
            continue
        skill = hits[0]
        spec.matched_skill = skill
        spec.cinematography = skill.cinematography_preset
        # Auto-inject coupled lessons that we actually have on hand.
        if lesson_library is not None:
            for lid in skill.coupled_lesson_ids:
                lesson = lesson_library.get(lid)
                if lesson is not None and lesson.fix not in spec.injected_lessons:
                    spec.injected_lessons.append(lesson.fix)
                    if lesson.lesson_id not in spec.injected_lesson_ids:
                        spec.injected_lesson_ids.append(lesson.lesson_id)


def plan_shots(
    user_prompt: str,
    asset_memory: AssetMemory,
    screenwriter: ScreenwriterAgent,
    director: DirectorAgent,
    physics_planner: PhysicsPlannerAgent,
    cache_dir: Path,
    lesson_library: Optional[LessonLibrary] = None,
    skill_library: Optional[SkillLibrary] = None,   # C7 (v0.3)
    plan_validator: Optional[PlanValidatorAgent] = None,
    max_plan_iters: int = 3,
    fps: int = 8,
) -> list[ShotSpec]:
    outline = screenwriter.run(user_prompt, asset_memory)
    specs = director.run(outline, asset_memory, lesson_library)

    # C7 lesson-coupling provenance: the Director injects lessons as FIX TEXTS
    # (spec.injected_lessons); map them back to stable lesson IDS here so a
    # skill distilled from this shot can carry `coupled_lesson_ids` that point
    # at the real LessonLibrary entries.
    if lesson_library is not None:
        fix_to_id = {l.fix: l.lesson_id for l in lesson_library.lessons}
        for spec in specs:
            for fix in spec.injected_lessons:
                lid = fix_to_id.get(fix)
                if lid and lid not in spec.injected_lesson_ids:
                    spec.injected_lesson_ids.append(lid)

    # Validate -> Correct loop (plan-level self-improvement). Cheap: runs before
    # any video synthesis. Skipped if no validator is provided (back-compat).
    if plan_validator is not None:
        for _ in range(max_plan_iters):
            passed, feedback = plan_validator.run(specs, asset_memory)
            if passed:
                break
            by_idx = {s.shot_idx: s for s in specs}
            for idx, issues in feedback.items():
                director.revise(by_idx[idx], asset_memory, issues)

    for spec in specs:
        physics_planner.run(spec, cache_dir, fps=fps)

    # C7 skill retrieval — after the sketch is built so expected_modes is
    # known, but before generation so downstream agents see matched_skill.
    if skill_library is not None and len(skill_library) > 0:
        _attach_skills(specs, skill_library, lesson_library)

    return specs
