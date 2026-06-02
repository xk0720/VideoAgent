"""Stage 1 — Planning. Screenwriter -> Director -> PhysicsPlanner, then an
optional Validate->Correct loop (PlanValidatorAgent) before generation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..agents.director import DirectorAgent
from ..agents.physics_planner import PhysicsPlannerAgent
from ..agents.plan_validator import PlanValidatorAgent
from ..agents.screenwriter import ScreenwriterAgent
from ..memory.lesson_library import LessonLibrary
from ..types import AssetMemory, ShotSpec


def plan_shots(
    user_prompt: str,
    asset_memory: AssetMemory,
    screenwriter: ScreenwriterAgent,
    director: DirectorAgent,
    physics_planner: PhysicsPlannerAgent,
    cache_dir: Path,
    lesson_library: Optional[LessonLibrary] = None,
    plan_validator: Optional[PlanValidatorAgent] = None,
    max_plan_iters: int = 3,
    fps: int = 8,
) -> list[ShotSpec]:
    outline = screenwriter.run(user_prompt, asset_memory)
    specs = director.run(outline, asset_memory, lesson_library)

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
    return specs
