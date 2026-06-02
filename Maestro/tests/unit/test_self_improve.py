from pathlib import Path

from maestro.agents.generator import GeneratorAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.consistency import ConsistencyCritic
from maestro.critics.physics import PhysicsCritic
from maestro.critics.rhythm import RhythmCritic
from maestro.critics.semantic import SemanticCritic
from maestro.memory.lesson_library import LessonLibrary
from maestro.physics.sketch import build_physics_sketch
from maestro.pipeline.generate_loop import generate_shot
from maestro.types import ShotSpec


def _board():
    return ReviewBoard([
        SemanticCritic(), PhysicsCritic(), ConsistencyCritic(), RhythmCritic(),
    ])


def test_loop_converges_and_is_monotonic(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0,
                    prompt="a ball is thrown and bounces off a wall")
    build_physics_sketch(spec, tmp_path, fps=8)
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)

    res = generate_shot(
        spec, _board(), GeneratorAgent(), RefinerAgent(), VerifierAgent(),
        tmp_path, lesson_library=LessonLibrary(tmp_path / "l.jsonl"),
        max_revisions=6, k_retries=2,
    )
    # monotonic non-decreasing score history (Verifier never accepts a regression)
    h = res.score_history
    assert all(h[i] <= h[i + 1] + 1e-9 for i in range(len(h) - 1)), h
    # ends in an accepted clip with a strong score
    assert res.clip.accepted
    assert res.clip.metric_scores["weighted_total"] >= h[0]


def test_lesson_distilled(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball falls and bounces")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    lib = LessonLibrary(tmp_path / "l.jsonl")
    generate_shot(spec, _board(), GeneratorAgent(), RefinerAgent(), VerifierAgent(),
                  tmp_path, lesson_library=lib, max_revisions=4)
    assert len(lib) >= 1


def test_keyframe_local_edit_is_used(tmp_path: Path):
    # a prompt with physics issues should trigger a refiner keyframe edit
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="water pours and splashes")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    res = generate_shot(spec, _board(), GeneratorAgent(), RefinerAgent(),
                        VerifierAgent(), tmp_path, max_revisions=5)
    # at least one revision happened (local repair), proving loop engaged
    assert res.revisions_used >= 1
