from pathlib import Path

from maestro.agents.director import DirectorAgent
from maestro.agents.generator import GeneratorAgent
from maestro.agents.physics_planner import PhysicsPlannerAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.consistency import ConsistencyCritic
from maestro.critics.physics import PhysicsCritic
from maestro.critics.rhythm import RhythmCritic
from maestro.critics.semantic import SemanticCritic
from maestro.memory.lesson_library import LessonLibrary
from maestro.models.video_gen import MockVideoGenClient
from maestro.physics.annotate import annotate_physics
from maestro.pipeline.generate_loop import generate_shot
from maestro.types import ShotSpec


def _board():
    return ReviewBoard([
        SemanticCritic(), PhysicsCritic(), ConsistencyCritic(), RhythmCritic(),
    ])


def test_loop_converges_and_is_monotonic(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0,
                    prompt="a ball is thrown and bounces off a wall")
    annotate_physics(spec)
    spec.physics_annotation = annotate_physics(spec)

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
    spec.physics_annotation = annotate_physics(spec)
    lib = LessonLibrary(tmp_path / "l.jsonl")
    generate_shot(spec, _board(), GeneratorAgent(), RefinerAgent(), VerifierAgent(),
                  tmp_path, lesson_library=lib, max_revisions=4)
    assert len(lib) >= 1


def test_keyframe_local_edit_is_used(tmp_path: Path):
    # a prompt with physics issues should trigger a refiner keyframe edit
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="water pours and splashes")
    spec.physics_annotation = annotate_physics(spec)
    res = generate_shot(spec, _board(), GeneratorAgent(), RefinerAgent(),
                        VerifierAgent(), tmp_path, max_revisions=5)
    # at least one revision happened (local repair), proving loop engaged
    assert res.revisions_used >= 1


# ── RepairRouter wiring: Tier-1 picks edit_clip when an edit-capable backend +
#    a physics motion verdict are present; degrades to regenerate otherwise. ──
class _EditCapVideoGen(MockVideoGenClient):
    """Mock backend that also exposes the 'edit' capability + records edit_video."""

    def __init__(self):
        super().__init__(name="mock-edit-gen")
        self.edit_calls: list[dict] = []

    def capabilities(self):
        return {"t2v", "i2v", "edit"}

    def edit_video(self, prompt, video_path, out_path, backend="runway",
                   task="depth", seed=0):
        self.edit_calls.append({"prompt": prompt, "video_path": str(video_path),
                                "backend": backend})
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"MOCK EDIT\nprompt={prompt}\n", encoding="utf-8")
        return out


class _NoOpRefiner(RefinerAgent):
    """A refiner that proposes NO local fix, so Tier-0 can never improve and the
    loop is forced to escalate to Tier-1 (where the RepairRouter runs)."""

    def plan(self, clip):
        return {"extra_prompt": "", "edit_keyframe_idx": None, "edit_instruction": ""}


def test_tier1_invokes_edit_clip_when_edit_capable(tmp_path: Path):
    """With an edit-capable backend and a physics motion verdict ('a ball falls'
    → gravity_inertia), Tier-1's RepairRouter routes to edit_clip and actually
    invokes video_gen.edit_video. Deterministic + fast (mock MLLM, no network).
    A no-op refiner forces escalation past Tier-0 so Tier-1 is reached."""
    gen = _EditCapVideoGen()
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball falls and bounces")
    spec.physics_annotation = annotate_physics(spec)
    generate_shot(
        spec, _board(), GeneratorAgent(video_gen=gen), _NoOpRefiner(),
        VerifierAgent(), tmp_path,
        physics_planner=PhysicsPlannerAgent(), director=DirectorAgent(),
        max_revisions=4, k_retries=1,
    )
    assert gen.edit_calls, "Tier-1 should have invoked edit_video for a motion verdict"
    assert gen.edit_calls[0]["backend"] == "runway"


def test_tier1_pure_mock_never_edits(tmp_path: Path):
    """The pure mock pipeline (caps={t2v,i2v}, no source shots) must NEVER route
    to a richer action — Tier-1 stays the current regenerate behaviour."""
    gen = _EditCapVideoGen()
    # Strip the edit capability → mock-equivalent backend.
    gen.capabilities = lambda: {"t2v", "i2v"}  # type: ignore
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball falls and bounces")
    spec.physics_annotation = annotate_physics(spec)
    generate_shot(
        spec, _board(), GeneratorAgent(video_gen=gen), RefinerAgent(),
        VerifierAgent(), tmp_path,
        physics_planner=PhysicsPlannerAgent(), director=DirectorAgent(),
        max_revisions=4, k_retries=1,
    )
    assert not gen.edit_calls, "mock pipeline must not invoke edit_video"
