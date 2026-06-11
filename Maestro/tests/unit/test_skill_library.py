"""SkillLibrary (C7) tests — distill rule, typed retrieval, lifecycle."""
from __future__ import annotations

from pathlib import Path

from maestro.memory.skill_library import SkillLibrary
from maestro.physics.annotate import annotate_physics
from maestro.types import CinematographyTags, PhysFailureMode, ShotSpec


def _sketch(tmp_path: Path, prompt: str = "a ball is thrown and bounces off a wall"):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt=prompt)
    return spec, annotate_physics(spec)


# ─────────────────────────────────────────────────────────────────────────────
# Distillation rule (§4.1 (b))
# ─────────────────────────────────────────────────────────────────────────────
def test_skill_distill_when_tier0_converges_on_nontrivial_severity(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    assert lib.should_distill(escalations=0, converged=True,
                              initial_severity_max=0.7)
    assert not lib.should_distill(escalations=1, converged=True,
                                  initial_severity_max=0.7)         # escalated
    assert not lib.should_distill(escalations=0, converged=False,
                                  initial_severity_max=0.7)         # escape hatch fired
    assert not lib.should_distill(escalations=0, converged=True,
                                  initial_severity_max=0.2)         # trivial


def test_skill_distill_writes_persisted_record(tmp_path: Path):
    spec, sketch = _sketch(tmp_path)
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    skill = lib.distill(
        name="projectile_bounce",
        spec_prompt=spec.prompt,
        annotation=sketch,
        cinematography=CinematographyTags(),
        thresholds={"weighted_total": 0.85},
        coupled_lesson_ids=["L_test_lesson"],
        weighted_total=0.9,
    )
    assert skill.skill_id.startswith("S")
    assert skill.physical_signature                                  # not empty
    assert PhysFailureMode.GRAVITY_INERTIA in skill.physical_signature

    # Reload from disk — same skill_id, all fields recovered.
    lib2 = SkillLibrary(tmp_path / "skills.jsonl")
    assert len(lib2) == 1
    s2 = lib2.skills[0]
    assert s2.skill_id == skill.skill_id
    assert "L_test_lesson" in s2.coupled_lesson_ids
    assert abs(s2.perf_score - 0.9) < 1e-3


def test_skill_distill_idempotent_reconfirms_perf(tmp_path: Path):
    spec, sketch = _sketch(tmp_path)
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    s1 = lib.distill("p", spec.prompt, sketch, CinematographyTags(), {},
                     weighted_total=0.6)
    s2 = lib.distill("p", spec.prompt, sketch, CinematographyTags(), {},
                     weighted_total=0.95)
    assert s2.skill_id == s1.skill_id                                # idempotent
    assert s2.perf_score > 0.6                                       # EMA pulled up
    assert s2.uses >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Typed retrieval (§4.1 (a)) — KEY differentiator vs Voyager
# ─────────────────────────────────────────────────────────────────────────────
def test_skill_retrieval_prefers_physical_signature_match(tmp_path: Path):
    """A skill whose physical_signature matches the target modes must
    outrank a text-only-similar skill — even when the text-only one has a
    closer prompt cosine. This is what Voyager / SkillWeaver can't do."""
    spec_gi, sketch_gi = _sketch(tmp_path, "a stone is thrown skyward")
    spec_fluid, sketch_fluid = _sketch(tmp_path / "f", "water pours and splashes")
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    # Skill A: same prompt domain but matches FLUID signature only.
    lib.distill("fluid_pour", spec_fluid.prompt, sketch_fluid,
                CinematographyTags(), {}, weighted_total=0.9)
    # Skill B: matches GRAVITY_INERTIA signature.
    lib.distill("projectile", spec_gi.prompt, sketch_gi,
                CinematographyTags(), {}, weighted_total=0.9)
    # Query: a gravity scenario.
    hits = lib.retrieve("an apple falls from a tree",
                        [PhysFailureMode.GRAVITY_INERTIA], top_k=1)
    assert len(hits) == 1
    assert PhysFailureMode.GRAVITY_INERTIA in hits[0].physical_signature
    assert hits[0].name == "projectile"


def test_skill_retrieval_filters_by_signature_overlap(tmp_path: Path):
    """A skill with zero signature overlap must not be returned even if its
    text matches perfectly (min_signature_overlap=1 default)."""
    spec_fluid, sketch_fluid = _sketch(tmp_path, "water pours from a cup")
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    lib.distill("fluid_only", spec_fluid.prompt, sketch_fluid,
                CinematographyTags(), {}, weighted_total=0.85)
    # Query asks for a mode the library doesn't have.
    hits = lib.retrieve("water pours from a cup",
                        [PhysFailureMode.PENETRATION], top_k=3)
    assert hits == []


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle (§4.1 lifecycle — borrowed from SkillOps)
# ─────────────────────────────────────────────────────────────────────────────
def test_skill_age_and_evict_drops_underperformers(tmp_path: Path):
    spec, sketch = _sketch(tmp_path)
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    skill = lib.distill("weak", spec.prompt, sketch, CinematographyTags(), {},
                        weighted_total=0.2)
    skill.uses = 10                       # uses > 5 and perf_score < 0.4 → evict
    skill.perf_score = 0.3
    lib._persist()
    evicted = lib.age_and_evict()
    assert evicted == 1
    assert len(lib) == 0
