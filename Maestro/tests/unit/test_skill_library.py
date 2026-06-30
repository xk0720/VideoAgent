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
                                  initial_severity_max=0.7)         # did not converge
    assert not lib.should_distill(escalations=0, converged=True,
                                  initial_severity_max=0.2)         # trivial


def test_skill_distill_refuses_escape_hatched_episode(tmp_path: Path):
    """F3b: `converged` can be True even though the escape hatch fired (a
    hatched defect is dropped from the review state, which can make the final
    board pass) — the separate `escape_hatched` flag must veto distillation."""
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    assert lib.should_distill(escalations=0, converged=True,
                              initial_severity_max=0.7, escape_hatched=False)
    assert not lib.should_distill(escalations=0, converged=True,
                                  initial_severity_max=0.7, escape_hatched=True)


def test_skill_distill_severity_threshold_is_configurable(tmp_path: Path):
    """F13b: the constructor's distill_severity_threshold is the default bar
    for should_distill (threaded from configs/default.yaml memory.*)."""
    strict = SkillLibrary(tmp_path / "skills.jsonl",
                          distill_severity_threshold=0.9)
    assert not strict.should_distill(escalations=0, converged=True,
                                     initial_severity_max=0.7)
    assert strict.should_distill(escalations=0, converged=True,
                                 initial_severity_max=0.95)


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


# ─────────────────────────────────────────────────────────────────────────────
# v0.4 — Repair skills: distilled, retrieved, replayed WORKFLOWS (the headline)
# ─────────────────────────────────────────────────────────────────────────────
class _FakeDefect:
    def __init__(self, fix_modality="motion", note="gravity_inertia (law_verifier)"):
        self.fix_modality = fix_modality
        self.note = note


class _FakeReport:
    def __init__(self, defect):
        self._d = defect

    def worst(self):
        return self._d


_WORKFLOW = [
    {"tool": "edit_clip", "args_template": {"prompt": "straighten the arc",
                                            "backend": "runway"}, "modality": "motion"},
    {"tool": "regenerate", "args_template": {"hint": "one continuous arc"},
     "modality": "motion"},
]
_EVIDENCE = {"weighted_total": 0.9, "escalations": 0, "converged": True,
             "defect_reduced": True}


def test_distill_repair_persists_and_loads_workflow_and_signature(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    skill = lib.distill_repair(
        name="fix_gravity__repair",
        defect_signature=["motion", "gravity_inertia"],
        repair_workflow=_WORKFLOW,
        evidence=_EVIDENCE,
        thresholds={"weighted_total": 0.8},
    )
    assert skill is not None
    assert skill.skill_class == "repair"
    assert skill.skill_id.startswith("R")
    assert skill.repair_workflow == _WORKFLOW
    assert set(skill.defect_signature) == {"motion", "gravity_inertia"}

    # Reload from disk — workflow + signature recovered, embedding rebuilt.
    lib2 = SkillLibrary(tmp_path / "skills.jsonl")
    assert len(lib2) == 1
    s2 = lib2.skills[0]
    assert s2.skill_class == "repair"
    assert s2.repair_workflow == _WORKFLOW
    assert set(s2.defect_signature) == {"motion", "gravity_inertia"}
    assert s2.embedding is not None


def test_distill_repair_idempotent_bumps_version(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    s1 = lib.distill_repair("r", ["motion"], _WORKFLOW, _EVIDENCE)
    s2 = lib.distill_repair("r", ["motion"], _WORKFLOW,
                            {**_EVIDENCE, "weighted_total": 0.95})
    assert s2.skill_id == s1.skill_id
    assert s2.version == 2
    assert len(lib) == 1


def test_distill_repair_rejects_empty_inputs(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    assert lib.distill_repair("r", [], _WORKFLOW, _EVIDENCE) is None
    assert lib.distill_repair("r", ["motion"], [], _EVIDENCE) is None
    assert len(lib) == 0


def test_retrieve_repair_matches_by_signature_overlap(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    lib.distill_repair("fix_gravity", ["motion", "gravity_inertia"],
                       _WORKFLOW, _EVIDENCE)
    # Worst defect is a motion / gravity_inertia defect → overlaps the signature.
    report = _FakeReport(_FakeDefect("motion", "gravity_inertia (law_verifier)"))
    hit = lib.retrieve_repair(report)
    assert hit is not None
    assert hit.name == "fix_gravity"


def test_retrieve_repair_returns_none_on_miss(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    lib.distill_repair("fix_gravity", ["motion", "gravity_inertia"],
                       _WORKFLOW, _EVIDENCE)
    # A style defect overlaps nothing in the signature.
    report = _FakeReport(_FakeDefect("content", "wrong palette"))
    assert lib.retrieve_repair(report) is None
    # Empty library → None.
    assert SkillLibrary(tmp_path / "empty.jsonl").retrieve_repair(report) is None


def test_retrieve_repair_tie_break_by_perf_then_recency(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    low = lib.distill_repair("low", ["motion"], _WORKFLOW,
                             {**_EVIDENCE, "weighted_total": 0.6})
    high = lib.distill_repair("high", ["motion"], _WORKFLOW,
                              {**_EVIDENCE, "weighted_total": 0.95})
    report = _FakeReport(_FakeDefect("motion", "x"))
    hit = lib.retrieve_repair(report)
    assert hit.skill_id == high.skill_id   # higher perf wins the overlap tie


def test_repair_skills_excluded_from_creation_retrieve(tmp_path: Path):
    spec, sketch = _sketch(tmp_path)
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    # A creation skill with a gravity signature + a repair skill.
    lib.distill("projectile", spec.prompt, sketch, CinematographyTags(), {},
                weighted_total=0.9)
    lib.distill_repair("fix_gravity", ["motion", "gravity_inertia"],
                       _WORKFLOW, _EVIDENCE)
    hits = lib.retrieve("a ball is thrown",
                        [PhysFailureMode.GRAVITY_INERTIA], top_k=5)
    # creation retrieve() must never surface a repair skill.
    assert all(h.skill_class == "creation" for h in hits)
    assert any(h.name == "projectile" for h in hits)


def test_repair_skill_exempt_from_perf_floor_eviction(tmp_path: Path):
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    skill = lib.distill_repair("r", ["motion"], _WORKFLOW, _EVIDENCE)
    skill.uses = 10
    skill.perf_score = 0.1   # below floor — would evict a creation skill
    lib._persist()
    assert lib.age_and_evict() == 0   # repair skills are perf-floor exempt
    assert len(lib) == 1
