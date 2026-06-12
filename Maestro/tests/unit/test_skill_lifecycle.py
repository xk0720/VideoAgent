"""Unified skill lifecycle tests — admission ("skill CI"), versioning,
three skill classes, legacy JSONL back-compat, usage ledger.

Covers the unified skill abstraction (INNOVATION_PLAN_2026_06.md §2):
distill → ADMISSION → retrieve → execute → evaluate → evolve/evict, all
training-free. No GPU, no network — the judge is the deterministic mock.
"""
from __future__ import annotations

import json
from pathlib import Path

from maestro.memory.skill_admission import (
    AdmissionVerdict,
    MockSkillJudge,
    SkillAdmission,
)
from maestro.memory.skill_library import SkillLibrary
from maestro.physics.annotate import annotate_physics
from maestro.types import (
    CinematographyTags,
    PhysEntity,
    PhysFailureMode,
    ShotSpec,
    Skill,
)

PROMPT = "a ball is thrown and bounces off a wall"

GOOD_EVIDENCE = {
    "weighted_total": 0.9,
    "escalations": 0,
    "resolved_modes": [PhysFailureMode.GRAVITY_INERTIA.value],
    "converged": True,
}


def _sketch(prompt: str = PROMPT):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt=prompt)
    return spec, annotate_physics(spec)


def _skill(
    signature=(PhysFailureMode.GRAVITY_INERTIA,),
    triggers=("thrown", "bounces"),
    entities=None,
    thresholds=None,
) -> Skill:
    return Skill(
        skill_id="S_test_entry",
        name="projectile_bounce",
        physical_signature=list(signature),
        triggers=list(triggers),
        entities=(entities if entities is not None
                  else [PhysEntity(name="ball", motion_class="ballistic")]),
        acceptance_thresholds=dict(thresholds or {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Admission gates (skill_admission.py)
# ─────────────────────────────────────────────────────────────────────────────
def test_admission_accepts_good_evidence():
    """Converged Tier-0 episode with a strong metric total and a coherent
    entry must pass all three gates."""
    verdict = SkillAdmission().review(_skill(), GOOD_EVIDENCE)
    assert isinstance(verdict, AdmissionVerdict)
    assert verdict.passed
    assert verdict.reasons == []
    assert verdict.score > 0.9


def test_admission_rejects_non_converged_episode():
    """An escape-hatched (non-converged) episode is not admissible evidence —
    the recipe may have left defects behind."""
    evidence = dict(GOOD_EVIDENCE, converged=False)
    verdict = SkillAdmission().review(_skill(), evidence)
    assert not verdict.passed
    assert any("converge" in r for r in verdict.reasons)


def test_admission_rejects_weak_metric_total_and_escalations():
    adm = SkillAdmission()
    weak = adm.review(_skill(), dict(GOOD_EVIDENCE, weighted_total=0.3))
    assert not weak.passed
    assert any("weighted_total" in r for r in weak.reasons)
    escalated = adm.review(_skill(), dict(GOOD_EVIDENCE, escalations=2))
    assert not escalated.passed
    assert any("escalation" in r for r in escalated.reasons)


def test_admission_judge_rejects_incoherent_entry():
    """Judge gate: a FLUID signature over a purely ballistic entity (or an
    entry with no trigger cues) is internally incoherent."""
    incoherent = _skill(signature=(PhysFailureMode.FLUID,))
    verdict = SkillAdmission().review(incoherent, GOOD_EVIDENCE)
    assert not verdict.passed
    assert any(r.startswith("judge:") for r in verdict.reasons)
    no_triggers = _skill(triggers=())
    verdict2 = SkillAdmission().review(no_triggers, GOOD_EVIDENCE)
    assert not verdict2.passed


def test_admission_rejects_bar_lowering_regression(tmp_path: Path):
    """Regression gate ("skill CI"): a new skill with the same physical
    signature may not LOWER the library's accepted acceptance_thresholds."""
    spec, sketch = _sketch()
    lib = SkillLibrary(tmp_path / "skills.jsonl", admission=SkillAdmission())
    incumbent = lib.distill(
        "strict_recipe", spec.prompt, sketch, CinematographyTags(),
        thresholds={"weighted_total": 0.85},
        weighted_total=0.95, evidence=GOOD_EVIDENCE,
    )
    assert incumbent is not None
    lowered = lib.distill(
        "lax_recipe", spec.prompt, sketch, CinematographyTags(),
        thresholds={"weighted_total": 0.5},          # below the incumbent bar
        weighted_total=0.95, evidence=GOOD_EVIDENCE,
    )
    assert lowered is None
    assert len(lib.by_class("creation")) == 1


def test_admission_rejects_omitted_threshold_keys(tmp_path: Path):
    """F10: omitting a key the library bar declares must FAIL gate (b) — an
    empty-thresholds skill would otherwise bypass the regression check
    entirely while still competing in retrieval against the incumbent."""
    spec, sketch = _sketch()
    lib = SkillLibrary(tmp_path / "skills.jsonl", admission=SkillAdmission())
    incumbent = lib.distill(
        "strict_recipe", spec.prompt, sketch, CinematographyTags(),
        thresholds={"weighted_total": 0.85},
        weighted_total=0.95, evidence=GOOD_EVIDENCE,
    )
    assert incumbent is not None
    empty = lib.distill(
        "no_bar_recipe", spec.prompt, sketch, CinematographyTags(),
        thresholds={},                               # asserts nothing at all
        weighted_total=0.95, evidence=GOOD_EVIDENCE,
    )
    assert empty is None
    assert len(lib.by_class("creation")) == 1
    # The reason names the omitted key (same physical signature as incumbent).
    adm = SkillAdmission(library=lib)
    candidate = _skill(signature=tuple(sketch.expected_modes), thresholds={})
    reasons = adm._regression_reasons(candidate)
    assert any("omits threshold 'weighted_total'" in r for r in reasons)


# ─────────────────────────────────────────────────────────────────────────────
# Library-side lifecycle (skill_library.py)
# ─────────────────────────────────────────────────────────────────────────────
def test_rejected_skill_is_not_persisted(tmp_path: Path):
    spec, sketch = _sketch()
    path = tmp_path / "skills.jsonl"
    lib = SkillLibrary(path, admission=SkillAdmission())
    rejected = lib.distill(
        "bad_recipe", spec.prompt, sketch, CinematographyTags(), {},
        weighted_total=0.9, evidence=dict(GOOD_EVIDENCE, converged=False),
    )
    assert rejected is None
    assert len(lib) == 0
    # Fresh reload sees nothing on disk either.
    assert len(SkillLibrary(path)) == 0


def test_accepted_skill_carries_admission_record(tmp_path: Path):
    """Every admitted entry keeps an auditable record of WHY it got in —
    persisted across reloads."""
    spec, sketch = _sketch()
    path = tmp_path / "skills.jsonl"
    lib = SkillLibrary(path, admission=SkillAdmission())
    skill = lib.distill("recipe", spec.prompt, sketch, CinematographyTags(),
                        {}, weighted_total=0.9, evidence=GOOD_EVIDENCE)
    assert skill is not None
    assert skill.admission["passed"] is True
    assert skill.admission["judge"] == "mock-skill-judge"
    reloaded = SkillLibrary(path).skills[0]
    assert reloaded.admission["passed"] is True


def test_version_bumps_on_redistill(tmp_path: Path):
    """Re-distilling the same (name, signature) bumps `version` and re-runs
    admission — the bar is monotone across versions."""
    spec, sketch = _sketch()
    lib = SkillLibrary(tmp_path / "skills.jsonl", admission=SkillAdmission())
    v1 = lib.distill("recipe", spec.prompt, sketch, CinematographyTags(),
                     {"weighted_total": 0.8}, weighted_total=0.85,
                     evidence=GOOD_EVIDENCE)
    assert v1 is not None and v1.version == 1
    v2 = lib.distill("recipe", spec.prompt, sketch, CinematographyTags(),
                     {"weighted_total": 0.82}, weighted_total=0.9,
                     evidence=GOOD_EVIDENCE)
    assert v2 is not None
    assert v2.skill_id == v1.skill_id
    assert v2.version == 2
    # A version that tries to LOWER its own bar is rejected (entry untouched).
    v3 = lib.distill("recipe", spec.prompt, sketch, CinematographyTags(),
                     {"weighted_total": 0.6}, weighted_total=0.9,
                     evidence=GOOD_EVIDENCE)
    assert v3 is None
    assert lib.skills[0].version == 2


def test_three_skill_classes_retrievable_separately(tmp_path: Path):
    spec, sketch = _sketch()
    lib = SkillLibrary(tmp_path / "skills.jsonl")          # legacy: no admission
    lib.distill("recipe", spec.prompt, sketch, CinematographyTags(), {},
                weighted_total=0.9)
    lib.register_review_skill("physics_review_measurement", "measurement",
                              params={"violation_threshold": 0.4})
    lib.register_memory_skill("skill_ema_retention",
                              params={"eviction_floor": 0.4})
    assert len(lib.by_class("creation")) == 1
    assert len(lib.by_class("review")) == 1
    assert len(lib.by_class("memory")) == 1
    assert lib.find_review_skill("measurement") is not None
    # Physics-typed retrieval must surface ONLY creation skills.
    hits = lib.retrieve(spec.prompt, list(sketch.expected_modes), top_k=5)
    assert hits and all(h.skill_class == "creation" for h in hits)


def test_review_skill_registration_is_idempotent(tmp_path: Path):
    path = tmp_path / "skills.jsonl"
    lib = SkillLibrary(path)
    a = lib.register_review_skill("physics_review_vlm", "vlm")
    b = lib.register_review_skill("physics_review_vlm", "vlm")
    assert a.skill_id == b.skill_id
    assert len(lib) == 1
    # Cross-run idempotence: a second "run" on the same JSONL adds nothing.
    lib2 = SkillLibrary(path)
    lib2.register_review_skill("physics_review_vlm", "vlm")
    assert len(lib2) == 1


def test_legacy_v03_jsonl_record_loads_with_defaults(tmp_path: Path):
    """A persisted v0.3 record (no skill_class/version/admission fields)
    loads as a grandfathered version-1 creation skill."""
    path = tmp_path / "skills.jsonl"
    legacy = {
        "skill_id": "Sdeadbeef0123",
        "name": "legacy_projectile",
        "physical_signature": ["gravity_inertia"],
        "triggers": ["thrown", "bounces"],
        "entities": [{"name": "ball", "motion_class": "ballistic"}],
        "interactions": [],
        "cinematography_preset": {},
        "acceptance_thresholds": {"weighted_total": 0.8},
        "coupled_lesson_ids": [],
        "perf_score": 0.7,
        "uses": 3,
        "last_used_ts": 123.0,
        "parent_id": "",
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    lib = SkillLibrary(path)
    assert len(lib) == 1
    s = lib.skills[0]
    assert s.skill_class == "creation"
    assert s.version == 1
    assert s.admission == {}
    assert PhysFailureMode.GRAVITY_INERTIA in s.physical_signature


def test_mark_used_increments_usage_ledger(tmp_path: Path):
    path = tmp_path / "skills.jsonl"
    lib = SkillLibrary(path)
    skill = lib.register_review_skill("physics_review_measurement", "measurement")
    uses_before, ts_before = skill.uses, skill.last_used_ts
    lib.mark_used(skill.skill_id)
    assert skill.uses == uses_before + 1
    assert skill.last_used_ts >= ts_before
    lib.mark_used("S_does_not_exist")               # no-op, must not raise
    # Persisted: a reload sees the incremented counter.
    assert SkillLibrary(path).skills[0].uses == uses_before + 1


def test_mock_judge_is_deterministic():
    judge = MockSkillJudge()
    s = _skill(signature=(PhysFailureMode.FLUID,), triggers=())
    score1, problems1 = judge.review_entry(s)
    score2, problems2 = judge.review_entry(s)
    assert (score1, problems1) == (score2, problems2)
    assert len(problems1) == 2                       # no triggers + incoherent sig
