"""Tests for v0.3 memory tiers 1/4/5 + A-MEM lesson evolution + MLM facade."""
from __future__ import annotations

from pathlib import Path

from maestro.memory.entity_store import EntityStore
from maestro.memory.episodic_store import EpisodicStore
from maestro.memory.lesson_library import LessonLibrary
from maestro.memory.multi_layer import MultiLayerMemory
from maestro.memory.preference_store import PreferenceStore
from maestro.types import PhysFailureMode


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4 — EntityStore (cross-run reuse)
# ─────────────────────────────────────────────────────────────────────────────
def test_entity_store_cross_run_reuse(tmp_path: Path):
    """Two distinct 'runs' (instances of EntityStore on the same JSONL) seeing
    the SAME canonical_name must yield the SAME entity_id — VideoMemory
    extended to cross-run scope."""
    path = tmp_path / "entities.jsonl"
    run1 = EntityStore(path)
    e1 = run1.find_or_create("hero", source_path="/data/d1/hero.png", task_id="t1")
    assert len(run1) == 1
    # Fresh instance, same JSONL — simulates a second day's run.
    run2 = EntityStore(path)
    assert len(run2) == 1                                 # loaded from disk
    e2 = run2.find_or_create("hero", source_path="/data/d2/hero.png", task_id="t2")
    assert e2.entity_id == e1.entity_id                   # cross-run reuse
    # Appearance log accumulated.
    assert any(a.get("task_id") == "t1" for a in e2.appearance_log)
    assert any(a.get("task_id") == "t2" for a in e2.appearance_log)
    # Source paths from BOTH runs are tracked.
    assert "/data/d1/hero.png" in e2.source_paths
    assert "/data/d2/hero.png" in e2.source_paths


def test_entity_store_distinct_names_get_distinct_ids(tmp_path: Path):
    s = EntityStore(tmp_path / "entities.jsonl")
    a = s.find_or_create("alice")
    b = s.find_or_create("bob")
    assert a.entity_id != b.entity_id
    assert len(s) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tier 5 — PreferenceStore
# ─────────────────────────────────────────────────────────────────────────────
def test_preference_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "prefs.json"
    s1 = PreferenceStore(path)
    s1.update_cinematic("kevin", "shot_movement", "static")
    s1.set_strictness("kevin", 1.3)
    s1.endorse_lesson("kevin", "L_abc")
    # Reload.
    s2 = PreferenceStore(path)
    pref = s2.get("kevin")
    assert pref.cinematic_priors["shot_movement"] == "static"
    assert pref.physics_strictness == 1.3
    assert "L_abc" in pref.endorsed_lesson_ids


def test_preference_default_user_is_lazy(tmp_path: Path):
    s = PreferenceStore(tmp_path / "prefs.json")
    assert "default" not in s
    p = s.get("default")
    assert p.user_id == "default"
    assert p.physics_strictness == 1.0
    assert "default" in s                                  # created on first get


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — EpisodicStore (append + similar-task retrieval)
# ─────────────────────────────────────────────────────────────────────────────
def test_episodic_store_records_and_retrieves(tmp_path: Path):
    s = EpisodicStore(tmp_path / "ep.jsonl")
    s.append("t1", "a ball is thrown", trajectory_path="/tmp/t1.jsonl",
             report={"n_shots": 1, "shots": [
                 {"final_metrics": {"weighted_total": 0.9},
                  "revisions_used": 2, "escalations": 0, "converged": True}
             ]})
    s.append("t2", "a cat eats sushi", trajectory_path="/tmp/t2.jsonl",
             report={"n_shots": 1, "shots": [
                 {"final_metrics": {"weighted_total": 0.8},
                  "revisions_used": 1, "escalations": 0, "converged": True}
             ]})
    hits = s.similar_tasks("a ball bounces off a wall", top_k=1, threshold=0.05)
    assert len(hits) == 1
    assert hits[0].task_id == "t1"


def test_episodic_store_down_weights_escalated_runs(tmp_path: Path):
    """A diverged run (escalations > 0) should rank below an equally-similar
    converged-cleanly run — the precedents we surface are the GOOD ones."""
    s = EpisodicStore(tmp_path / "ep.jsonl")
    good = {"n_shots": 1, "shots": [
        {"final_metrics": {"weighted_total": 0.9}, "revisions_used": 1,
         "escalations": 0, "converged": True}
    ]}
    bad = {"n_shots": 1, "shots": [
        {"final_metrics": {"weighted_total": 0.5}, "revisions_used": 3,
         "escalations": 2, "converged": False}
    ]}
    s.append("ok", "a ball is thrown and bounces", report=good)
    s.append("bad", "a ball is thrown and bounces", report=bad)
    hits = s.similar_tasks("a ball is thrown and bounces", top_k=2)
    assert hits[0].task_id == "ok"                         # good run ranks first


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — A-MEM lesson evolution
# ─────────────────────────────────────────────────────────────────────────────
def test_lesson_add_creates_bidirectional_link_with_related(tmp_path: Path):
    """Adding a related lesson must auto-link it to existing ones (A-MEM
    evolution heuristic) — bidirectional, both directions persisted."""
    lib = LessonLibrary(tmp_path / "lessons.jsonl")
    a = lib.add("a ball is thrown and bounces off a wall",
                "enforce gravity arc + bounce response",
                PhysFailureMode.GRAVITY_INERTIA)
    b = lib.add("a ball drops and bounces on the ground",
                "enforce gravity arc + collision bounce constraint",
                PhysFailureMode.GRAVITY_INERTIA)
    # Bidirectional link must exist.
    assert b.lesson_id in a.linked_lesson_ids
    assert a.lesson_id in b.linked_lesson_ids


def test_lesson_add_idempotent_reconfirms_confidence(tmp_path: Path):
    lib = LessonLibrary(tmp_path / "lessons.jsonl")
    a = lib.add("water pours from a cup", "fluid continuity",
                PhysFailureMode.FLUID)
    confidence_before = a.confidence
    a2 = lib.add("water pours from a cup", "fluid continuity",
                 PhysFailureMode.FLUID)
    assert a2.lesson_id == a.lesson_id
    assert a2.confidence >= confidence_before


# ─────────────────────────────────────────────────────────────────────────────
# MultiLayerMemory façade — associative query lights up multiple tiers
# ─────────────────────────────────────────────────────────────────────────────
def test_mlm_associative_query_returns_hits_across_tiers(tmp_path: Path):
    mlm = MultiLayerMemory.open(base_dir=tmp_path, user_id="u")
    mlm.lessons.add("a ball is thrown and bounces", "enforce gravity arc",
                    PhysFailureMode.GRAVITY_INERTIA)
    # Skill: need to build a sketch.
    from maestro.physics.sketch import build_physics_sketch
    from maestro.types import CinematographyTags, ShotSpec
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    sketch = build_physics_sketch(spec, tmp_path, fps=8)
    mlm.skills.distill("projectile_v1", spec.prompt, sketch,
                       CinematographyTags(), {}, weighted_total=0.9)
    mlm.episodes.append("t1", "a ball thrown and bounces",
                        report={"n_shots": 1, "shots": [
                            {"final_metrics": {"weighted_total": 0.9},
                             "revisions_used": 1, "escalations": 0,
                             "converged": True}]})
    hits = mlm.query("a ball thrown", expected_modes=list(sketch.expected_modes))
    tiers = {h.tier for h in hits}
    # At least two distinct tiers should light up.
    assert len(tiers & {"semantic", "procedural", "episodic"}) >= 2
