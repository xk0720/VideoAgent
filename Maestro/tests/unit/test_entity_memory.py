"""Tests for v0.4 dual-register entity memory + verification-gated writes.

Covers survey_memory_2026_06.md Angles 1+2: immutable identity register ⊕
mutable state register changed only through typed, gate-verified transitions
in an append-only log (EntityMem 2605.15199 can't evolve; VideoMemory
2601.03655 / StoryMem 2512.19539 drift; we factorize + audit).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from maestro.memory.entity_store import (
    EntityStore,
    make_identity,
    propose_transitions_from_spec,
)
from maestro.memory.write_gate import VerificationWriteGate
from maestro.types import (
    CandidateClip,
    ChecklistItem,
    EntityState,
    ShotSpec,
    StateTransition,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clip(tmp_path: Path, shot_idx: int, body: str, accepted: bool = True,
          consistency_failed: bool = False) -> CandidateClip:
    """Build a clip whose 'render' is the mock prompt echo in the body file."""
    vp = tmp_path / f"shot{shot_idx:03d}.mp4"
    vp.write_text(f"MOCK VIDEO\nprompt={body}\n", encoding="utf-8")
    clip = CandidateClip(shot_idx=shot_idx, video_path=vp, accepted=accepted)
    if consistency_failed:
        clip.checklist.items.append(ChecklistItem(
            question="identity consistent?", kind="consistency", passed=False,
        ))
    return clip


def _wet_transition(entity_id: str, shot_idx: int = 1) -> StateTransition:
    return StateTransition(
        entity_id=entity_id, shot_idx=shot_idx, field="condition",
        old="", new="wet", cause="director: hero falls in the river",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. identity register — idempotent + immutable
# ─────────────────────────────────────────────────────────────────────────────
def test_identity_register_idempotent_and_immutable(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    a = store.register(make_identity("hero", ["/refs/hero.png"], "the hero"))
    b = store.register(make_identity("hero", ["/refs/OTHER.png"], "imposter"))
    # Idempotent on entity_id: the ORIGINAL register wins, untouched.
    assert b is a
    assert len(store.identities) == 1
    assert a.reference_paths == ["/refs/hero.png"]
    # Frozen dataclass: no attribute mutation possible.
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.name = "villain"
    # The store exposes no identity-mutation API.
    mutators = [m for m in dir(store)
                if "identity" in m.lower() and any(
                    v in m.lower() for v in ("update", "set", "edit", "mutate"))]
    assert mutators == []


# ─────────────────────────────────────────────────────────────────────────────
# 2-5. verification-gated writes
# ─────────────────────────────────────────────────────────────────────────────
def test_propose_then_commit_on_accepted_clip_with_evidence(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    store.propose(_wet_transition(ident.entity_id))
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet after the river")
    clip = _clip(tmp_path, 1, spec.prompt, accepted=True)
    out = store.commit_gated(clip, spec, VerificationWriteGate())
    assert out == {"committed": 1, "rejected": 0}
    view = store.current_state(ident.entity_id)
    assert view["attributes"] == {"condition": "wet"}
    assert view["state_version"] == 1
    (t,) = store.history(ident.entity_id)
    assert t.status == "committed"
    assert "wet" in t.evidence            # evidence string records what was seen


def test_reject_when_clip_not_accepted(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    store.propose(_wet_transition(ident.entity_id))
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    clip = _clip(tmp_path, 1, spec.prompt, accepted=False)
    out = store.commit_gated(clip, spec, VerificationWriteGate())
    assert out == {"committed": 0, "rejected": 1}
    # State register untouched — unrendered proposals never write memory.
    assert store.current_state(ident.entity_id)["attributes"] == {}
    (t,) = store.history(ident.entity_id)
    assert t.status == "rejected"
    assert "not accepted" in t.evidence


def test_reject_when_evidence_absent(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    store.propose(_wet_transition(ident.entity_id))
    # Accepted clip, but the render shows the hero in sunshine — no "wet".
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero walks in sunshine")
    clip = _clip(tmp_path, 1, spec.prompt, accepted=True)
    out = store.commit_gated(clip, spec, VerificationWriteGate())
    assert out == {"committed": 0, "rejected": 1}
    assert store.current_state(ident.entity_id)["attributes"] == {}
    (t,) = store.history(ident.entity_id)
    assert t.status == "rejected"         # discrepancy logged, not silent
    assert "absent" in t.evidence


def test_reject_when_consistency_checklist_failed(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    store.propose(_wet_transition(ident.entity_id))
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    clip = _clip(tmp_path, 1, spec.prompt, accepted=True, consistency_failed=True)
    out = store.commit_gated(clip, spec, VerificationWriteGate())
    assert out == {"committed": 0, "rejected": 1}
    (t,) = store.history(ident.entity_id)
    assert "consistency" in t.evidence


# ─────────────────────────────────────────────────────────────────────────────
# 6. correction entry — explicit, auditable
# ─────────────────────────────────────────────────────────────────────────────
def test_correction_entry_path(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    store.propose(_wet_transition(ident.entity_id, shot_idx=1))
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    store.commit_gated(_clip(tmp_path, 1, spec.prompt), spec,
                       VerificationWriteGate())
    # Shot 2's render contradicts memory: hero rendered dry. Memory follows
    # the pixels — but only through an explicit correction entry.
    t = store.record_correction(
        ident.entity_id, shot_idx=2, field="condition", new="dry",
        cause="render contradicted committed state",
        evidence="gate saw 'dry' in shot 2 prompt echo; 'wet' absent",
    )
    assert t.status == "correction"
    assert t.old == "wet"
    view = store.current_state(ident.entity_id)
    assert view["attributes"]["condition"] == "dry"
    assert view["state_version"] == 2
    statuses = [h.status for h in store.history(ident.entity_id)]
    assert statuses == ["committed", "correction"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. history ordering + audit completeness
# ─────────────────────────────────────────────────────────────────────────────
def test_history_ordering_and_audit_completeness(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    gate = VerificationWriteGate()
    # shot 1: wet (commits) + broken (rejected: no evidence in render)
    store.propose(_wet_transition(ident.entity_id, shot_idx=1))
    store.propose(StateTransition(
        entity_id=ident.entity_id, shot_idx=1, field="condition",
        old="", new="broken", cause="spurious proposal"))
    spec1 = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    store.commit_gated(_clip(tmp_path, 1, spec1.prompt), spec1, gate)
    # shot 3: emotion change commits
    store.propose(StateTransition(
        entity_id=ident.entity_id, shot_idx=3, field="emotion",
        old="", new="angry", cause="prompt cue 'angry'"))
    spec3 = ShotSpec(shot_idx=3, duration=2.0, prompt="the hero is angry")
    store.commit_gated(_clip(tmp_path, 3, spec3.prompt), spec3, gate)

    hist = store.history(ident.entity_id)
    assert [(t.shot_idx, t.new, t.status) for t in hist] == [
        (1, "wet", "committed"), (1, "broken", "rejected"),
        (3, "angry", "committed"),
    ]
    # Audit completeness: state version == # applied transitions, and
    # replaying ONLY committed/correction entries reproduces the state.
    view = store.current_state(ident.entity_id)
    applied = [t for t in hist if t.status in ("committed", "correction")]
    assert view["state_version"] == len(applied)
    replayed = EntityState()
    for t in applied:
        replayed.attributes[t.field] = t.new
        replayed.version += 1
    assert replayed.attributes == view["attributes"]
    assert replayed.version == view["state_version"]
    # Every transition (rejected included) carries gate evidence.
    assert all(t.evidence for t in hist)


# ─────────────────────────────────────────────────────────────────────────────
# 8. re-entry after a gap (EntityBench-style)
# ─────────────────────────────────────────────────────────────────────────────
def test_reentry_context_after_gap(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    # Shot 0: identity registered with verified reference paths.
    ident = store.register(make_identity(
        "hero", ["/refs/hero_face.png"], "the protagonist"))
    # Shot 1: state transitions commit.
    store.propose(_wet_transition(ident.entity_id, shot_idx=1))
    spec1 = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    store.commit_gated(_clip(tmp_path, 1, spec1.prompt), spec1,
                       VerificationWriteGate())
    # Shot 5: long-gap re-entry — payload must carry identity refs AND the
    # last committed state, ready to condition the generator.
    ctx = store.reentry_context(ident.entity_id)
    assert ctx["reference_paths"] == ["/refs/hero_face.png"]
    assert ctx["attributes"] == {"condition": "wet"}
    assert ctx["last_state_shot_idx"] == 1
    assert "hero" in ctx["conditioning_text"]
    assert "wet" in ctx["conditioning_text"]
    assert store.reentry_context("E_unknown") is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. JSONL persistence round-trip (registers + log)
# ─────────────────────────────────────────────────────────────────────────────
def test_jsonl_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "entities.jsonl"
    s1 = EntityStore(path)
    ident = s1.register(make_identity("hero", ["/refs/hero.png"]))
    s1.propose(_wet_transition(ident.entity_id, shot_idx=1))
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero is wet")
    s1.commit_gated(_clip(tmp_path, 1, spec.prompt), spec,
                    VerificationWriteGate())
    s1.propose(StateTransition(           # left proposed on disk
        entity_id=ident.entity_id, shot_idx=2, field="emotion",
        old="", new="happy", cause="pending"))

    s2 = EntityStore(path)                # fresh instance, same JSONL
    assert set(s2.identities) == {ident.entity_id}
    assert s2.identities[ident.entity_id].reference_paths == ["/refs/hero.png"]
    assert [(t.status, t.new) for t in s2.history(ident.entity_id)] == [
        ("committed", "wet"), ("proposed", "happy"),
    ]
    # State was REPLAYED from the log — proposed entries did not apply.
    view = s2.current_state(ident.entity_id)
    assert view["attributes"] == {"condition": "wet"}
    assert view["state_version"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. legacy v0.3 records still load alongside the new registers
# ─────────────────────────────────────────────────────────────────────────────
def test_legacy_v03_entity_records_load(tmp_path: Path):
    path = tmp_path / "entities.jsonl"
    # Exact v0.3 persisted shape (no "kind" key).
    legacy = {
        "entity_id": "Eabcdefabcdef", "canonical_name": "hero",
        "source_paths": ["/d1/hero.png"], "style_descriptors": {},
        "appearance_log": [{"task_id": "t1", "source_path": "/d1/hero.png",
                            "bbox": [], "ts": 1.0}],
        "physics_profile": {}, "first_seen_ts": 1.0, "last_seen_ts": 1.0,
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    store = EntityStore(path)
    assert len(store) == 1                       # legacy register loaded
    # Legacy cross-run dedup still works.
    ent = store.find_or_create("hero", source_path="/d2/hero.png", task_id="t2")
    assert ent.entity_id == "Eabcdefabcdef"
    # New registers coexist and survive a rewrite + reload.
    store.register(make_identity("hero"))
    reloaded = EntityStore(path)
    assert len(reloaded) == 1
    assert len(reloaded.identities) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 11. deterministic transition authoring (mock Director stand-in)
# ─────────────────────────────────────────────────────────────────────────────
def test_propose_transitions_from_spec_cues(tmp_path: Path):
    store = EntityStore(tmp_path / "entities.jsonl")
    hero = store.register(make_identity("hero"))
    store.register(make_identity("castle"))      # registered but not in prompt
    spec = ShotSpec(shot_idx=2, duration=2.0,
                    prompt="the hero is wet and angry, holding a sword")
    proposals = propose_transitions_from_spec(spec, store)
    by_field = {t.field: t for t in proposals}
    assert set(by_field) == {"condition", "emotion", "holding"}
    assert by_field["condition"].new == "wet"
    assert by_field["emotion"].new == "angry"
    assert by_field["holding"].new == "sword"
    assert all(t.entity_id == hero.entity_id for t in proposals)
    assert all(t.status == "proposed" for t in proposals)
    # Unregistered names and absent entities never get proposals.
    assert not any(t.entity_id != hero.entity_id for t in proposals)


# ─────────────────────────────────────────────────────────────────────────────
# 12. real-path injection contract: an MLLM client takes over the look
# ─────────────────────────────────────────────────────────────────────────────
def test_gate_delegates_to_injected_mllm(tmp_path: Path):
    class FrameLookMLLM:
        """Stand-in for the real backend: looks at decoded frames."""
        def confirm_state_change(self, transition, clip, spec):
            return True, "MLLM: change visible in decoded frames"

    store = EntityStore(tmp_path / "entities.jsonl")
    ident = store.register(make_identity("hero"))
    store.propose(_wet_transition(ident.entity_id))
    # Prompt deliberately lacks 'wet' — mock echo would reject, MLLM confirms.
    spec = ShotSpec(shot_idx=1, duration=2.0, prompt="the hero by the river")
    clip = _clip(tmp_path, 1, spec.prompt, accepted=True)
    gate = VerificationWriteGate(mllm=FrameLookMLLM())
    out = store.commit_gated(clip, spec, gate)
    assert out == {"committed": 1, "rejected": 0}
    (t,) = store.history(ident.entity_id)
    assert t.evidence.startswith("MLLM:")
    # But the accept/consistency preconditions still apply even with an MLLM.
    store.propose(StateTransition(
        entity_id=ident.entity_id, shot_idx=4, field="emotion",
        old="", new="sad", cause="x"))
    spec4 = ShotSpec(shot_idx=4, duration=2.0, prompt="the hero is sad")
    out = store.commit_gated(_clip(tmp_path, 4, spec4.prompt, accepted=False),
                             spec4, gate)
    assert out == {"committed": 0, "rejected": 1}


# ─────────────────────────────────────────────────────────────────────────────
# 13. pipeline wiring — run_maestro registers, gates and reports
# ─────────────────────────────────────────────────────────────────────────────
def test_run_maestro_gates_entity_writes_and_reports(tmp_path: Path):
    from maestro.pipeline.run import run_maestro

    result = run_maestro(
        user_prompt="a dog falls into the river and gets wet; the wet dog shakes",
        output_path=tmp_path / "out.mp4",
        cache_dir=tmp_path / "cache",
        memory_dir=tmp_path / "memory",
    )
    report = result["report"]
    # The dual-register tier surfaces in the run report.
    assert report["entities"] >= 1
    assert report["transitions_committed"] >= 1
    mlm = result["mlm"]
    # The annotated entity ("dog", crude noun cue) got an identity register
    # and a gate-committed state, and its re-entry payload is ready.
    dog_id = [eid for eid, i in mlm.entities.identities.items()
              if i.name == "dog"]
    assert dog_id
    ctx = mlm.entities.reentry_context(dog_id[0])
    assert ctx["attributes"].get("condition") == "wet"
    assert all(t.evidence for t in mlm.entities.history(dog_id[0]))
    # And it all persisted: a fresh store on the same JSONL replays the state.
    reloaded = EntityStore(tmp_path / "memory" / "entities.jsonl")
    assert reloaded.current_state(dog_id[0])["attributes"]["condition"] == "wet"
