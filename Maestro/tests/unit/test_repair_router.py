"""RepairRouter — the verdict→action routing of the Review→Execute bridge.

All pure logic: no I/O, no network, no torch. The router maps the WORST review
verdict to a tool-backed action, gated by backend capabilities + uploaded assets,
and ALWAYS degrades to "regenerate_hint" when nothing richer applies (so the mock
pipeline keeps its exact current behaviour)."""
from __future__ import annotations

from pathlib import Path

from maestro.agents.repair_router import RepairRouter
from maestro.types import (
    CandidateClip,
    Checklist,
    ChecklistItem,
    PhysFailureMode,
    PhysicsVerdict,
    ShotSpec,
)

MOCK_CAPS = {"t2v", "i2v"}
EDIT_CAPS = {"t2v", "i2v", "flf2v", "edit"}
EXTEND_CAPS = {"t2v", "i2v", "flf2v", "edit", "extend"}


def _clip(verdicts=None, items=None) -> CandidateClip:
    c = CandidateClip(shot_idx=0, video_path=Path("x.mp4"))
    c.physics_verdicts = verdicts or []
    c.checklist = Checklist(items=items or [])
    return c


def _verdict(mode, severity=0.8):
    return PhysicsVerdict(
        mode=mode, frame_range=(0, 4), severity=severity,
        suggested_intervention="fix the motion",
    )


def _spec(prompt="a ball falls") -> ShotSpec:
    return ShotSpec(shot_idx=0, duration=2.0, prompt=prompt)


def test_physics_motion_verdict_with_edit_cap_routes_to_edit_clip():
    clip = _clip(verdicts=[_verdict(PhysFailureMode.GRAVITY_INERTIA)])
    a = RepairRouter().choose(clip, _spec(), capabilities=EDIT_CAPS, has_source_shots=False)
    assert a.action == "edit_clip"
    assert a.hint  # verdict-derived


def test_physics_motion_verdict_without_edit_cap_falls_back():
    """Same verdict but a mock backend (no 'edit') → guaranteed regenerate_hint."""
    clip = _clip(verdicts=[_verdict(PhysFailureMode.COLLISION)])
    a = RepairRouter().choose(clip, _spec(), capabilities=MOCK_CAPS, has_source_shots=False)
    assert a.action == "regenerate_hint"


def test_semantic_miss_with_source_shots_routes_to_retrieve_replace():
    item = ChecklistItem(question="is the red car present?", kind="semantic",
                         passed=False, fix_instruction="add the red car")
    clip = _clip(items=[item])
    a = RepairRouter().choose(clip, _spec(), capabilities=MOCK_CAPS, has_source_shots=True)
    assert a.action == "retrieve_replace"
    assert a.hint == "add the red car"


def test_semantic_miss_without_source_shots_falls_back():
    item = ChecklistItem(question="is the red car present?", kind="semantic",
                         passed=False, fix_instruction="add the red car")
    clip = _clip(items=[item])
    a = RepairRouter().choose(clip, _spec(), capabilities=MOCK_CAPS, has_source_shots=False)
    assert a.action == "regenerate_hint"


def test_object_permanence_with_extend_cap_routes_to_extend_clip():
    clip = _clip(verdicts=[_verdict(PhysFailureMode.OBJECT_PERMANENCE)])
    a = RepairRouter().choose(clip, _spec(), capabilities=EXTEND_CAPS, has_source_shots=False)
    assert a.action == "extend_clip"


def test_incomplete_cue_with_extend_cap_routes_to_extend_clip():
    item = ChecklistItem(question="is the action complete?", kind="semantic",
                         passed=False, fix_instruction="the shot is too short, continue it")
    # No source shots → semantic-miss branch can't fire → extend cue wins.
    clip = _clip(items=[item])
    a = RepairRouter().choose(clip, _spec(), capabilities=EXTEND_CAPS, has_source_shots=False)
    assert a.action == "extend_clip"


def test_clean_clip_routes_to_regenerate_hint():
    a = RepairRouter().choose(_clip(), _spec(), capabilities=EXTEND_CAPS, has_source_shots=True)
    assert a.action == "regenerate_hint"


def test_edit_preferred_over_extend_for_motion_verdict():
    """Motion verdict (a) is checked before the extend cue (c)."""
    clip = _clip(verdicts=[_verdict(PhysFailureMode.PENETRATION)])
    a = RepairRouter().choose(clip, _spec("a ball goes through a wall"),
                              capabilities=EXTEND_CAPS, has_source_shots=False)
    assert a.action == "edit_clip"
