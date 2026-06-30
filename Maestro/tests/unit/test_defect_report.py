"""DefectReport — review → localized, worst-first defects. CPU-only, no torch."""
from pathlib import Path

from maestro.agents.defect_report import (
    Defect,
    DefectReport,
    build_defect_report,
)
from maestro.types import (
    CandidateClip,
    Checklist,
    ChecklistItem,
    PhysFailureMode,
    PhysicsVerdict,
    ShotSpec,
)


def _spec(prompt="a glass falls off a table"):
    return ShotSpec(shot_idx=0, duration=1.0, prompt=prompt)


def _clip(verdicts=None, items=None):
    c = CandidateClip(shot_idx=0, video_path=Path("x.mp4"))
    c.physics_verdicts = verdicts or []
    c.checklist = Checklist(items=items or [])
    return c


def test_physics_verdict_becomes_localized_defect():
    v = PhysicsVerdict(
        mode=PhysFailureMode.GRAVITY_INERTIA, frame_range=(2, 5),
        severity=0.8, suggested_intervention="fix arc",
        source="law_verifier", entity="glass",
    )
    rep = build_defect_report(_clip(verdicts=[v]), _spec(), fps=8)
    assert len(rep.defects) == 1
    d = rep.defects[0]
    assert d.kind == "physics"
    assert d.entity == "glass"
    assert d.frame_range == (2, 5)
    assert d.fix_modality == "motion"      # gravity_inertia → motion


def test_object_permanence_is_presence_modality():
    v = PhysicsVerdict(
        mode=PhysFailureMode.OBJECT_PERMANENCE, frame_range=(0, 8),
        severity=0.6, suggested_intervention="persist", source="vlm",
        entity="ball",
    )
    rep = build_defect_report(_clip(verdicts=[v]), _spec(), fps=8)
    assert rep.defects[0].fix_modality == "presence"


def test_failed_semantic_item_becomes_content_defect_whole_clip():
    item = ChecklistItem(
        question="Is the glass visible in the shot?", kind="semantic",
        passed=False, fix_instruction="show the glass",
    )
    rep = build_defect_report(
        _clip(items=[item]), _spec("a glass falls"), fps=8
    )
    assert len(rep.defects) == 1
    d = rep.defects[0]
    assert d.kind == "semantic"
    assert d.fix_modality == "content"
    assert d.frame_range == (0, 8)         # whole clip (duration*fps)


def test_non_semantic_checklist_items_are_ignored():
    items = [
        ChecklistItem(question="rhythm ok?", kind="rhythm", passed=False),
        ChecklistItem(question="consistent?", kind="consistency", passed=False),
    ]
    rep = build_defect_report(_clip(items=items), _spec(), fps=8)
    assert rep.defects == []


def test_worst_first_ordering_and_worst():
    v_lo = PhysicsVerdict(PhysFailureMode.COLLISION, (1, 2), 0.3,
                          "x", "law_verifier", "a")
    v_hi = PhysicsVerdict(PhysFailureMode.CONSERVATION, (3, 4), 0.9,
                          "y", "law_verifier", "b")
    rep = build_defect_report(_clip(verdicts=[v_lo, v_hi]), _spec(), fps=8)
    order = [d.severity for d in rep.defects]
    assert order == sorted(order, reverse=True)
    assert rep.worst().severity == 0.9
    assert rep.worst().entity == "b" or rep.worst().fix_modality == "motion"


def test_to_brain_json_is_compact_dicts():
    v = PhysicsVerdict(PhysFailureMode.GRAVITY_INERTIA, (2, 5), 0.8,
                       "i", "law_verifier", "glass")
    js = build_defect_report(_clip(verdicts=[v]), _spec()).to_brain_json()
    assert js == [{
        "kind": "physics", "entity": "glass", "frame_range": [2, 5],
        "severity": 0.8, "fix_modality": "motion",
        "note": js[0]["note"],
    }]
    assert "gravity_inertia" in js[0]["note"]


def test_empty_report_is_falsy():
    assert not DefectReport()
    assert not build_defect_report(_clip(), _spec())
