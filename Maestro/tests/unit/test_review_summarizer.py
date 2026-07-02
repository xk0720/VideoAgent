"""ReviewSummarizerAgent — heterogeneous reviews → ONE prioritized brief.

CPU-only, no network. The structured brief must be DETERMINISTIC and every
issue traceable to an actual reviewer output (signal honesty); the LLM only
rephrases prose (mock LLMs are ignored)."""
from pathlib import Path

from maestro.agents.defect_report import build_defect_report
from maestro.agents.review_summarizer import ReviewSummarizerAgent
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


def _measured(sev=0.8, fr=(2, 5), entity="glass", mode=PhysFailureMode.GRAVITY_INERTIA):
    return PhysicsVerdict(mode=mode, frame_range=fr, severity=sev,
                          suggested_intervention="fix the arc",
                          source="law_verifier", entity=entity)


def _opinion(sev=0.5, fr=(2, 5), entity="glass", mode=PhysFailureMode.GRAVITY_INERTIA):
    return PhysicsVerdict(mode=mode, frame_range=fr, severity=sev,
                          suggested_intervention="",
                          source="vlm", entity=entity)


def _brief(clip, spec, history=None, prev=None, llm=None):
    report = build_defect_report(clip, spec, fps=8)
    return ReviewSummarizerAgent(llm=llm).summarize(
        clip, spec, report, history=history, prev_issues=prev)


def test_empty_review_yields_empty_brief():
    b = _brief(_clip(), _spec())
    assert b["issues"] == []
    assert b["headline"] == ""
    assert "passed" in b["brief_nl"]


def test_cross_type_agreement_merges_into_one_issue_with_boost():
    """Measured + opinion on the SAME entity/span → ONE issue, provenance from
    both, cross_type_confirmed agreement, confidence boosted."""
    clip = _clip(verdicts=[_measured(0.8), _opinion(0.6)])
    b = _brief(clip, _spec())
    assert len(b["issues"]) == 1
    iss = b["issues"][0]
    types = {e["type"] for e in iss["evidence"]}
    assert types == {"measured", "opinion"}
    assert iss["agreement"] == "cross_type_confirmed"
    assert iss["confidence"] == 0.95
    assert iss["entity"] == "glass"


def test_measured_backed_ranks_above_opinion_only():
    """A measured issue outranks an opinion-only issue even at LOWER severity
    (measurement outranks opinion)."""
    clip = _clip(
        verdicts=[_measured(0.55, fr=(2, 5), entity="glass"),
                  _opinion(0.9, fr=(10, 12), entity="table",
                           mode=PhysFailureMode.OBJECT_PERMANENCE)],
    )
    b = _brief(clip, _spec())
    assert len(b["issues"]) == 2
    assert b["issues"][0]["entity"] == "glass"           # measured first
    assert any(e["type"] == "measured" for e in b["issues"][0]["evidence"])
    assert b["issues"][0]["id"] == "I-1"


def test_fix_classes_are_hints_never_tool_calls():
    """The summarizer suggests fix CLASSES only — no 'tool'/'args' keys anywhere
    in an issue (tool choice belongs to the brain)."""
    clip = _clip(verdicts=[_measured()])
    b = _brief(clip, _spec())
    iss = b["issues"][0]
    assert iss["fix_classes"] == ["localized_regen", "edit_in_place"]
    assert "tool" not in iss and "args" not in iss


def test_every_issue_traces_to_a_reviewer_output():
    """Signal honesty: issue count is bounded by real reviewer outputs; every
    evidence entry names its reviewer."""
    items = [ChecklistItem(question="does the glass shatter?", kind="semantic",
                           passed=False)]
    clip = _clip(verdicts=[_measured()], items=items)
    b = _brief(clip, _spec())
    assert len(b["issues"]) == 2                          # 1 verdict + 1 semantic
    for iss in b["issues"]:
        assert iss["evidence"], iss
        assert all(e.get("reviewer") for e in iss["evidence"])


def test_do_not_repeat_ledger_from_rejected_history():
    clip = _clip(verdicts=[_measured()])
    history = [
        ({"tool": "edit_clip", "args": {"prompt": "x"}}, "rejected", 0.41),
        ({"tool": "regenerate_segment", "args": {}}, "accepted", 0.55),
    ]
    b = _brief(clip, _spec(), history=history)
    assert len(b["do_not_repeat"]) == 1
    assert b["do_not_repeat"][0]["tool"] == "edit_clip"
    assert "do NOT repeat" in b["brief_nl"] or "Do NOT repeat" in b["brief_nl"]


def test_progress_fixed_new_unchanged_across_turns():
    spec = _spec()
    clip1 = _clip(verdicts=[_measured(0.8, fr=(2, 5), entity="glass"),
                            _opinion(0.6, fr=(10, 12), entity="table",
                                     mode=PhysFailureMode.OBJECT_PERMANENCE)])
    b1 = _brief(clip1, spec)
    assert all(i["status"] == "initial" for i in b1["issues"])

    # turn 2: the glass issue is GONE; the table issue persists.
    clip2 = _clip(verdicts=[_opinion(0.6, fr=(10, 12), entity="table",
                                     mode=PhysFailureMode.OBJECT_PERMANENCE)])
    b2 = _brief(clip2, spec, prev=b1["issues"])
    assert len(b2["progress"]["fixed"]) == 1
    assert b2["issues"][0]["status"] == "unchanged"


def test_regressed_issue_ranks_first():
    spec = _spec()
    clip1 = _clip(verdicts=[_measured(0.3, fr=(2, 5), entity="glass"),
                            _measured(0.9, fr=(10, 12), entity="table",
                                      mode=PhysFailureMode.COLLISION)])
    b1 = _brief(clip1, spec)
    # turn 2: glass got WORSE (0.3 → 0.7), table improved slightly (0.9 → 0.85)
    clip2 = _clip(verdicts=[_measured(0.7, fr=(2, 5), entity="glass"),
                            _measured(0.85, fr=(10, 12), entity="table",
                                      mode=PhysFailureMode.COLLISION)])
    b2 = _brief(clip2, spec, prev=b1["issues"])
    assert b2["issues"][0]["entity"] == "glass"
    assert b2["issues"][0]["status"] == "regressed"
    assert b2["progress"]["regressed"] == [b2["issues"][0]["key"]]


def test_mock_llm_is_never_consulted_for_prose():
    class MockLLMClient:
        def complete(self, prompt):
            raise AssertionError("mock LLM must not be called")

    clip = _clip(verdicts=[_measured()])
    b = _brief(clip, _spec(), llm=MockLLMClient())
    assert b["brief_nl_source"] == "template"


def test_real_llm_polishes_prose_but_structure_stays_deterministic():
    class StubLLM:
        def complete(self, prompt):
            return "The glass violates gravity in frames 2-5; repair that span."

    clip = _clip(verdicts=[_measured()])
    b_plain = _brief(clip, _spec())
    b_llm = _brief(clip, _spec(), llm=StubLLM())
    assert b_llm["brief_nl_source"] == "llm"
    assert "frames 2-5" in b_llm["brief_nl"]
    # the STRUCTURED brief is identical with or without the LLM
    assert b_llm["issues"] == b_plain["issues"]
    assert b_llm["headline"] == b_plain["headline"]


def test_llm_failure_falls_back_to_template():
    class BrokenLLM:
        def complete(self, prompt):
            raise RuntimeError("api down")

    clip = _clip(verdicts=[_measured()])
    b = _brief(clip, _spec(), llm=BrokenLLM())
    assert b["brief_nl_source"] == "template"
    assert b["brief_nl"]
