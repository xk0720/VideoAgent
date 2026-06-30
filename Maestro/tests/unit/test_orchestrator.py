"""OrchestratorAgent (the brain) + generate_shot_orchestrated.

CPU-only, NO network / NO torch. The brain is a tiny StubBrainLLM returning
canned STRICT JSON (the real MockLLMClient returns "ack:" not JSON, so it is
NOT usable here). Review uses the deterministic MockMLLMClient via the default
critics + mock video gen, so convergence is content-derived and fast.
"""
import json
from pathlib import Path

from maestro.agents.generator import GeneratorAgent
from maestro.agents.orchestrator import INVALID, OrchestratorAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.consistency import ConsistencyCritic
from maestro.critics.physics import PhysicsCritic
from maestro.critics.rhythm import RhythmCritic
from maestro.critics.semantic import SemanticCritic
from maestro.memory.lesson_library import LessonLibrary
from maestro.models.llm import BaseLLMClient
from maestro.models.video_gen import MockVideoGenClient
from maestro.physics.annotate import annotate_physics
from maestro.pipeline.generate_loop import generate_shot_orchestrated
from maestro.types import (
    AssetMemory,
    CandidateClip,
    Checklist,
    ChecklistItem,
    PhysFailureMode,
    PhysicsVerdict,
    Shot,
    ShotSpec,
)


# ── stubs ──────────────────────────────────────────────────────────────────
class StubBrainLLM(BaseLLMClient):
    """Returns canned JSON. `replies` is a list cycled per complete() call; a
    plain str is wrapped. Records every prompt it saw for history assertions."""

    def __init__(self, replies):
        self.replies = replies if isinstance(replies, list) else [replies]
        self.prompts: list[str] = []
        self._i = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        r = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return r


class _EditCapVideoGen(MockVideoGenClient):
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
        out.write_text(f"MOCK VIDEO\nprompt={prompt}\n", encoding="utf-8")
        return out


class _StubRetrieval:
    def __init__(self, memory):
        self.memory = memory
        self.queries: list[str] = []

    def retrieve_source_shots(self, query, top_k=3):
        self.queries.append(query)
        return list(self.memory.video_shots.keys())

    def retrieve_identity_refs(self, identity_refs):
        return None


def _board():
    return ReviewBoard([
        SemanticCritic(), PhysicsCritic(), ConsistencyCritic(), RhythmCritic(),
    ])


def _clip(verdicts=None, items=None) -> CandidateClip:
    c = CandidateClip(shot_idx=0, video_path=Path("x.mp4"))
    c.physics_verdicts = verdicts or []
    c.checklist = Checklist(items=items or [])
    c.metric_scores = {"weighted_total": 0.5}
    c.keyframes = [Path("kf0.txt"), Path("kf1.txt")]
    return c


def _spec(prompt="a ball falls and bounces") -> ShotSpec:
    s = ShotSpec(shot_idx=0, duration=1.0, prompt=prompt)
    s.physics_annotation = annotate_physics(s)
    return s


# ── available_actions gating ─────────────────────────────────────────────────
def test_available_actions_base_menu_no_caps_no_assets():
    gen = GeneratorAgent(video_gen=MockVideoGenClient())  # caps = {t2v,i2v}
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    names = {a["name"] for a in orch.available_actions(asset_memory=None)}
    # Whole-clip + the LOCALIZED-propagated tools (offered for any backend with
    # a video capability); no edit/extend/retrieve/flf2v without those caps/assets.
    assert names == {"regenerate", "keyframe_edit", "accept",
                     "regenerate_segment", "keyframe_edit_propagate"}
    assert "frame_to_frame" not in names   # flf2v not in {t2v,i2v}
    assert "edit_clip" not in names and "extend_clip" not in names


def test_available_actions_edit_appears_with_edit_cap():
    gen = GeneratorAgent(video_gen=_EditCapVideoGen())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    names = {a["name"] for a in orch.available_actions()}
    assert "edit_clip" in names
    assert "extend_clip" not in names  # _EditCapVideoGen has no 'extend'


def test_available_actions_extend_and_retrieve_gate():
    class _AllCaps(MockVideoGenClient):
        def capabilities(self):
            return {"t2v", "i2v", "edit", "extend"}

    mem = AssetMemory(video_shots={"s0": Shot("s0", "src.mp4", 0.0, 1.0)})
    gen = GeneratorAgent(video_gen=_AllCaps())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen,
                             retrieval=_StubRetrieval(mem))
    names = {a["name"] for a in orch.available_actions(asset_memory=mem)}
    assert {"edit_clip", "extend_clip", "retrieve_replace"} <= names
    # retrieve disappears when there are no source shots
    names_no = {a["name"] for a in orch.available_actions(asset_memory=AssetMemory())}
    assert "retrieve_replace" not in names_no


# ── decide(): valid JSON → validated; garbage → INVALID sentinel ─────────────
def test_decide_returns_validated_decision_for_known_tool():
    gen = GeneratorAgent(video_gen=_EditCapVideoGen())
    reply = json.dumps({"tool": "edit_clip",
                        "args": {"prompt": "fix the arc", "backend": "runway"},
                        "reason": "motion verdict"})
    orch = OrchestratorAgent(llm=StubBrainLLM(reply), generator=gen)
    menu = orch.available_actions()
    d = orch.decide(_clip(), _spec(), menu, history=[])
    assert d["tool"] == "edit_clip"
    assert d["args"]["prompt"] == "fix the arc"
    assert d["reason"] == "motion verdict"


def test_decide_garbage_reply_returns_invalid_sentinel():
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM("I think you should maybe edit it?"),
                             generator=gen)
    menu = orch.available_actions()
    d = orch.decide(_clip(), _spec(), menu, history=[])
    assert d == INVALID


def test_decide_out_of_menu_tool_returns_invalid():
    gen = GeneratorAgent(video_gen=MockVideoGenClient())  # no edit cap
    reply = json.dumps({"tool": "edit_clip", "args": {}, "reason": "x"})
    orch = OrchestratorAgent(llm=StubBrainLLM(reply), generator=gen)
    menu = orch.available_actions()  # edit_clip NOT in menu (no cap)
    assert orch.decide(_clip(), _spec(), menu, history=[]) == INVALID


# ── execute(): tool routing ──────────────────────────────────────────────────
def test_execute_edit_clip_routes_to_video_gen(tmp_path):
    gen = GeneratorAgent(video_gen=_EditCapVideoGen())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    best = gen.run(_spec(), tmp_path, revision=0, seed=0)
    decision = {"tool": "edit_clip",
                "args": {"prompt": "straighten the trajectory", "backend": "runway"},
                "reason": "r"}
    cand = orch.execute(decision, best, _spec(), tmp_path, r=1, board=_board())
    assert cand is not None
    assert gen.video_gen.edit_calls, "edit_clip must hit video_gen.edit_video"
    assert gen.video_gen.edit_calls[0]["backend"] == "runway"
    assert cand.metric_scores  # board.review ran


def test_execute_retrieve_replace_uses_stub_retrieval(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_text("MOCK VIDEO\nprompt=source\n", encoding="utf-8")
    mem = AssetMemory(video_shots={"s0": Shot("s0", str(src), 0.0, 1.0)})
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    retr = _StubRetrieval(mem)
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen, retrieval=retr)
    best = gen.run(_spec(), tmp_path, revision=0, seed=0)
    decision = {"tool": "retrieve_replace", "args": {"query": "a ball"}, "reason": "r"}
    cand = orch.execute(decision, best, _spec(), tmp_path, r=1, board=_board(),
                        asset_memory=mem)
    assert cand is not None
    assert Path(cand.video_path) == src
    assert retr.queries == ["a ball"]


def test_execute_accept_returns_none(tmp_path):
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    best = gen.run(_spec(), tmp_path, revision=0, seed=0)
    assert orch.execute({"tool": "accept", "args": {}}, best, _spec(),
                        tmp_path, r=1, board=_board()) is None


def test_execute_missing_capability_returns_none(tmp_path):
    """edit_clip decision but a backend without 'edit' → no-op (no crash)."""
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    best = gen.run(_spec(), tmp_path, revision=0, seed=0)
    decision = {"tool": "edit_clip", "args": {"prompt": "x"}, "reason": "r"}
    assert orch.execute(decision, best, _spec(), tmp_path, r=1, board=_board()) is None


# ── integration: the full brain loop (regenerate JSON) ───────────────────────
def test_orchestrated_loop_converges_and_records_trace(tmp_path):
    # Brain always asks for a regenerate with an anti-defect hint → the hint
    # lands in the clip body, applied_fixes grows, the mock review converges.
    # Turn 1: a multi-part physics hint (parts joined by " | ") resolves both
    # expected modes. Turn 2: the brain REACTS to the remaining semantic defect
    # the review still reports, folding its fix in. This is the agentic loop:
    # read review → call a tool → gate → read the new review → adapt.
    reply1 = json.dumps({"tool": "regenerate",
                         "args": {"hint": "one continuous passive ballistic arc "
                                  "under gravity | resolve the wall collision with "
                                  "momentum conserved"},
                         "reason": "worst verdict is gravity_inertia"})
    reply2 = json.dumps({"tool": "regenerate",
                         "args": {"hint": "one continuous passive ballistic arc "
                                  "under gravity | resolve the wall collision with "
                                  "momentum conserved | strengthen depiction of "
                                  "'bounces'"},
                         "reason": "remaining semantic miss: bounces not clear"})
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM([reply1, reply2]), generator=gen,
                             refiner=RefinerAgent())
    spec = _spec()
    res = generate_shot_orchestrated(
        spec, _board(), gen, RefinerAgent(), VerifierAgent(), tmp_path, orch,
        lesson_library=LessonLibrary(tmp_path / "l.jsonl"),
        n_candidates=2, max_turns=5,
    )
    # the brain's action trace is populated
    assert res.actions, "brain action trace must be recorded"
    assert all(a["tool"] == "regenerate" for a in res.actions if a["tool"] != "accept")
    # the verifier gated at least one decision (accepted or rejected recorded)
    assert any(a["outcome"] in ("accepted", "rejected") for a in res.actions)
    assert any(a["outcome"] == "accepted" for a in res.actions)
    # monotonic non-decreasing score history (gate never accepts a regression)
    h = res.score_history
    assert all(h[i] <= h[i + 1] + 1e-9 for i in range(len(h) - 1)), h
    assert res.clip.accepted
    assert res.converged  # content-derived convergence


def test_orchestrated_loop_garbage_falls_back_to_router(tmp_path):
    # Brain returns garbage every turn → loop must fall back to the deterministic
    # RepairRouter, still terminate, and never crash.
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM(["not json at all"]), generator=gen)
    spec = _spec()
    res = generate_shot_orchestrated(
        spec, _board(), gen, RefinerAgent(), VerifierAgent(), tmp_path, orch,
        n_candidates=2, max_turns=3,
    )
    assert res.actions
    assert all(a.get("via") == "repair_router_fallback"
               for a in res.actions if a["tool"] != "accept")
    # terminated within budget, shipped a clip
    assert res.clip.accepted
    h = res.score_history
    assert all(h[i] <= h[i + 1] + 1e-9 for i in range(len(h) - 1)), h


# ── v0.4 LOCALIZED / PROPAGATED repair: defect report + routing + no-accept ──
from maestro.agents.defect_report import build_defect_report  # noqa: E402


def test_decide_prompt_includes_localized_defects():
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM('{"tool":"accept","args":{}}'),
                             generator=gen)
    spec = _spec()
    v = PhysicsVerdict(PhysFailureMode.GRAVITY_INERTIA, (2, 5), 0.8,
                       "fix arc", "law_verifier", "ball")
    clip = _clip(verdicts=[v])
    report = build_defect_report(clip, spec)
    menu = orch.available_actions()
    orch.decide(clip, spec, menu, history=[], defect_report=report)
    prompt = orch.llm.prompts[-1]
    assert "localized_defects" in prompt
    assert '"fix_modality": "motion"' in prompt
    assert '"entity": "ball"' in prompt


def test_execute_regenerate_segment_routes_to_propagate_repair(tmp_path, monkeypatch):
    # Brain picks a LOCALIZED tool → execute builds a timeline and calls
    # propagate_repair (monkeypatched to a fake spliced clip).
    import maestro.pipeline.timeline as tl

    captured = {}

    class _FakeTimeline:
        degraded = False
        segments = [object()]

    def fake_from_clip(clip, cache_dir, n_segments=3):
        return _FakeTimeline()

    def fake_propagate(timeline, defect, **kw):
        captured["defect"] = defect
        captured["hint"] = kw.get("hint")
        out = Path(kw["cache_dir"]) / "spliced.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("MOCK VIDEO\nprompt=spliced\n", encoding="utf-8")
        return out

    monkeypatch.setattr(tl.ClipTimeline, "from_clip", classmethod(
        lambda cls, clip, cache_dir, n_segments=3: fake_from_clip(clip, cache_dir)))
    monkeypatch.setattr(tl, "propagate_repair", fake_propagate)

    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    best = gen.run(_spec(), tmp_path, revision=0, seed=0)
    decision = {"tool": "regenerate_segment",
                "args": {"frame_start": 2, "frame_end": 5, "hint": "fix the arc"},
                "reason": "localized motion defect"}
    cand = orch.execute(decision, best, _spec(), tmp_path, r=1, board=_board())
    assert cand is not None
    assert "spliced" in str(cand.video_path)
    assert captured["defect"].frame_range == (2, 5)
    assert captured["hint"] == "fix the arc"
    assert cand.metric_scores  # board.review ran


def test_execute_regenerate_segment_degrades_to_none_on_mock_clip(tmp_path):
    # Real path: mock clip has no decodable frames → timeline degraded →
    # propagate_repair returns None → execute returns None (no-op).
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    best = gen.run(_spec(), tmp_path, revision=0, seed=0)
    decision = {"tool": "regenerate_segment",
                "args": {"frame_start": 0, "frame_end": 3, "hint": "x"},
                "reason": "r"}
    assert orch.execute(decision, best, _spec(), tmp_path, r=1,
                        board=_board()) is None


def test_loop_does_not_accept_while_defects_remain(tmp_path):
    # Brain says "accept" on turn 1 while the review still has unresolved defects.
    # The loop must OVERRIDE the accept with a deterministic fallback action
    # instead of stopping, and only really stop once converged / out of turns.
    accept = json.dumps({"tool": "accept", "args": {},
                        "reason": "looks fine to me"})
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(llm=StubBrainLLM([accept]), generator=gen,
                             refiner=RefinerAgent())
    spec = _spec()
    res = generate_shot_orchestrated(
        spec, _board(), gen, RefinerAgent(), VerifierAgent(), tmp_path, orch,
        n_candidates=2, max_turns=4,
    )
    # An accept-while-defects turn was overridden (not a clean stop) at least once.
    assert any(a["tool"] == "accept_overridden" for a in res.actions), res.actions
    # The override took a real fallback repair, not a stop.
    assert any(a.get("via") == "repair_router_fallback"
               for a in res.actions if a["tool"] == "accept_overridden")
    # Terminates within budget, ships a clip, monotonic history.
    assert res.clip.accepted
    h = res.score_history
    assert all(h[i] <= h[i + 1] + 1e-9 for i in range(len(h) - 1)), h
    # Every recorded turn carries its localized defect snapshot.
    assert all("defects" in a for a in res.actions)
