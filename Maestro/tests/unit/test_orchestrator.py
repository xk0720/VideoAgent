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


# ── v0.4 widened palette: depth_edit / style_edit appear with caps ───────────
class _FullPaletteVideoGen(MockVideoGenClient):
    def __init__(self):
        super().__init__(name="mock-full-gen")
        self.depth_calls: list[dict] = []
        self.style_calls: list[dict] = []

    def capabilities(self):
        return {"t2v", "i2v", "edit", "extend", "depth", "style"}

    def depth_modify(self, prompt, video_path, out_path, seed=0):
        self.depth_calls.append({"prompt": prompt})
        out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"MOCK VIDEO\nprompt={prompt}\n", encoding="utf-8")
        return out

    def style_transfer(self, prompt, video_path, out_path, seed=0):
        self.style_calls.append({"prompt": prompt})
        out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"MOCK VIDEO\nprompt={prompt}\n", encoding="utf-8")
        return out


def test_available_actions_full_palette_includes_depth_and_style():
    gen = GeneratorAgent(video_gen=_FullPaletteVideoGen())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    names = {a["name"] for a in orch.available_actions()}
    assert {"depth_edit", "style_edit", "edit_clip", "extend_clip"} <= names
    # The menu text describes each by defect modality (mirrors UniVA plan.txt).
    menu = orch.available_actions()
    depth = next(a for a in menu if a["name"] == "depth_edit")
    assert "background" in depth["description"].lower()


def test_execute_depth_and_style_edit_route_to_backend(tmp_path):
    gen = GeneratorAgent(video_gen=_FullPaletteVideoGen())
    orch = OrchestratorAgent(llm=StubBrainLLM("{}"), generator=gen)
    best = gen.run(_spec(), tmp_path, revision=0, seed=0)
    d_cand = orch.execute({"tool": "depth_edit", "args": {"prompt": "beach bg"}},
                          best, _spec(), tmp_path, r=1, board=_board())
    s_cand = orch.execute({"tool": "style_edit", "args": {"prompt": "van gogh"}},
                          best, _spec(), tmp_path, r=2, board=_board())
    assert d_cand is not None and gen.video_gen.depth_calls
    assert s_cand is not None and gen.video_gen.style_calls


# ── v0.4 RETRIEVE-FIRST decide(): replay a learned repair workflow ───────────
class _StubSkillLibrary:
    """Stub: retrieve_repair returns a canned repair skill (or None). mark_used
    records the call so the loop's ledgering can be asserted."""

    def __init__(self, skill):
        self._skill = skill
        self.used: list[str] = []

    def retrieve_repair(self, defect_report):
        return self._skill

    def mark_used(self, skill_id):
        self.used.append(skill_id)


class _StubSkill:
    def __init__(self, skill_id, workflow):
        self.skill_id = skill_id
        self.name = "fix_gravity__repair"
        self.repair_workflow = workflow


def test_decide_replays_skill_step_when_repair_skill_hits():
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    skill = _StubSkill("R123", [
        {"tool": "regenerate", "args_template": {"hint": "one continuous arc"},
         "modality": "motion"},
    ])
    # The LLM would say "accept" — but a skill matches, so it is NEVER called.
    brain = StubBrainLLM('{"tool":"accept","args":{}}')
    orch = OrchestratorAgent(llm=brain, generator=gen,
                             skill_library=_StubSkillLibrary(skill))
    spec = _spec()
    v = PhysicsVerdict(PhysFailureMode.GRAVITY_INERTIA, (2, 5), 0.8,
                       "fix arc", "law_verifier", "ball")
    clip = _clip(verdicts=[v])
    report = build_defect_report(clip, spec)
    menu = orch.available_actions()
    d = orch.decide(clip, spec, menu, history=[], defect_report=report)
    assert d["via"] == "skill"
    assert d["tool"] == "regenerate"
    assert d["skill_id"] == "R123"
    assert d["args"]["hint"] == "one continuous arc"
    assert brain.prompts == []   # LLM never consulted — workflow replayed


def test_decide_falls_to_llm_when_no_skill_matches():
    gen = GeneratorAgent(video_gen=_EditCapVideoGen())
    reply = json.dumps({"tool": "edit_clip",
                        "args": {"prompt": "fix", "backend": "runway"},
                        "reason": "motion"})
    brain = StubBrainLLM(reply)
    # No skill library → must reason fresh.
    orch = OrchestratorAgent(llm=brain, generator=gen)
    report = build_defect_report(_clip(), _spec())
    d = orch.decide(_clip(), _spec(), orch.available_actions(), history=[],
                    defect_report=report)
    assert d["via"] == "llm"
    assert d["tool"] == "edit_clip"
    assert brain.prompts, "LLM consulted for fresh reasoning"


def test_decide_falls_to_llm_when_skill_steps_exhausted():
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    skill = _StubSkill("R9", [
        {"tool": "regenerate", "args_template": {"hint": "h"}, "modality": "motion"},
    ])
    reply = json.dumps({"tool": "regenerate", "args": {"hint": "fresh"},
                        "reason": "llm"})
    brain = StubBrainLLM(reply)
    orch = OrchestratorAgent(llm=brain, generator=gen,
                             skill_library=_StubSkillLibrary(skill))
    spec = _spec()
    report = build_defect_report(_clip(), spec)
    # History already shows the skill's single step replayed → exhausted.
    history = [({"via": "skill", "skill_id": "R9", "tool": "regenerate"},
                "rejected", 0.5)]
    d = orch.decide(_clip(), spec, orch.available_actions(), history, report)
    assert d["via"] == "llm"
    assert brain.prompts


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


class _DistillRecorderLibrary:
    """Records distill_repair calls and replays a canned skill via
    retrieve_repair. distill_severity_threshold mirrors the real default so the
    loop's gate fires on a non-trivial initial defect."""

    distill_severity_threshold = 0.5

    def __init__(self, retrieve_skill=None):
        self.distilled: list[dict] = []
        self.used: list[str] = []
        self._retrieve = retrieve_skill

    def retrieve_repair(self, defect_report):
        return self._retrieve

    def mark_used(self, skill_id):
        self.used.append(skill_id)

    def distill_repair(self, name, defect_signature, repair_workflow,
                       evidence=None, thresholds=None):
        self.distilled.append({
            "name": name, "defect_signature": defect_signature,
            "repair_workflow": repair_workflow, "evidence": evidence,
        })

        class _S:
            skill_id = "Rdistilled"
        return _S()


def test_orchestrated_loop_distills_repair_workflow_on_convergence(tmp_path):
    """On a stub-driven convergence with a non-trivial initial defect, the loop
    must call distill_repair with the verifier-ACCEPTED action sequence as the
    repair_workflow and the initial defect modalities as the signature."""
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
                         "reason": "remaining semantic miss"})
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    lib = _DistillRecorderLibrary()
    orch = OrchestratorAgent(llm=StubBrainLLM([reply1, reply2]), generator=gen,
                             refiner=RefinerAgent(), skill_library=lib)
    spec = _spec()
    res = generate_shot_orchestrated(
        spec, _board(), gen, RefinerAgent(), VerifierAgent(), tmp_path, orch,
        skill_library=lib, n_candidates=2, max_turns=5,
    )
    assert res.converged
    # A repair workflow was distilled from the accepted sequence.
    assert lib.distilled, "distill_repair must be called on convergence"
    rec = lib.distilled[-1]
    assert rec["repair_workflow"], "workflow must carry the accepted steps"
    assert all(step["tool"] == "regenerate" for step in rec["repair_workflow"])
    assert rec["defect_signature"], "signature must carry initial defect modalities"
    assert res.distilled_repair_skill_id == "Rdistilled"


def test_orchestrated_loop_marks_replayed_repair_skill_used(tmp_path):
    """When a repair skill is retrieved and replayed, the loop ledgers its
    usage via mark_used."""
    skill = _StubSkill("Rreplay", [
        {"tool": "regenerate",
         "args_template": {"hint": "one continuous passive ballistic arc under "
                           "gravity | resolve the wall collision with momentum "
                           "conserved | strengthen depiction of 'bounces'"},
         "modality": "motion"},
    ])
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    lib = _DistillRecorderLibrary(retrieve_skill=skill)
    # The LLM is never needed — the skill is replayed.
    orch = OrchestratorAgent(llm=StubBrainLLM('{"tool":"accept","args":{}}'),
                             generator=gen, refiner=RefinerAgent(),
                             skill_library=lib)
    spec = _spec()
    res = generate_shot_orchestrated(
        spec, _board(), gen, RefinerAgent(), VerifierAgent(), tmp_path, orch,
        skill_library=lib, n_candidates=2, max_turns=5,
    )
    assert "Rreplay" in lib.used, "replayed repair skill must be marked used"
    # The replayed step shows up tagged via="skill" in the action trace.
    assert any(a.get("via") == "skill" for a in res.actions), res.actions


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
