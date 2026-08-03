"""M2 管线重构回归(dev-music-bailian):
① §A0 剧本三态(用户提供 / LLM 创作 / idea 直通);
② §A1 角色提取(成功 → 正典;垃圾输出 → 空,不编造);
③ 分镜阶段正典合并(同名正典覆盖,分镜只许补缺);
④ 台账持久化:scene_anchors / transition_path round-trip;
⑤ 拼装:转场片插在本镜之前,文件丢失响亮跳过;
⑥ add_transition 工具:菜单门控 + 执行挂路径(终结动作);
⑦ enable_review=False:评审/修复/蒸馏全跳过(评审板被调用即为失败)。
全部离线。"""
import json
from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.agents.generator import GeneratorAgent
from maestro.agents.orchestrator import OrchestratorAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.memory.storyboard import ShotEntry, StoryboardMemory
from maestro.models.llm import BaseLLMClient
from maestro.models.video_gen import MockVideoGenClient
from maestro.pipeline.window_loop import generate_movie_windowed
from maestro.types import CandidateClip, ShotSpec


class _LLM(BaseLLMClient):
    def __init__(self, replies: dict):
        self.replies = dict(replies)   # 关键词 → 回复

    def complete(self, prompt: str, **kwargs) -> str:
        for key, rep in self.replies.items():
            if key in prompt[:400]:
                return rep
        return self.replies.get("*", "{}")


# ── ①② §A0/§A1 ─────────────────────────────────────────────────────

def test_screenplay_three_paths():
    text, via = wl._write_screenplay(None, "an idea", "USER SCRIPT", [])
    assert (text, via) == ("USER SCRIPT", "user")
    llm = _LLM({"Screenplay": json.dumps({"screenplay": "SCENE 1 — x"})})
    text2, via2 = wl._write_screenplay(llm, "an idea", None, [])
    assert via2 == "llm" and text2.startswith("SCENE 1")
    text3, via3 = wl._write_screenplay(None, "an idea", None, [])
    assert (text3, via3) == ("an idea", "idea_passthrough")


def test_character_extract_honest():
    llm = _LLM({"Character Extraction": json.dumps(
        {"characters": {"the baker": "static: slender; dynamic: pose"}})})
    chars, via = wl._extract_characters(llm, "SCENE 1 — bakery")
    assert via == "llm" and "the baker" in chars
    bad = _LLM({"*": "not json at all"})
    chars2, via2 = wl._extract_characters(bad, "SCENE 1")
    assert chars2 == {} and via2 == "unusable"


# ── ③ 正典合并 ──────────────────────────────────────────────────────

def test_outline_merges_cast_canon():
    reply = json.dumps({
        "cast": {"x": "static: WRONG; dynamic: p",
                 "z": "static: new guy; dynamic: p"},
        "setting": "a room",
        "shots": [{"description": "Shot 1: <x> walks", "duration_s": 5,
                   "end_state": "x stands; camera: static"}]})
    llm = _LLM({"*": reply})
    _o, _d, _e, meta, via = wl._write_outline(
        llm, "script", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["shot 1: x"],
        cast_canon={"x": "static: CANON; dynamic: p"})
    assert via == "llm"
    assert meta["cast"]["x"].startswith("static: CANON")   # 正典覆盖
    assert "z" in meta["cast"]                             # 补缺保留


# ── ④ 台账持久化 ────────────────────────────────────────────────────

def test_storyboard_persists_anchors_and_transitions(tmp_path):
    sb = StoryboardMemory.from_outline(["shot 1: x", "shot 2: y"],
                                       path=tmp_path / "sb.json")
    sb.scene_anchors[1] = str(tmp_path / "a.png")
    sb.entries[1].transition_path = str(tmp_path / "t.mp4")
    sb._save()
    back = StoryboardMemory.load(tmp_path / "sb.json")
    assert back.scene_anchors == {1: str(tmp_path / "a.png")}
    assert back.entries[1].transition_path == str(tmp_path / "t.mp4")


# ── ⑤ 拼装插转场 ────────────────────────────────────────────────────

def test_final_cut_inserts_transition(tmp_path):
    sb = StoryboardMemory.from_outline(["shot 1: x", "shot 2: y"],
                                       path=tmp_path / "sb.json")
    for i, e in enumerate(sb.entries):
        v = tmp_path / f"v{i}.mp4"
        v.write_text("MOCK")
        e.video_path = str(v)
    t = tmp_path / "trans.mp4"
    t.write_text("MOCK T")
    sb.entries[1].transition_path = str(t)
    clips, notes = wl._final_cut(sb, tmp_path)
    assert [p.name for p in clips] == ["v0.mp4", "trans.mp4", "v1.mp4"]
    assert any(n.get("action") == "transition_inserted" for n in notes)
    # 丢失 → 响亮跳过
    t.unlink()
    clips2, notes2 = wl._final_cut(sb, tmp_path)
    assert [p.name for p in clips2] == ["v0.mp4", "v1.mp4"]
    assert any(n.get("action") == "transition_missing" for n in notes2)


# ── ⑥ add_transition 工具 ───────────────────────────────────────────

def test_add_transition_menu_and_execute(tmp_path):
    orch = OrchestratorAgent()
    names = [m["name"] for m in orch.available_actions(
        transition_available=True)]
    assert "add_transition" in names
    assert "add_transition" not in [m["name"] for m in
                                    orch.available_actions()]
    best = CandidateClip(shot_idx=1, video_path=tmp_path / "v.mp4",
                         revision=0)
    seen = {}

    def fn(prompt_text, current_video):
        seen.update(p=prompt_text, v=str(current_video))
        out = tmp_path / "trans.mp4"
        out.write_text("MOCK")
        return out

    class _Board:
        def review(self, *a, **k):
            return None

    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="x")
    r = orch.execute({"tool": "add_transition",
                      "args": {"prompt": "smooth bridge"}},
                     best, spec, tmp_path, 1, _Board(), transition_fn=fn)
    assert r is None                                   # 终结动作,不换片
    assert best.transition_path == str(tmp_path / "trans.mp4")
    assert seen["p"] == "smooth bridge"


# ── ⑦ 评审总开关 ────────────────────────────────────────────────────

class _VG(MockVideoGenClient):
    def capabilities(self):
        return {"t2v", "i2v", "t2i"}

    def text_to_image(self, prompt, out_path, seed=0):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("MOCK IMAGE")
        return out


class _BoomBoard:
    def review(self, *a, **k):
        raise AssertionError("review must not run when enable_review=False")

    def all_passed(self, *a, **k):
        raise AssertionError("review must not run")


def test_enable_review_off_skips_review_and_distill(tmp_path, monkeypatch):
    vg = _VG()
    gen = GeneratorAgent(video_gen=vg)
    llm = _LLM({
        "Screenplay": json.dumps({"screenplay": "SCENE 1 — a room"}),
        "Character Extraction": json.dumps({"characters": {}}),
        "Image Plan": json.dumps({"strategy": "none", "images": [],
                                  "reason": "stub"}),
        "*": json.dumps({"strategy": "t2v", "reason": "stub",
                         "shots": [{"description": "Shot 1: a cat sits",
                                    "duration_s": 5,
                                    "end_state": "cat sits; camera: "
                                                 "static"}],
                         "cast": {}, "setting": "a room"}),
    })

    class _Concat:
        def run(self, clips, out):
            out = Path(out)
            out.write_text("MERGED")
            return out
    import maestro.tools.video_concat as vc
    monkeypatch.setattr(vc, "VideoConcatTool", _Concat)
    from maestro.memory.episode_memory import EpisodeMemory
    res = generate_movie_windowed(
        "a cat sits", cache_dir=tmp_path, llm=llm, max_turns=2,
        n_candidates=1, enable_review=False,
        episode_memory=EpisodeMemory(tmp_path / "ep.jsonl"),
        board=_BoomBoard(), generator=gen, refiner=RefinerAgent(),
        verifier=VerifierAgent(), orchestrator=OrchestratorAgent(
            generator=gen))
    assert res.storyboard.all_generated()
    assert res.episode_id == ""            # 蒸馏跳过(无客观信号)
    assert res.final_video is not None
    stages = {d.get("stage") for d in res.decisions}
    assert {"screenplay", "character_extract"} <= stages
