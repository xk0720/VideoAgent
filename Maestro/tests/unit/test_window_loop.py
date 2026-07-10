"""generate_movie_windowed (R3) — 窗口大循环端到端(mock 后端,CPU-only)。

覆盖:§A playwriting→台账;§B keyframe 策略门控+选择;§C 条件策略门控+窗口
条件真的被用上;§D 小循环复用(评审轨迹嵌入台账);§E 合成;§M episode
蒸馏 + replay 采纳(via="episode")。Mock LLM 回非 JSON → 决策走确定性
fallback,绝不伪造 brain 决策。"""
import json
from pathlib import Path

from maestro.agents.generator import GeneratorAgent
from maestro.agents.orchestrator import OrchestratorAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.consistency import ConsistencyCritic
from maestro.critics.physics import PhysicsCritic
from maestro.critics.rhythm import RhythmCritic
from maestro.critics.semantic import SemanticCritic
from maestro.memory.episode_memory import EpisodeMemory
from maestro.models.llm import BaseLLMClient
from maestro.models.video_gen import MockVideoGenClient
from maestro.pipeline.window_loop import (
    _condition_menu,
    _decide,
    _keyframe_menu,
    generate_movie_windowed,
)
from maestro.types import AssetMemory


class _WindowVideoGen(MockVideoGenClient):
    """Mock 后端 + 全部窗口能力;记录每次调用的条件,供断言。"""

    def __init__(self):
        super().__init__(name="mock-window-gen")
        self.calls: list[dict] = []

    def capabilities(self):
        return {"t2v", "i2v", "flf2v", "ref_video", "t2i"}

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0, reference_video=None):
        self.calls.append({"kind": "generate",
                           "first_frame": str(first_frame) if first_frame else None,
                           "reference_video": str(reference_video)
                           if reference_video else None})
        return super().generate(prompt, duration, out_path, fps=fps,
                                first_frame=first_frame,
                                reference_images=reference_images, seed=seed)

    def frame_to_frame(self, prompt, first_frame, last_frame, out_path,
                       duration=5, seed=0):
        self.calls.append({"kind": "flf2v", "first": str(first_frame),
                           "last": str(last_frame)})
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"MOCK VIDEO\nprompt={prompt}\n", encoding="utf-8")
        return out

    def text_to_image(self, prompt, out_path, seed=0):
        self.calls.append({"kind": "t2i"})
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"MOCK IMAGE\nprompt={prompt}\n", encoding="utf-8")
        return out


class _JsonLLM(BaseLLMClient):
    """按提问类别回固定 JSON 的窗口 brain 桩。"""

    def __init__(self, keyframe="t2i", condition="i2v_keyframe"):
        self.keyframe, self.condition = keyframe, condition
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        pick = self.keyframe if "keyframe" in prompt[:400] else self.condition
        return json.dumps({"strategy": pick, "reason": "stub"})


def _components(video_gen):
    board = ReviewBoard([SemanticCritic(), PhysicsCritic(),
                         ConsistencyCritic(), RhythmCritic()])
    gen = GeneratorAgent(video_gen=video_gen)
    orch = OrchestratorAgent(generator=gen)      # Mock LLM → 内层走 Router 兜底
    return dict(board=board, generator=gen, refiner=RefinerAgent(),
                verifier=VerifierAgent(), orchestrator=orch)


def test_full_windowed_run_updates_storyboard_and_merges(tmp_path, monkeypatch):
    vg = _WindowVideoGen()
    comp = _components(vg)
    # mock 视频不可解码 → 尾帧类策略从菜单消失;stub 掉合成以覆盖 §E 主路径
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(
        wl, "_last_frame",
        lambda video, out: None)
    class _Concat:
        def run(self, clips, out):
            out = Path(out); out.write_text("MERGED", encoding="utf-8")
            self.clips = list(clips)
            return out
    import maestro.tools.video_concat as vc
    monkeypatch.setattr(vc, "VideoConcatTool", _Concat)

    res = generate_movie_windowed(
        "a glass falls off a table; shards scatter on the floor",
        cache_dir=tmp_path, llm=_JsonLLM(), max_turns=1, n_candidates=1,
        episode_memory=EpisodeMemory(tmp_path / "ep.jsonl"), **comp)

    sb = res.storyboard
    assert sb.all_generated()
    assert (tmp_path / "storyboard.json").exists()          # R1 落盘
    for e in sb.entries:
        assert e.keyframe_source == "t2i"                   # §B 选择生效
        assert e.condition["strategy"] == "i2v_keyframe"    # §C 选择生效
        assert e.reviews, "评审轨迹必须嵌入台账"               # §D
        assert e.status in ("verified", "generated_with_defects")
    assert res.final_video is not None                      # §E
    assert res.episode_id                                   # §M
    # 条件真的用上了:i2v_keyframe → generate 带 first_frame
    i2v_calls = [c for c in vg.calls if c["kind"] == "generate"
                 and c["first_frame"]]
    assert i2v_calls, "keyframe 必须真正作为 first_frame 传给生成器"


def test_condition_menu_gating(tmp_path):
    from maestro.memory.storyboard import StoryboardMemory
    vg = _WindowVideoGen()
    sb = StoryboardMemory.from_outline(["Shot 1: a", "Shot 2: b"])
    e0, e1 = sb.entries
    # 无 keyframe、无上镜 → 只有 t2v
    assert {m["name"] for m in _condition_menu(e0, None, vg)} == {"t2v"}
    # 有 keyframe + 上镜已生成 + flf2v/ref_video 能力 → 全菜单
    kf = tmp_path / "kf.png"; kf.write_text("x")
    e1.keyframe_path = str(kf)
    e0.video_path = str(tmp_path / "v0.mp4")
    names = {m["name"] for m in _condition_menu(e1, e0, vg)}
    assert names == {"t2v", "i2v_keyframe", "ti2v_prev_last",
                     "flf2v_bridge", "tiv2v_window"}
    # 能力收窄 → 对应策略消失
    plain = MockVideoGenClient()                        # 只有 t2v/i2v
    names2 = {m["name"] for m in _condition_menu(e1, e0, plain)}
    assert "flf2v_bridge" not in names2 and "tiv2v_window" not in names2


def test_keyframe_menu_gating():
    vg = _WindowVideoGen()
    # 无素材 → t2i + none
    assert {m["name"] for m in _keyframe_menu(vg, AssetMemory())} == {"t2i", "none"}
    # 无 t2i 能力、无素材 → 只剩 none(诚实:没法造就是没法造)
    assert {m["name"] for m in _keyframe_menu(MockVideoGenClient(),
                                              AssetMemory())} == {"none"}


def test_decide_three_layer_fallback():
    menu = [{"name": "flf2v_bridge"}, {"name": "t2v"}]
    # 1) episode replay 命中 → via=episode,LLM 根本不被咨询
    class _Boom(BaseLLMClient):
        def complete(self, prompt, **kw):
            raise AssertionError("must not be called")
    d = _decide(_Boom(), "condition", menu, {}, replay_hint="flf2v_bridge",
                priority=["flf2v_bridge", "t2v"])
    assert d["strategy"] == "flf2v_bridge" and d["via"] == "episode"
    # 2) LLM 严格 JSON → via=llm
    d = _decide(_JsonLLM(condition="t2v"), "condition", menu, {},
                replay_hint=None, priority=["flf2v_bridge", "t2v"])
    assert d["strategy"] == "t2v" and d["via"] == "llm"
    # 3) 垃圾回复/越界 → 确定性优先级,via=fallback
    class _Garbage(BaseLLMClient):
        def complete(self, prompt, **kw):
            return "I think maybe use something nice?"
    d = _decide(_Garbage(), "condition", menu, {}, replay_hint=None,
                priority=["flf2v_bridge", "t2v"])
    assert d["strategy"] == "flf2v_bridge" and d["via"] == "fallback"


def test_replay_adoption_from_episode_memory(tmp_path, monkeypatch):
    """长期记忆的可执行化:历史 good episode 的 per-shot 策略在相似新任务上被
    【直接采纳】(via=episode,LLM 不被咨询)——检索即执行。"""
    import maestro.pipeline.window_loop as wl
    from maestro.memory.storyboard import StoryboardMemory
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)

    # 合成一条已收敛的历史任务(labels 和新任务的剧本产出一致:scene 1 shot 1..3)
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    old = StoryboardMemory.from_outline(
        [f"Shot {i + 1}: a glass falls off a table" for i in range(3)])
    for i in range(3):
        old.set_keyframe(i, tmp_path / f"kf{i}.png", "t2i")
        old.set_condition(i, {"strategy": "i2v_keyframe"})
        old.add_review(i, {"weighted_total": 0.8, "n_failed": 0})
        old.set_result(i, tmp_path / f"v{i}.mp4", converged=True)
    assert em.distill_episode("a glass falls off a table", old).outcome == "good"

    class _Boom(BaseLLMClient):        # 命中 replay 的决策绝不消耗 LLM 推理
        def __init__(self):
            self.called = False

        def complete(self, prompt, **kw):
            self.called = True
            return "should not matter"

    boom = _Boom()
    res2 = generate_movie_windowed(
        "the glass falls off the table again", cache_dir=tmp_path / "run2",
        llm=boom, max_turns=1, n_candidates=1, episode_memory=em,
        **_components(_WindowVideoGen()))
    via = {d["via"] for d in res2.decisions}
    assert via == {"episode"}, res2.decisions           # 检索即执行
    assert not boom.called                              # 零 LLM 消耗
    # 采纳的正是历史策略
    assert all(e.keyframe_source == "t2i" for e in res2.storyboard.entries)
    assert all(e.condition["strategy"] == "i2v_keyframe"
               for e in res2.storyboard.entries)
