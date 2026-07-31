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
        self.calls.append({"kind": "generate", "prompt": prompt,
                           "first_frame": str(first_frame) if first_frame else None,
                           "reference_images": [str(p) for p in reference_images]
                           if reference_images else None,
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
    """按提问类别回固定 JSON 的窗口 brain 桩(靠技能标题区分决策类型)。"""

    def __init__(self, keyframe="t2i", condition="i2v_keyframe"):
        self.keyframe, self.condition = keyframe, condition
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if "Image Plan" in prompt[:200]:      # image_plan 技能全文打头
            return json.dumps({
                "strategy": "single_first_frame",
                "images": [{"source": self.keyframe,
                            "description": "opening frame of the shot"}],
                "reason": "stub"})
        return json.dumps({"strategy": self.condition, "reason": "stub"})


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
    # 2026-07-16:extend_prev 顶替 tiv2v_window(需要 extend 能力+方法,
    # 本桩后端没有 → 不出现;tiv2v_window 已从菜单退役)
    assert names == {"t2v", "i2v_keyframe", "ti2v_prev_last", "flf2v_bridge"}
    assert "tiv2v_window" not in names
    # 能力收窄 → 对应策略消失
    plain = MockVideoGenClient()                        # 只有 t2v/i2v
    names2 = {m["name"] for m in _condition_menu(e1, e0, plain)}
    assert "flf2v_bridge" not in names2 and "extend_prev" not in names2


def test_keyframe_menu_gating():
    vg = _WindowVideoGen()
    # 无素材 → t2i + none
    assert {m["name"] for m in _keyframe_menu(vg, AssetMemory())} == {"t2i", "none"}
    # 无 t2i 能力、无素材 → 只剩 none(诚实:没法造就是没法造)
    assert {m["name"] for m in _keyframe_menu(MockVideoGenClient(),
                                              AssetMemory())} == {"none"}


def test_decide_three_layer_fallback():
    menu = [{"name": "flf2v_bridge"}, {"name": "t2v"}]
    # 1) 2026-07-31 用户裁决:episode 命中【不再短路】—— 只作建议注入
    # 上下文,LLM 照常被咨询并可以选别的
    class _Capture(BaseLLMClient):
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            return '{"strategy": "t2v", "reason": "advice weighed, t2v"}'
    cap = _Capture()
    d = _decide(cap, "condition", menu, {}, replay_hint="flf2v_bridge",
                priority=["flf2v_bridge", "t2v"])
    assert d["via"] == "llm" and d["strategy"] == "t2v"   # LLM 说了算
    assert "episode_recommendation" in cap.prompts[0]     # 建议在场
    assert "flf2v_bridge" in cap.prompts[0]
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
    """2026-07-31 用户裁决:episode 记忆只作 guidance —— 历史策略作为
    episode_recommendation 注入上下文,LLM 照常决策,绝不直接继承。"""
    import maestro.pipeline.window_loop as wl
    from maestro.memory.storyboard import StoryboardMemory
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)

    # 合成一条已收敛的历史任务(labels 和新任务的剧本产出一致:scene 1 shot 1..3)
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    old = StoryboardMemory.from_outline(
        [f"Shot {i + 1}: a glass falls off a table" for i in range(3)])
    for i in range(3):
        kf = tmp_path / f"kf{i}.png"; kf.write_bytes(b"\x89PNG")
        old.set_image_plan(i, "single_first_frame",
                           [{"path": str(kf), "role": "first_frame",
                             "source": "t2i", "description": "glass"}])
        old.set_condition(i, {"strategy": "i2v_keyframe"})
        old.add_review(i, {"weighted_total": 0.8, "n_failed": 0})
        old.set_result(i, tmp_path / f"v{i}.mp4", converged=True)
    assert em.distill_episode("a glass falls off a table", old).outcome == "good"

    class _Boom(BaseLLMClient):        # 命中 replay 的决策绝不消耗 LLM 推理
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            return "should not matter"

    boom = _Boom()
    res2 = generate_movie_windowed(
        "the glass falls off the table again", cache_dir=tmp_path / "run2",
        llm=boom, max_turns=1, n_candidates=1, episode_memory=em,
        **_components(_WindowVideoGen()))
    # 裁决后:episode 命中不再产生 via=episode —— LLM 被咨询(_Boom 的
    # 回复不可解析 → 确定性兜底),且咨询的 prompt 里带着历史建议。
    via = {d["via"] for d in res2.decisions
           if d["stage"] in ("image_plan", "condition")}
    assert "episode" not in via, res2.decisions          # 短路已废除
    rec_prompts = [pr for pr in boom.prompts
                   if "episode_recommendation" in pr]
    assert rec_prompts, "历史建议必须注入 LLM 上下文"
    assert any("i2v_keyframe" in pr or "single_first_frame" in pr
               for pr in rec_prompts)                    # 建议内容在场


# ── 诚实性修复回归(对抗审查确认的 2 个 bug)──────────────────────────────
def test_exception_fallback_recorded_with_degraded_from(tmp_path, monkeypatch):
    """种子生成崩溃降级到 t2v 时,台账必须写 degraded_from + 原因——
    绝不能把降级伪装成 brain 主动选了 t2v(否则污染 episode 记忆)。"""
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)

    class _CrashOnceGen(_WindowVideoGen):
        def __init__(self):
            super().__init__()
            self._crashed = False

        def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                     reference_images=None, seed=0, reference_video=None):
            # 第一次带 first_frame 的调用(i2v_keyframe 策略)模拟 API 400
            if first_frame is not None and not self._crashed:
                self._crashed = True
                raise RuntimeError("WaveSpeed submit failed: HTTP 400 — boom")
            return super().generate(prompt, duration, out_path, fps=fps,
                                    seed=seed)

    res = generate_movie_windowed(
        "a glass falls off a table", cache_dir=tmp_path,
        llm=_JsonLLM(keyframe="t2i", condition="i2v_keyframe"),
        max_turns=1, n_candidates=1, **_components(_CrashOnceGen()))
    crashed = [e for e in res.storyboard.entries
               if e.condition.get("degraded_from")]
    assert crashed, "崩溃降级必须留痕"
    c = crashed[0].condition
    assert c["strategy"] == "t2v"                      # 实际执行的
    assert c["degraded_from"] == "i2v_keyframe"        # brain 原本决定的
    assert "400" in c["degraded_reason"]
    assert c["decided_strategy"] == "i2v_keyframe"
    assert c["decided_via"] == "llm"


def test_condition_attributed_to_tournament_winner_not_last_seed(tmp_path, monkeypatch):
    """n_candidates>1 且各 seed 条件不同(seed1 崩溃降级)时,台账条件必须
    归因给【初选胜出者】那个 seed,不是最后一个 seed。"""
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)

    class _CrashSeed1Gen(_WindowVideoGen):
        def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                     reference_images=None, seed=0, reference_video=None):
            if first_frame is not None and seed == 1:
                raise RuntimeError("HTTP 400 seed1")
            return super().generate(prompt, duration, out_path, fps=fps,
                                    first_frame=first_frame, seed=seed)

    res = generate_movie_windowed(
        "a glass falls off a table", cache_dir=tmp_path,
        llm=_JsonLLM(keyframe="t2i", condition="i2v_keyframe"),
        max_turns=1, n_candidates=2, **_components(_CrashSeed1Gen()))
    for e in res.storyboard.entries:
        cond = e.condition
        # mock 评审下两个候选内容相同 → 锦标赛平局取第一个 = seed 0(未降级)
        assert cond["strategy"] == "i2v_keyframe", cond
        assert "degraded_from" not in cond or cond["degraded_from"] is None
        # 但 per_seed 流水必须保留 seed1 的降级记录(有分歧就展开)
        assert "per_seed" in cond
        assert any(c.get("degraded_from") == "i2v_keyframe"
                   for c in cond["per_seed"])


# ── Q1 多图策略(调研落地:seedance ref_images / kling multi-i2v)──────────
class _MultiImageVideoGen(_WindowVideoGen):
    """加上多图能力的窗口 mock;记录多图调用。"""

    def capabilities(self):
        return {"t2v", "i2v", "flf2v", "ref_video", "ref_images",
                "multi_i2v", "t2i"}

    def multi_image_to_video(self, prompt, images, out_path, duration=5,
                             seed=0, video=None):
        self.calls.append({"kind": "multi_i2v", "prompt": prompt,
                           "images": [str(p) for p in images],
                           "video": str(video) if video else None})
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"MOCK VIDEO\nprompt={prompt}\n", encoding="utf-8")
        return out


def test_multi_image_strategies_gated_and_executed(tmp_path, monkeypatch):
    from maestro.memory.storyboard import StoryboardMemory
    from maestro.pipeline.window_loop import _generate_with_condition
    from maestro.types import ShotSpec

    vg = _MultiImageVideoGen()
    sb = StoryboardMemory.from_outline(["Shot 1: a", "Shot 2: b"])
    e0, e1 = sb.entries
    kf = tmp_path / "kf.png"; kf.write_bytes(b"\x89PNG")
    e1.keyframe_path = str(kf)
    e0.video_path = str(tmp_path / "v0.mp4")

    # 门控:ti2v_prev_plus_keyframe 出现;multi_image_fusion 已退役
    # (2026-07-17:kling 融合无指定首帧,与"首帧引用优先"方针冲突,
    # 菜单摘除,执行分支保留兼容旧 episode)
    names = {m["name"] for m in _condition_menu(e1, e0, vg)}
    assert "ti2v_prev_plus_keyframe" in names
    assert "multi_image_fusion" not in names
    no_kf_names = {m["name"] for m in _condition_menu(e0, None, vg)}
    assert "ti2v_prev_plus_keyframe" not in no_kf_names
    assert "multi_image_fusion" not in no_kf_names

    # 执行 ti2v_prev_plus_keyframe:走 t2v+refs 软锚(refs 只在 t2v 端点验证
    # 过)——[尾帧, keyframe] 都进 reference_images,无 first_frame
    import maestro.pipeline.window_loop as wl
    prev_last = tmp_path / "prev_last.png"; prev_last.write_bytes(b"\x89PNG")
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: prev_last)
    spec = ShotSpec(shot_idx=1, duration=5.0, prompt="b continues")
    _, cond = _generate_with_condition(
        "ti2v_prev_plus_keyframe", e1, e0, spec, vg, tmp_path / "g",
        seed=0, fps=8, window_tail_s=2.0)
    assert cond["strategy"] == "ti2v_prev_plus_keyframe"
    assert cond["anchoring"] == "soft_t2v_refs"
    assert cond["reference_images"] == [str(prev_last), str(kf)]
    call = vg.calls[-1]
    assert call["first_frame"] is None                 # t2v 路线,无首帧
    assert call["reference_images"] == [str(prev_last), str(kf)]
    assert "@Image1" in call["prompt"] and "@Image2" in call["prompt"]

    # 执行 multi_image_fusion:[尾帧, keyframe] 进 images 数组
    _, cond2 = _generate_with_condition(
        "multi_image_fusion", e1, e0, spec, vg, tmp_path / "g2",
        seed=0, fps=8, window_tail_s=2.0)
    assert cond2["strategy"] == "multi_image_fusion"
    assert vg.calls[-1]["kind"] == "multi_i2v"
    assert vg.calls[-1]["images"] == [str(prev_last), str(kf)]

    # 降级链:尾帧抽不出 → 两策略都如实降级(degraded_from 保留)
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)
    _, cond3 = _generate_with_condition(
        "ti2v_prev_plus_keyframe", e1, e0, spec, vg, tmp_path / "g3",
        seed=0, fps=8, window_tail_s=2.0)
    assert cond3["strategy"] == "i2v_keyframe"
    assert cond3["degraded_from"] == "ti2v_prev_plus_keyframe"
    _, cond4 = _generate_with_condition(
        "multi_image_fusion", e1, e0, spec, vg, tmp_path / "g4",
        seed=0, fps=8, window_tail_s=2.0)
    assert cond4["strategy"] == "i2v_keyframe"
    assert cond4["degraded_from"] == "multi_image_fusion"


# ── Image Plan(数量+角色+来源;Q-A/Q-B/Q-D 裁决)──────────────────────────
def _plan_entry(tmp_path, desc="a glass falls off a table"):
    from maestro.memory.storyboard import StoryboardMemory
    sb = StoryboardMemory.from_outline([f"Shot 1: {desc}"],
                                       path=tmp_path / "sb.json")
    return sb, sb.entries[0]


def test_image_plan_menu_gating(tmp_path):
    from maestro.pipeline.window_loop import _image_plan_menu
    from maestro.types import Identity

    vg = _MultiImageVideoGen()          # t2i + flf2v + ref_images + multi_i2v
    names = {m["name"] for m in _image_plan_menu(vg, AssetMemory())}
    assert names == {"none", "single_first_frame", "single_reference",
                     "pair_first_last", "pair_reference"}
    # 无任何来源(无 t2i、无素材)→ 只剩 none(没法产图就别许诺计划)
    plain = MockVideoGenClient()
    assert {m["name"] for m in _image_plan_menu(plain, AssetMemory())} == {"none"}
    # 无 t2i 但有素材 → 计划可选(来源=素材)
    kf = tmp_path / "hero.png"; kf.write_bytes(b"\x89PNG")
    mem = AssetMemory(identity_anchors={
        "h": Identity(identity_id="h", name="hero", source=str(kf))})
    names2 = {m["name"] for m in _image_plan_menu(plain, mem)}
    assert "single_first_frame" in names2
    assert "pair_first_last" not in names2      # 无 flf2v 能力


def test_image_plan_execution_mixed_sources_and_roles(tmp_path):
    """Q-B:双图混搭来源;角色按 plan 落进台账;keyframe_path 兼容同步。"""
    from maestro.pipeline.window_loop import _execute_image_plan
    from maestro.types import Identity

    vg = _MultiImageVideoGen()
    hero = tmp_path / "hero_portrait.png"; hero.write_bytes(b"\x89PNG")
    mem = AssetMemory(identity_anchors={
        "h": Identity(identity_id="h", name="hero",
                      description="hero portrait", source=str(hero))})
    sb, entry = _plan_entry(tmp_path)
    decision = {"strategy": "pair_reference", "images": [
        {"source": "asset_image", "description": "hero portrait"},
        {"source": "t2i", "description": "a cozy living room at night"}]}
    plan, images, degraded = _execute_image_plan(
        decision, entry, vg, mem, None, tmp_path / "kf")
    assert plan == "pair_reference" and degraded == ""
    assert [im["role"] for im in images] == ["reference", "reference"]
    assert images[0]["source"] == "asset_image"
    assert images[0]["path"] == str(hero)          # 检索命中用户素材
    assert images[1]["source"] == "t2i"
    sb.set_image_plan(entry.shot_idx, plan, images)
    ff, refs, pf, pl = __import__(
        "maestro.pipeline.window_loop", fromlist=["_entry_images"]
    )._entry_images(sb.get(0))
    assert ff is None and len(refs) == 2           # 参考角色,不冒充首帧


def test_image_plan_degrades_pair_to_single_honestly(tmp_path):
    """pair 第二张产不出 → 降级 single,degraded_from 留痕(台账不说谎)。"""
    from maestro.pipeline.window_loop import _execute_image_plan

    class _T2IFailsSecond(_MultiImageVideoGen):
        def __init__(self):
            super().__init__()
            self._n = 0

        def text_to_image(self, prompt, out_path, seed=0):
            self._n += 1
            if self._n >= 2:
                raise RuntimeError("HTTP 400 — t2i quota")
            return super().text_to_image(prompt, out_path, seed=seed)

    sb, entry = _plan_entry(tmp_path)
    decision = {"strategy": "pair_first_last", "images": [
        {"source": "t2i", "description": "opening frame"},
        {"source": "t2i", "description": "closing frame"}]}
    plan, images, degraded = _execute_image_plan(
        decision, entry, _T2IFailsSecond(), AssetMemory(), None,
        tmp_path / "kf")
    assert plan == "single_first_frame"
    assert degraded == "pair_first_last"
    assert len(images) == 1 and images[0]["role"] == "first_frame"


def test_flf2v_own_pair_and_t2v_own_refs_execution(tmp_path):
    """新条件策略:自有首尾双图 → frame_to_frame;参考角色图 → t2v refs;
    brain 的 video_prompt(语义字段)原样传给生成调用。"""
    from maestro.pipeline.window_loop import (
        _condition_menu,
        _generate_with_condition,
    )
    from maestro.types import ShotSpec

    vg = _MultiImageVideoGen()
    sb, entry = _plan_entry(tmp_path)
    f1 = tmp_path / "f1.png"; f1.write_bytes(b"\x89PNG")
    f2 = tmp_path / "f2.png"; f2.write_bytes(b"\x89PNG")
    sb.set_image_plan(0, "pair_first_last", [
        {"path": str(f1), "role": "first", "source": "t2i", "description": "a"},
        {"path": str(f2), "role": "last", "source": "t2i", "description": "b"}])
    entry = sb.get(0)
    names = {m["name"] for m in _condition_menu(entry, None, vg)}
    assert "flf2v_own_pair" in names
    assert "t2v_own_refs" not in names             # 角色门控:非参考图
    spec = ShotSpec(shot_idx=0, duration=5.0, prompt="p")
    _, cond = _generate_with_condition(
        "flf2v_own_pair", entry, None, spec, vg, tmp_path / "g",
        seed=0, fps=8, window_tail_s=2.0,
        brain_prompt="one continuous dolly from frame A to frame B")
    assert cond["strategy"] == "flf2v_own_pair" and cond["brain_prompt"]
    assert vg.calls[-1]["kind"] == "flf2v"

    # reference 计划 → t2v_own_refs(无上镜也可用),@ImageN prompt 透传
    sb2, e2 = _plan_entry(tmp_path, desc="hero waves")
    sb2.set_image_plan(0, "single_reference", [
        {"path": str(f1), "role": "reference", "source": "asset_image",
         "description": "hero"}])
    e2 = sb2.get(0)
    names2 = {m["name"] for m in _condition_menu(e2, None, vg)}
    assert "t2v_own_refs" in names2
    assert "i2v_keyframe" not in names2            # 角色门控:参考图不当首帧
    _, cond2 = _generate_with_condition(
        "t2v_own_refs", e2, None, spec, vg, tmp_path / "g2",
        seed=0, fps=8, window_tail_s=2.0,
        brain_prompt="@Image1 is the hero — keep his face recognizable")
    assert cond2["reference_images"] == [str(f1)]
    call = vg.calls[-1]
    assert call["kind"] == "generate" and call["first_frame"] is None
    assert call["reference_images"] == [str(f1)]
    assert "@Image1" in call["prompt"]             # brain 的角色化 prompt 生效


def test_asset_retrieval_scores_by_overlap_not_order(tmp_path):
    """Q-D:多素材按描述关键词重叠打分选,不再"拿第一张"。"""
    from maestro.pipeline.window_loop import _retrieve_asset_image
    from maestro.types import Identity

    room = tmp_path / "room.png"; room.write_bytes(b"\x89PNG")
    hero = tmp_path / "hero.png"; hero.write_bytes(b"\x89PNG")
    mem = AssetMemory(identity_anchors={
        "bg": Identity(identity_id="bg", name="living room",
                       description="a cozy living room background at night",
                       source=str(room)),
        "hero": Identity(identity_id="hero", name="hero",
                         description="portrait of the male hero character",
                         source=str(hero))})
    path, label = _retrieve_asset_image("the male hero character smiles", mem)
    assert path == hero                             # 第二个素材才是最优
    assert "hero" in label                          # 裁决 1.2:语义跟着图走
    path2, label2 = _retrieve_asset_image("cozy living room at night", mem)
    assert path2 == room and "living room" in label2


def test_ensure_asset_descriptions_qd_chain(tmp_path):
    """Q-D 打标链:有用户描述不覆盖;无描述且 VLM 能 caption → 回填;
    mock VLM(caption 返回 "")→ 不写,不伪造。"""
    from maestro.models.mllm import MockMLLMClient
    from maestro.pipeline.window_loop import ensure_asset_descriptions
    from maestro.types import Identity

    img = tmp_path / "img.png"; img.write_bytes(b"\x89PNG")
    mem = AssetMemory(identity_anchors={
        "a": Identity(identity_id="a", name="", description="", source=str(img)),
        "b": Identity(identity_id="b", name="", description="user says: a dog",
                      source=str(img))})
    assert ensure_asset_descriptions(mem, MockMLLMClient()) == 0   # mock 不发明

    class _CaptionVLM(MockMLLMClient):
        def caption_image(self, image_path):
            return "background: a cozy living room at night"

    n = ensure_asset_descriptions(mem, _CaptionVLM())
    assert n == 1
    assert mem.identity_anchors["a"].description.startswith("background:")
    assert mem.identity_anchors["b"].description == "user says: a dog"  # 不覆盖


# ── 模型输入语言纪律(用户裁决:模型输入输出一律英文)─────────────────────
def test_all_skill_files_are_english_only():
    """全部 SKILL.md 是模型输入 —— 不允许出现中文字符。"""
    import re

    from maestro.skills.loader import load_skill_catalog

    cjk = re.compile(r"[\u4e00-\u9fff]")
    for name, meta in load_skill_catalog().items():
        text = open(meta["path"], encoding="utf-8").read()
        hits = cjk.findall(text)
        assert not hits, f"skill '{name}' contains CJK characters: {hits[:5]}"


def test_window_brain_prompts_load_the_skill_bodies(tmp_path, monkeypatch):
    """窗口 brain 的两类决策 prompt 必须载入对应技能全文(和修复 brain 载
    orchestrator/SKILL.md 同一机制)。"""
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)
    llm = _JsonLLM()
    generate_movie_windowed(
        "a glass falls off a table", cache_dir=tmp_path, llm=llm,
        max_turns=1, n_candidates=1, **_components(_WindowVideoGen()))
    plan_prompts = [p for p in llm.prompts if "Image Plan" in p[:200]]
    cond_prompts = [p for p in llm.prompts if "Window Generation" in p[:200]]
    assert plan_prompts and cond_prompts
    assert "Role → video-model family" in plan_prompts[0]     # 技能正文在场
    assert "Reference syntax per model family" in cond_prompts[0]


# ── §A 真·LLM playwriting(实测翻车修复:确定性拆条循环填充产重复分镜)──
def test_llm_playwriting_replaces_clause_cycling(tmp_path):
    """2 子句的故事:老拆条按 n_shots=3 循环会让第 3 镜重复第 1 镜;LLM
    剧本由 brain 自己定数量、每镜带细节、互不相同,且能看到历史任务形状。"""
    from maestro.pipeline.window_loop import _write_outline

    class _Playwright(BaseLLMClient):
        def complete(self, prompt, **kw):
            assert "Scene Write" in prompt[:200]      # 技能全文在场
            assert "past_task_shapes" in prompt       # 历史经验在上下文里
            assert "suggested_shot_count" not in prompt   # 数量绝不预设
            return json.dumps({"shots": [
                {"description": "Shot 1: scene 1 — a clear glass teeters on "
                 "the edge of a wooden kitchen table, warm daylight, "
                 "eye-level close-up", "duration_s": 5},
                {"description": "Shot 2: scene 1 — the glass shatters on the "
                 "tile floor, shards scattering outward, low floor-level "
                 "camera", "duration_s": 4},
                {"description": "Shot 3: scene 1 — a young boy kneels, "
                 "collects the shards into his hand and walks away smiling, "
                 "medium shot", "duration_s": 30},
            ]})

    outline, durs, ends, _meta, via = _write_outline(
        _Playwright(), "a glass falls; a boy collects shards", [],
        episode_guidance={"past_task_shapes": [
            {"n_shots": 3, "outcome": "good", "user_prompt": "similar"}]},
        max_shots=6,
        fallback_fn=lambda: ["Shot 1: x", "Shot 2: y", "Shot 3: x"])
    assert via == "llm"
    assert len(outline) == 3
    assert len({o.lower() for o in outline}) == 3       # 绝无重复分镜
    assert all(len(o.split()) >= 10 for o in outline)   # 描述带细节
    # 时长是 brain 定的;越界(30)夹回规划域 [4,10](2026-07-14 裁决)
    assert durs == [5, 4, 10]


def test_llm_playwriting_validation_and_fallback(tmp_path):
    """LLM 输出重复/超上限 → 去重+截断;垃圾输出 → 确定性拆条兜底。"""
    from maestro.pipeline.window_loop import _write_outline

    class _Dupes(BaseLLMClient):
        def complete(self, prompt, **kw):
            return json.dumps({"shots": [
                "Shot 1: the glass teeters on the table edge and tips over",
                "Shot 1: the glass teeters on the table edge and tips over",
                "Shot 2: shards scatter across the tile floor at low angle",
                "Shot 3: a boy collects the shards and leaves smiling",
                "Shot 4: extra beyond the cost cap in this test run",
            ]})

    outline, durs, ends, _meta, via = _write_outline(
        _Dupes(), "p", [], episode_guidance={}, max_shots=3,
        fallback_fn=lambda: ["fb"])
    assert via == "llm"
    assert durs == [None, None]             # 纯字符串形态(brain 没输出时长)
    #                                          → None = 不传 duration,API 用默认
    assert len(outline) == 2                # 硬顶 3 截断后再去重(1 条重复被丢)
    assert len(set(outline)) == len(outline)

    class _Garbage(BaseLLMClient):
        def complete(self, prompt, **kw):
            return "I would suggest maybe some nice shots?"

    outline2, durs2, ends2, _meta2, via2 = _write_outline(
        _Garbage(), "p", [], episode_guidance={}, max_shots=6,
        fallback_fn=lambda: ["Shot 1: fallback split"])
    assert via2 == "fallback" and outline2 == ["Shot 1: fallback split"]
    assert durs2 == [None]                  # 兜底 = 不传 duration(API 默认),
    #                                          绝非 config 预设


def test_guidance_carries_past_task_shapes(tmp_path):
    """episode 记忆给 playwriting 供数量经验:相似任务的 (n_shots, outcome)。"""
    em = EpisodeMemory(tmp_path / "ep.jsonl")
    from maestro.memory.storyboard import StoryboardMemory
    old = StoryboardMemory.from_outline(
        ["Shot 1: a glass falls", "Shot 2: shards scatter"])
    for i in range(2):
        old.set_result(i, tmp_path / f"v{i}.mp4", converged=True)
    em.distill_episode("a glass falls off a table", old)
    g = em.guidance_for("the glass falls from the table")
    assert g["past_task_shapes"] == [
        {"n_shots": 2, "outcome": "good",
         "user_prompt": "a glass falls off a table"}]


def test_review_evidence_recorded_in_ledger(tmp_path, monkeypatch):
    """评审证据量进台账:mock 评审有 checklist 项 → 数字非零;"零证据即
    收敛"的空洞收敛在台账里现形(review_evidence 全 0 才可疑)。"""
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_last_frame", lambda v, o: None)
    res = generate_movie_windowed(
        "a glass falls off a table", cache_dir=tmp_path, llm=_JsonLLM(),
        max_turns=1, n_candidates=1, **_components(_WindowVideoGen()))
    for e in res.storyboard.entries:
        ev = e.reviews[-1]["review_evidence"]
        assert set(ev) == {"checklist_items", "physics_verdicts"}
        assert ev["checklist_items"] > 0     # mock 评审真的说了话
