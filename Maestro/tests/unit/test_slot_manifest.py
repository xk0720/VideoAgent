"""方案 A(2026-07-16):槽位清单 + 引用校验闸。锁死三件事:
(1) 清单与执行器装配一一对应;(2) 出口闸确定性拦截错编号/补漏提;
(3) enhancer 坏引用重试一次。CPU-only,无网络。"""
import json
from pathlib import Path

from maestro.agents.prompt_enhancer import PromptEnhancerAgent
from maestro.memory.storyboard import ShotEntry
from maestro.pipeline.ref_slots import validate_references
from maestro.pipeline.window_loop import _slot_manifest


def _entry(images=None):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="the cat jumps onto the windowsill")
    e.images = list(images or [])
    return e


class _Prev:
    def __init__(self, tmp):
        self.video_path = str(tmp / "prev.mp4")


def _img(tmp, name, role, desc):
    p = tmp / name
    p.write_bytes(b"\x89PNG\r\n")
    return {"path": str(p), "role": role, "description": desc}


def test_manifest_matches_assembly_order(tmp_path):
    prev = _Prev(tmp_path)
    # tiv2v_window + 首帧角色图:@Video1(尾段)在前,@Image1=本镜图(软参考)
    e = _entry([_img(tmp_path, "kf.png", "first_frame", "an orange tabby cat")])
    rows = _slot_manifest("tiv2v_window", e, prev)
    assert [r["slot"] for r in rows] == ["@Video1", "@Image1"]
    assert rows[1]["content"] == "an orange tabby cat"
    assert all(r["referenceable"] for r in rows)
    # ti2v_prev_plus_keyframe + 两张参考:@Image1=上镜尾帧,自有图从 2 号起
    e2 = _entry([_img(tmp_path, "a.png", "reference", "the cat"),
                 _img(tmp_path, "b.png", "reference", "the sunlit room")])
    rows2 = _slot_manifest("ti2v_prev_plus_keyframe", e2, prev)
    assert [r["slot"] for r in rows2] == ["@Image1", "@Image2", "@Image3"]
    assert rows2[0]["content"].startswith("the previous shot's final frame")
    assert rows2[1]["content"] == "the cat"
    # multi_image_fusion:kling 措辞;上镜尾帧占 1 号;带尾段视频=不可引用行
    rows3 = _slot_manifest("multi_image_fusion", e2, prev, use_prev_tail=True)
    assert [r["slot"] for r in rows3] == [
        "reference image 1", "reference image 2", "reference image 3",
        "the reference video"]
    assert rows3[3]["referenceable"] is False
    # i2v/ti2v_prev_last:锚帧不可 @ 引用
    rows4 = _slot_manifest("ti2v_prev_last", _entry(), prev)
    assert rows4 == [rows4[0]] and rows4[0]["slot"] == "FIRST_FRAME"
    assert rows4[0]["referenceable"] is False
    # t2v:什么都不装配 → 空清单
    assert _slot_manifest("t2v", _entry(), prev) == []


def test_validator_gate_behaviors():
    slots = [{"slot": "@Video1", "referenceable": True,
              "content": "the previous tail"},
             {"slot": "@Image1", "referenceable": True,
              "content": "an orange tabby cat"},
             {"slot": "FIRST_FRAME", "referenceable": False, "content": "x"}]
    # 合法引用 + 全部提及 → 原样通过
    ok, audit = validate_references(
        "Continue @Video1's motion; @Image1, the tabby cat, jumps up.", slots)
    assert audit["ok"] and audit["appended"] == [] and "@Video1" in ok
    # 引用不存在的编号 → 整条作废(错编号绝不出门)
    bad, audit2 = validate_references("Reference @Image9 for the cat.", slots)
    assert bad == "" and audit2["ok"] is False and audit2["unknown"] == ["@Image9"]
    # 漏提可引用槽位 → 自动补一句(带实况语义)
    fixed, audit3 = validate_references("The cat jumps, continue @Video1.",
                                        slots)
    assert audit3["appended"] == ["@Image1"]
    assert "@Image1 shows: an orange tabby cat" in fixed
    # kling 措辞 + 大小写不敏感
    kslots = [{"slot": "reference image 1", "referenceable": True,
               "content": "the cat"}]
    ok2, a4 = validate_references("Use Reference Image 1 as the cat.", kslots)
    assert a4["ok"] and a4["appended"] == []


def test_enhancer_retries_once_on_unknown_ref():
    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)
            self.prompts = []

        def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            return self.replies.pop(0)

    conds = [{"kind": "video", "slot": "@Video1", "referenceable": True,
              "description": "the previous tail"}]
    good = json.dumps({"video_prompt":
                       "Continue @Video1's motion as the cat leaps up "
                       "onto the sunlit windowsill, tracking shot."})
    bad = json.dumps({"video_prompt":
                      "Reference @Image7 for the cat leaping onto the "
                      "windowsill in warm light."})
    # 第一次坏引用 → 带错误反馈重试 → 第二次好 → 采纳
    llm = _LLM([bad, good])
    out = PromptEnhancerAgent(llm=llm).run(
        "the cat jumps", strategy="tiv2v_window", conditions=conds)
    assert out and "@Video1" in out and len(llm.prompts) == 2
    assert "REJECTED" in llm.prompts[1] and "@Image7" in llm.prompts[1]
    # 两次都坏 → None(调用方保留原 prompt,主循环闸再兜底)
    llm2 = _LLM([bad, bad])
    assert PromptEnhancerAgent(llm=llm2).run(
        "the cat jumps", strategy="tiv2v_window", conditions=conds) is None


def test_loop_gate_drops_bad_refs_end_to_end(tmp_path, monkeypatch):
    """主循环:brain 的 video_prompt 引用了不存在的编号 → 出口闸弃用 →
    内容感知兜底模板顶上,错编号绝不到 API。"""
    from maestro.agents.generator import GeneratorAgent
    from maestro.agents.orchestrator import OrchestratorAgent
    from maestro.agents.refiner import RefinerAgent
    from maestro.agents.verifier import VerifierAgent
    from maestro.critics.board import ReviewBoard
    from maestro.critics.semantic import SemanticCritic
    from maestro.models.llm import BaseLLMClient
    from maestro.models.video_gen import MockVideoGenClient
    from maestro.pipeline.window_loop import generate_movie_windowed

    class _Gen(MockVideoGenClient):
        def __init__(self):
            super().__init__(name="mock-window-gen")
            self.prompts = []

        def capabilities(self):
            return {"t2v", "i2v", "t2i"}

        def generate(self, prompt, duration, out_path, fps=8,
                     first_frame=None, reference_images=None, seed=0,
                     reference_video=None):
            self.prompts.append(prompt)
            return super().generate(prompt, duration, out_path, fps=fps,
                                    first_frame=first_frame,
                                    reference_images=reference_images,
                                    seed=seed)

        def text_to_image(self, prompt, out_path, seed=0):
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"MOCK IMAGE\n{prompt}\n", encoding="utf-8")
            return out

    class _BadRefLLM(BaseLLMClient):
        def complete(self, prompt, **kw):
            if "Scene Write" in prompt[:200]:
                return json.dumps({"shots": [
                    {"description": "Shot 1: a red apple rolls across the "
                                    "kitchen counter toward the edge",
                     "duration_s": 5, "end_state": "the apple is rolling"}]})
            if "Image Plan" in prompt[:200]:
                return json.dumps({"strategy": "single_first_frame",
                                   "images": [{"source": "t2i",
                                               "description": "a red apple"}],
                                   "reason": "stub"})
            # 条件决策:i2v_keyframe 没有可引用槽位,却写了 @Image1 → 必拦
            return json.dumps({"strategy": "i2v_keyframe", "reason": "stub",
                               "video_prompt":
                               "Open on @Image1 and roll the apple."})

    vg = _Gen()
    gen = GeneratorAgent(video_gen=vg)
    res = generate_movie_windowed(
        "a red apple rolls", board=ReviewBoard([SemanticCritic()]),
        generator=gen, refiner=RefinerAgent(), verifier=VerifierAgent(),
        orchestrator=OrchestratorAgent(generator=gen), cache_dir=tmp_path,
        llm=_BadRefLLM(), max_turns=1, n_candidates=1)
    assert all("@Image1" not in p for p in vg.prompts), \
        "错编号绝不允许到达生成 API"
    gates = [d for d in res.decisions if d.get("stage") == "ref_validate"]
    assert gates and "unknown refs" in gates[0]["reason"]


def test_manifest_marks_user_assets(tmp_path):
    """修正 B:asset_image 来源的槽位 content 带 "user asset: " 前缀 ——
    enhancer 做"剧本提及 → 编号"翻译时的确定性锚点;t2i 图不带前缀。"""
    prev = _Prev(tmp_path)
    e = _entry([
        {"path": str(tmp_path / "cat.png"), "role": "reference",
         "source": "asset_image",
         "description": "an orange tabby cat curled on a sofa"},
        {"path": str(tmp_path / "bg.png"), "role": "reference",
         "source": "t2i", "description": "a sunlit windowsill"},
    ])
    for im in e.images:
        Path(im["path"]).write_bytes(b"\x89PNG\r\n")
    rows = _slot_manifest("ti2v_prev_plus_keyframe", e, prev)
    assert rows[1]["content"] == "user asset: an orange tabby cat curled on a sofa"
    assert rows[2]["content"] == "a sunlit windowsill"


def test_outline_warns_when_assets_unmentioned(caplog):
    """修正 A:有素材但全剧本无一提及 → 大声警告(不阻断);提及了则安静。"""
    import logging

    from maestro.pipeline.window_loop import _write_outline

    class _LLM:
        def __init__(self, desc):
            self.desc = desc

        def complete(self, prompt, **kw):
            return json.dumps({"shots": [
                {"description": self.desc, "duration_s": 5,
                 "end_state": "x"}]})

    catalog = [{"kind": "identity", "name": "cat", "path": "/x/cat.png",
                "label": "identity: cat an orange tabby cat",
                "desc": "an orange tabby cat"}]
    # maestro logger propagate=False(自带 handler)→ 测试内临时打开传播,
    # 让 caplog(挂在 root)能收到
    mlog = logging.getLogger("maestro")
    old_prop = mlog.propagate
    mlog.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="maestro"):
            _write_outline(_LLM("Shot 1: a quiet beach at dawn with rolling "
                                "waves and seagulls"), "p", catalog,
                           episode_guidance={}, max_shots=3,
                           fallback_fn=lambda: ["fb"])
        assert any("wasting the assets" in r.getMessage()
                   for r in caplog.records)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="maestro"):
            _write_outline(_LLM("Shot 1: the orange tabby cat wakes up on "
                                "the windowsill in warm light"), "p",
                           catalog, episode_guidance={}, max_shots=3,
                           fallback_fn=lambda: ["fb"])
        assert not any("wasting the assets" in r.getMessage()
                       for r in caplog.records)
    finally:
        mlog.propagate = old_prop


def test_skills_carry_asset_formalization_laws():
    """三个 skill 的新法则在场(编辑技能文件时不许弄丢)。"""
    from maestro.skills.loader import load_skill

    assert "ASSET MENTION LAW" in load_skill("scene_write")["body"]
    assert "user asset:" in load_skill("window_generation")["body"]
    enh = load_skill("prompt_enhancer")["body"]
    assert "FORMALIZE ASSET MENTIONS" in enh
    assert "do NOT invent a reference ID" in enh
