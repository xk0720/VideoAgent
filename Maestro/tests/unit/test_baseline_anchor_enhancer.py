"""需求 1/2(2026-07-15):基线锚点(确定性路线 + 失败隔离)与 prompt
enhancer(技能驱动润色 + 失败时保留原文)。CPU-only,无网络。"""
import json
from pathlib import Path

from maestro.agents.prompt_enhancer import PromptEnhancerAgent
from maestro.pipeline.window_loop import (
    _conditions_for_prompt,
    _generate_baseline_anchor,
)
from maestro.types import AssetMemory, Identity, Shot


class _Gen:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0, reference_video=None):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append({"prompt": prompt, "duration": duration,
                           "first_frame": first_frame,
                           "reference_images": reference_images,
                           "reference_video": reference_video})
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("MOCK")
        return p


def _assets(tmp_path, n_imgs=0, n_vids=0):
    am = AssetMemory()
    for i in range(n_imgs):
        p = tmp_path / f"user{i}.png"
        p.write_bytes(b"\x89PNG\r\n")
        am.identity_anchors[f"id{i}"] = Identity(
            identity_id=f"id{i}", name=f"cat{i}", source=str(p),
            description=f"an orange tabby cat {i}")
    for i in range(n_vids):
        p = tmp_path / f"user{i}.mp4"
        p.write_bytes(b"\x00" * 32)
        am.video_shots[f"s{i}"] = Shot(shot_id=f"s{i}", source_video=str(p),
                                       start_time=0, end_time=4,
                                       caption="boardwalk walk")
    return am


def test_anchor_route_t2v_when_no_assets(tmp_path):
    gen = _Gen()
    a = _generate_baseline_anchor("a red apple rolls", AssetMemory(), gen,
                                  None, tmp_path)
    assert a["route"] == "t2v" and a["via"] == "fallback"
    c = gen.calls[-1]
    assert c["first_frame"] is None and c["reference_video"] is None
    assert c["duration"] is None                      # 未配置 → API 默认
    assert a["prompt"] == "a red apple rolls"          # 无 LLM → 用户指令原文


def test_anchor_route_ti2v_with_images(tmp_path):
    gen = _Gen()
    a = _generate_baseline_anchor("the cat wakes up", _assets(tmp_path, 2),
                                  gen, None, tmp_path, duration=10)
    assert a["route"] == "ti2v"
    c = gen.calls[-1]
    assert c["first_frame"] is not None                # 首图当首帧(用户裁决)
    assert c["duration"] == 10


def test_anchor_route_t2v_refs_with_video(tmp_path, monkeypatch):
    # 假视频文件切不了头段(无 ffprobe 元数据)→ 打桩 _head_clip 直通
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_head_clip", lambda v, s, o: v)
    gen = _Gen()
    a = _generate_baseline_anchor("continue the walk",
                                  _assets(tmp_path, 1, 1), gen, None, tmp_path)
    assert a["route"] == "t2v_refs"
    c = gen.calls[-1]
    assert c["first_frame"] is None                    # 视频在场 → t2v+refs
    assert c["reference_video"] is not None
    assert len(c["reference_images"]) == 1


def test_anchor_failure_never_breaks_the_run(tmp_path):
    a = _generate_baseline_anchor("p", AssetMemory(), _Gen(fail=True),
                                  None, tmp_path)
    assert a is None                                   # 只记日志,不抛


class _EnhLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.reply


def test_prompt_enhancer_polishes_with_skill_and_conditions():
    llm = _EnhLLM(json.dumps({"video_prompt":
        "Continuing directly from @Video1, the glossy red apple rolls off "
        "the counter edge and drops, tracking shot, warm morning light."}))
    enh = PromptEnhancerAgent(llm=llm)
    out = enh.run("the apple drops from the counter",
                  strategy="tiv2v_window",
                  conditions=[{"kind": "video", "slot": "@Video1",
                               "referenceable": True,
                               "description": "the previous shot's tail"}],
                  base_prompt="apple falls", label="scene 1 shot 2")
    assert out and "@Video1" in out
    sent = llm.prompts[0]
    assert "Prompt Enhancer" in sent                   # 技能全文在场
    assert "seedance_t2v" in sent                      # 家族由策略确定性推导
    assert "@Video1" in sent                           # 槽位清单进上下文
    assert "STRICT JSON" in sent


def test_prompt_enhancer_fails_closed():
    assert PromptEnhancerAgent(llm=None).run("d", strategy="t2v",
                                             conditions=[]) is None
    garbage = PromptEnhancerAgent(llm=_EnhLLM("sure, sounds great!"))
    assert garbage.run("d", strategy="t2v", conditions=[]) is None
    tiny = PromptEnhancerAgent(llm=_EnhLLM(json.dumps({"video_prompt": "ok"})))
    assert tiny.run("d", strategy="t2v", conditions=[]) is None  # 过短拒收


def test_conditions_for_prompt_facts(tmp_path):
    from maestro.memory.storyboard import ShotEntry

    kf = tmp_path / "kf.png"
    kf.write_bytes(b"\x89PNG\r\n")
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="d")
    e.images = [{"path": str(kf), "role": "reference",
                 "description": "the tabby cat"}]

    class _P:
        video_path = tmp_path / "prev.mp4"

    conds = _conditions_for_prompt("tiv2v_window", e, _P(), False)
    # 方案 A:媒体条件 = 槽位清单(slot 即执行器将装配的编号,照抄即正确)
    slots = {c["slot"]: c for c in conds if c["kind"] in ("image", "video")}
    assert "the ongoing motion" in slots["@Video1"]["description"]
    assert slots["@Image1"]["description"] == "the tabby cat"
    assert slots["@Image1"]["referenceable"] is True
    # t2v 不装配任何图 → 媒体条件必须为空(旧行为会谎称带图,已修正);
    # 2026-08-05 起恒有一行 prompt_language 状态行(语言随剧本)
    t2v_conds = _conditions_for_prompt("t2v", e, None, False)
    assert [c for c in t2v_conds if c["kind"] != "state"] == []
    assert any(c["role"] == "prompt_language" for c in t2v_conds)
