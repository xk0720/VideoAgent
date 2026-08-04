"""2026-07-16 大修回归:extend_prev 真续接(路由/裁头/last_image)、
regenerate_segment 免级联重做、repair_severity 容忍旋钮、菜单退役。
CPU-only,无网络。"""
import json
from pathlib import Path

import pytest

from maestro.memory.storyboard import ShotEntry
from maestro.pipeline.window_loop import (
    _condition_menu,
    _generate_with_condition,
    _slot_manifest,
)
from maestro.types import ShotSpec


class _ExtendGen:
    """带 extend 能力的记录桩。"""

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return {"t2v", "i2v", "extend"}

    def extend(self, prompt, video_path, out_path, duration=None, seed=0,
               last_image=None):
        self.calls.append({"fn": "extend", "prompt": prompt,
                           "video": str(video_path),
                           "duration": duration,
                           "last_image": str(last_image) if last_image else None})
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("MOCK EXTENDED")
        return p

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0, reference_video=None):
        self.calls.append({"fn": "generate"})
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("MOCK")
        return p


def _entry(images=None):
    e = ShotEntry(shot_idx=1, scene_idx=1, label="scene 1 shot 2",
                  description="the cat trots to the bowl")
    e.images = list(images or [])
    return e


class _Prev:
    def __init__(self, tmp):
        p = tmp / "prev.mp4"
        p.write_text("MOCK PREV")
        self.video_path = str(p)
        self.end_state = "trotting rightward"


def test_extend_prev_routes_to_extend_and_trims(tmp_path, monkeypatch):
    """extend_prev:调 extend()(不是 generate);源=上镜尾段;输出裁掉
    头部源时长;last_image 只在有 'last' 角色图时传。"""
    import maestro.pipeline.window_loop as wl

    tail = tmp_path / "tail.mp4"
    tail.write_text("TAIL")
    monkeypatch.setattr(wl, "_cut_tail", lambda v, s, o: tail)
    monkeypatch.setattr(wl, "_probe_seconds", lambda v: 2.0)
    trimmed_calls = []

    def _fake_trim(video, seconds, out):
        trimmed_calls.append(seconds)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("TRIMMED")
        return out
    monkeypatch.setattr(wl, "_trim_head", _fake_trim)

    gen = _ExtendGen()
    spec = ShotSpec(shot_idx=1, duration=6.0, prompt="the cat trots on")
    path, cond = _generate_with_condition(
        "extend_prev", _entry(), _Prev(tmp_path), spec, gen,
        tmp_path / "s", seed=0, fps=8, window_tail_s=2.0,
        brain_prompt="the cat keeps trotting; keep the same cat and room")
    call = gen.calls[-1]
    assert call["fn"] == "extend" and call["video"] == str(tail)
    assert call["duration"] == 6.0 and call["last_image"] is None
    assert trimmed_calls == [2.0]              # 裁掉源尾段时长
    assert Path(path).read_text() == "TRIMMED"
    assert cond["strategy"] == "extend_prev"
    assert cond["extended_from"] == str(tail)
    assert "untrimmed" not in cond
    # 有 'last' 角色图 → 作为目标尾帧传入
    pl = tmp_path / "goal.png"
    pl.write_bytes(b"\x89PNG\r\n")
    e2 = _entry([{"path": str(pl), "role": "last",
                  "description": "cat at the bowl"}])
    _generate_with_condition(
        "extend_prev", e2, _Prev(tmp_path), spec, gen, tmp_path / "s2",
        seed=0, fps=8, window_tail_s=2.0)
    assert gen.calls[-1]["last_image"] == str(pl)


def test_extend_prev_untrimmed_honesty(tmp_path, monkeypatch):
    """裁不了(无 ffmpeg)→ 未裁版本 + cond['untrimmed'] 留痕,不装死。"""
    import maestro.pipeline.window_loop as wl
    monkeypatch.setattr(wl, "_cut_tail", lambda v, s, o: None)
    monkeypatch.setattr(wl, "_probe_seconds", lambda v: 3.0)
    monkeypatch.setattr(wl, "_trim_head", lambda v, s, o: None)
    gen = _ExtendGen()
    spec = ShotSpec(shot_idx=1, duration=None, prompt="p")
    path, cond = _generate_with_condition(
        "extend_prev", _entry(), _Prev(tmp_path), spec, gen,
        tmp_path / "s", seed=0, fps=8, window_tail_s=2.0)
    assert cond["untrimmed"] is True
    assert Path(path).read_text() == "MOCK EXTENDED"


def test_extend_prev_menu_and_manifest(tmp_path):
    gen = _ExtendGen()
    prev = _Prev(tmp_path)
    names = {m["name"] for m in _condition_menu(_entry(), prev, gen)}
    assert "extend_prev" in names and "tiv2v_window" not in names
    rows = _slot_manifest("extend_prev", _entry(), prev)
    assert rows[0]["slot"] == "CONTINUATION_SOURCE"
    assert all(not r["referenceable"] for r in rows)   # 无引用通道


def test_repair_severity_tolerates_minor_defects(tmp_path):
    """repair_severity:最坏缺陷低于阈值 → 不进修复,stop_reason 留痕;
    阈值 0 = 关闭(行为不变)。"""
    from maestro.agents.generator import GeneratorAgent
    from maestro.agents.orchestrator import OrchestratorAgent
    from maestro.agents.refiner import RefinerAgent
    from maestro.agents.verifier import VerifierAgent
    from maestro.critics.board import ReviewBoard
    from maestro.critics.semantic import SemanticCritic
    from maestro.models.video_gen import MockVideoGenClient
    from maestro.pipeline.generate_loop import generate_shot_orchestrated

    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    spec = ShotSpec(shot_idx=0, duration=2.0,
                    prompt="a ball falls impossible")  # mock 评审必出缺陷
    res = generate_shot_orchestrated(
        spec, board=ReviewBoard([SemanticCritic()]), generator=gen,
        refiner=RefinerAgent(), verifier=VerifierAgent(),
        orchestrator=OrchestratorAgent(generator=gen),
        cache_dir=tmp_path, max_turns=3, repair_severity=0.99)
    assert res.stop_reason == "minor_defects_tolerated"
    assert not res.actions                     # 一次修复都没花


def test_propagate_interior_uses_flf2v_without_cascade(tmp_path, monkeypatch):
    """重做后的段修复:interior 跨度 → 1 笔 flf2v(左邻尾帧+右邻首帧),
    绝无级联 generate 调用。"""
    import maestro.pipeline.timeline as tl

    class _Seg:
        def __init__(self, i, tmp):
            self.idx = i
            self.start_frame, self.end_frame = i * 24, (i + 1) * 24
            self.video_path = tmp / f"seg{i}.mp4"
            self.video_path.write_text("SEG")
            self.first_frame_path = tmp / f"seg{i}_f.png"
            self.last_frame_path = tmp / f"seg{i}_l.png"
            self.first_frame_path.write_bytes(b"\x89PNG\r\n")
            self.last_frame_path.write_bytes(b"\x89PNG\r\n")

    class _TL:
        degraded = False
        fps = 24.0
        n_frames = 72

        def __init__(self, tmp):
            self.segments = [_Seg(0, tmp), _Seg(1, tmp), _Seg(2, tmp)]

        def segment_for_frame_range(self, fr):
            return self.segments[1]            # interior 段

    class _Gen:
        def __init__(self):
            self.flf, self.gen = 0, 0

        def capabilities(self):
            return {"t2v", "i2v", "flf2v"}

        def frame_to_frame(self, prompt, first_frame, last_frame, out_path,
                           duration=None, seed=0):
            self.flf += 1
            self.first, self.last = str(first_frame), str(last_frame)
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("FLF")
            return p

        def generate(self, *a, **k):
            self.gen += 1
            raise AssertionError("interior repair must not cascade via i2v")

    monkeypatch.setattr(tl, "_same_shot", lambda a, b: True)
    monkeypatch.setattr(tl, "_fit_to_seconds", lambda v, d, o: v)
    monkeypatch.setattr(tl, "extract_frame", lambda v, i, o: None)
    monkeypatch.setattr(tl, "_splice", lambda paths, out: out)

    class _Defect:
        frame_range = (30, 40)
        fix_modality = "motion"

    tmb = tmp_path / "tl"
    tmb.mkdir()
    gen = _Gen()
    out = tl.propagate_repair(_TL(tmb), _Defect(), video_gen=gen,
                              hint="fix", cache_dir=tmp_path / "rep")
    assert out is not None
    assert gen.flf == 1 and gen.gen == 0       # 1 笔 flf2v,零级联
    assert gen.first.endswith("seg0_l.png")    # 左邻【原始】尾帧
    assert gen.last.endswith("seg2_f.png")     # 右邻【原始】首帧


def test_propagate_head_without_anchor_degrades(tmp_path, monkeypatch):
    """head 跨度且无条件首帧 → None(诚实降级到整镜工具)。"""
    import maestro.pipeline.timeline as tl

    class _Seg:
        def __init__(self, i, tmp):
            self.idx = i
            self.start_frame, self.end_frame = i * 24, (i + 1) * 24
            self.video_path = tmp / f"s{i}.mp4"
            self.video_path.write_text("SEG")
            self.first_frame_path = self.last_frame_path = None

    class _TL:
        degraded = False
        fps = 24.0
        n_frames = 48

        def __init__(self, tmp):
            self.segments = [_Seg(0, tmp), _Seg(1, tmp)]

        def segment_for_frame_range(self, fr):
            return self.segments[0]

    class _Defect:
        frame_range = (0, 5)
        fix_modality = "motion"

    class _Gen:
        def capabilities(self):
            return {"t2v", "i2v", "flf2v"}

    d = tmp_path / "t"
    d.mkdir()
    assert tl.propagate_repair(_TL(d), _Defect(), video_gen=_Gen(),
                               cache_dir=tmp_path / "r") is None


def test_regenerate_uses_original_method_closure(tmp_path):
    """R-1(2026-07-17):窗口路径下 regenerate = 严格按原始条件方法重生成
    (闭包被调用,hint 附加);无闭包(旧管线)→ generator.run 旧行为。"""
    from maestro.agents.generator import GeneratorAgent
    from maestro.agents.orchestrator import OrchestratorAgent
    from maestro.critics.board import ReviewBoard
    from maestro.critics.semantic import SemanticCritic
    from maestro.models.video_gen import MockVideoGenClient

    calls = []

    def _regen(seed, hint=""):
        calls.append({"seed": seed, "hint": hint})
        p = tmp_path / f"regen_{seed}.mp4"
        p.write_text("REGEN")
        return p, {"strategy": "extend_prev", "regen_of_original": True}

    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    orch = OrchestratorAgent(generator=gen)
    best = gen.run(ShotSpec(shot_idx=0, duration=2.0, prompt="p"),
                   tmp_path, revision=0, seed=0)
    cand = orch.execute({"tool": "regenerate",
                         "args": {"hint": "keep the cat visible"}},
                        best, ShotSpec(shot_idx=0, duration=2.0, prompt="p"),
                        tmp_path, r=1,
                        board=ReviewBoard([SemanticCritic()]),
                        regen_fn=_regen)
    assert cand is not None and calls[0]["hint"] == "keep the cat visible"
    assert Path(cand.video_path).read_text() == "REGEN"


def test_route_suggestion_deterministic():
    """R-3 方案 B:最坏缺陷覆盖 ≥90% → 建议 regenerate;局部 → 建议
    regenerate_segment;建议进 brain 上下文(_build_user)。"""
    from maestro.agents.defect_report import Defect, DefectReport
    from maestro.agents.orchestrator import OrchestratorAgent

    def _report(lo, hi, n):
        return DefectReport(defects=[Defect(
            kind="physics", entity="x", frame_range=(lo, hi), severity=0.8,
            fix_modality="motion", note="n")], n_frames=n)

    s_local = OrchestratorAgent._route_suggestion(_report(10, 30, 100))
    assert s_local["tool"] == "regenerate_segment"
    assert s_local["frame_range"] == [10, 30]
    s_global = OrchestratorAgent._route_suggestion(_report(0, 96, 100))
    assert s_global["tool"] == "regenerate"
    assert OrchestratorAgent._route_suggestion(None) is None
    assert OrchestratorAgent._route_suggestion(
        DefectReport(defects=[], n_frames=100)) is None


def test_add_transition_failure_never_replayed_nor_crashes(tmp_path):
    """2026-08-04 run7 崩溃回归:add_transition 失败(风控
    DataInspectionFailed)后曾掉进通用执行路径被无守护重放 → 异常上抛
    炸掉整个运行。守护块必须是封闭终点:失败 → 记账 failed → 下一轮,
    绝不重放、绝不崩。"""
    from maestro.agents.generator import GeneratorAgent
    from maestro.agents.orchestrator import OrchestratorAgent
    from maestro.agents.refiner import RefinerAgent
    from maestro.agents.verifier import VerifierAgent
    from maestro.critics.board import ReviewBoard
    from maestro.critics.semantic import SemanticCritic
    from maestro.models.video_gen import MockVideoGenClient
    from maestro.pipeline.generate_loop import generate_shot_orchestrated

    class _TransLLM:
        def complete(self, prompt, **k):
            return json.dumps({"tool": "add_transition",
                               "args": {"prompt": "smooth bridge"},
                               "reason": "junction defect"})

    calls = []

    def _boom(prompt_text, current_video):
        calls.append(prompt_text)
        raise RuntimeError("DataInspectionFailed: risk control")

    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    spec = ShotSpec(shot_idx=1, duration=2.0,
                    prompt="a ball falls impossible")  # mock 评审必出缺陷
    res = generate_shot_orchestrated(
        spec, board=ReviewBoard([SemanticCritic()]), generator=gen,
        refiner=RefinerAgent(), verifier=VerifierAgent(),
        orchestrator=OrchestratorAgent(generator=gen, llm=_TransLLM()),
        cache_dir=tmp_path, max_turns=2, transition_fn=_boom,
        repair_mode="consistency")
    # 没炸就已经是回归的一半;另一半:每轮只试一次(不重放),留痕 failed
    assert len(calls) == 2                      # max_turns=2,一轮一次
    failed = [a for a in res.actions
              if a["tool"] == "add_transition" and a["outcome"] == "failed"]
    assert failed and "DataInspectionFailed" in failed[0]["error"]
    assert getattr(res.clip, "transition_path", None) in (None, "")
