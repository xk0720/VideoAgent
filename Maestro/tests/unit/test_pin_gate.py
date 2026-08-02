"""§G 钉帧完整性闸门(2026-08-02 用户批准,默认关)回归:
① _pin_frame_mad 真测量:帧 2 瞬移的合成片 → 爆表;静止片 → ≈0;
② 布线:阈值开启 + 钉帧路线 + MAD 爆表 → 免费重掷一次(seed+1000),
   台账记 pin_gate 决策与两次测量;
③ 默认关闭:不测量、不重掷、无任何行为变化。
真 ffmpeg 合成测试片(lavfi 本地绘制,无网络)。"""
import json
import subprocess
from pathlib import Path

import maestro.pipeline.window_loop as wl
from maestro.agents.generator import GeneratorAgent
from maestro.agents.orchestrator import OrchestratorAgent
from maestro.agents.refiner import RefinerAgent
from maestro.agents.verifier import VerifierAgent
from maestro.critics.board import ReviewBoard
from maestro.critics.semantic import SemanticCritic
from maestro.models.llm import BaseLLMClient
from maestro.models.video_gen import MockVideoGenClient
from maestro.pipeline.window_loop import generate_movie_windowed


def _lavfi_clip(out: Path, *segments) -> Path:
    """segments = [(颜色, 秒数), ...] → 8fps 小视频(纯本地绘制)。"""
    cmd = ["ffmpeg", "-y"]
    for color, dur in segments:
        cmd += ["-f", "lavfi", "-i", f"color=c={color}:s=64x64:r=8:d={dur}"]
    if len(segments) > 1:
        fc = ("".join(f"[{i}:v]" for i in range(len(segments)))
              + f"concat=n={len(segments)}:v=1[v]")
        cmd += ["-filter_complex", fc, "-map", "[v]"]
    cmd += [str(out)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return out


# ── ① 真测量 ─────────────────────────────────────────────────────────

def test_pin_frame_mad_detects_frame2_teleport(tmp_path):
    # 帧 0,1 黑、帧 2 起全白 —— "钉两帧后瞬移"(144652 实录形状)
    teleport = _lavfi_clip(tmp_path / "t.mp4",
                           ("black", 0.25), ("white", 1))
    mad = wl._pin_frame_mad(teleport, tmp_path)
    assert mad is not None and mad > 100
    # 静止片:相邻帧几乎无差
    still = _lavfi_clip(tmp_path / "s.mp4", ("gray", 1))
    mad2 = wl._pin_frame_mad(still, tmp_path)
    assert mad2 is not None and mad2 < 2
    # 坏输入 → None(不猜)
    bad = tmp_path / "bad.mp4"
    bad.write_text("not video")
    assert wl._pin_frame_mad(bad, tmp_path) is None


def test_pin_frame_mad_covers_frame01_and_junction(tmp_path):
    """对抗核查修正的两个盲区:
    ① 裁头路线上撕裂前移到帧 0→1(仅 1 帧黑 + 白)也要抓到;
    ② 钉帧被完全无视 → 片内无撕裂,只有对上镜末帧(接点)可见。"""
    tear01 = _lavfi_clip(tmp_path / "t01.mp4",
                         ("black", 0.125), ("white", 1))   # 帧 0 黑,1 起白
    mad = wl._pin_frame_mad(tear01, tmp_path)
    assert mad is not None and mad > 100
    # 接点撕裂:本片通体白(片内无撕裂),上镜末帧是黑 → 只有接点分量能抓
    prev = _lavfi_clip(tmp_path / "prev.mp4", ("black", 1))
    smooth_white = _lavfi_clip(tmp_path / "w.mp4", ("white", 1))
    assert wl._pin_frame_mad(smooth_white, tmp_path) is not None
    assert wl._pin_frame_mad(smooth_white, tmp_path) < 2      # 片内看不出
    mad_j = wl._pin_frame_mad(smooth_white, tmp_path, prev_video=prev)
    assert mad_j is not None and mad_j > 100                  # 接点抓到


# ── ②③ 布线(mock 后端夹具,与 test_window_loop 同款)────────────────

class _VG(MockVideoGenClient):
    def __init__(self):
        super().__init__(name="mock-window-gen")
        self.seeds: list = []

    def capabilities(self):
        return {"t2v", "i2v", "t2i"}

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0, reference_video=None):
        self.seeds.append(seed)
        return super().generate(prompt, duration, out_path, fps=fps,
                                first_frame=first_frame,
                                reference_images=reference_images, seed=seed)

    def text_to_image(self, prompt, out_path, seed=0):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("MOCK IMAGE", encoding="utf-8")
        return out


class _LLM(BaseLLMClient):
    def complete(self, prompt: str, **kwargs) -> str:
        if "Image Plan" in prompt[:200]:
            return json.dumps({"strategy": "single_first_frame",
                               "images": [{"source": "t2i",
                                           "description": "opening"}],
                               "reason": "stub"})
        return json.dumps({"strategy": "i2v_keyframe", "reason": "stub"})


def _run(tmp_path, monkeypatch, mads: list, **kw):
    vg = _VG()
    gen = GeneratorAgent(video_gen=vg)
    comp = dict(board=ReviewBoard([SemanticCritic()]), generator=gen,
                refiner=RefinerAgent(), verifier=VerifierAgent(),
                orchestrator=OrchestratorAgent(generator=gen))
    calls = []

    def fake_mad(video, work_dir, prev_video=None):
        calls.append(str(video))
        return mads[min(len(calls) - 1, len(mads) - 1)]
    monkeypatch.setattr(wl, "_pin_frame_mad", fake_mad)

    class _Concat:
        def run(self, clips, out):
            out = Path(out)
            out.write_text("MERGED", encoding="utf-8")
            return out
    import maestro.tools.video_concat as vc
    monkeypatch.setattr(vc, "VideoConcatTool", _Concat)
    kw.setdefault("n_candidates", 1)
    res = generate_movie_windowed(
        "a glass falls off a table", cache_dir=tmp_path, llm=_LLM(),
        max_turns=1, **comp, **kw)
    return res, vg, calls


def test_pin_gate_rerolls_once_and_ledgers(tmp_path, monkeypatch):
    res, vg, calls = _run(tmp_path, monkeypatch, mads=[13.0, 1.0],
                          pin_gate_mad=8.0)
    gates = [d for d in res.decisions if d.get("stage") == "pin_gate"]
    assert [g["action"] for g in gates].count("reroll") >= 1
    # 重掷 seed = 原 seed + 1000;台账记的是重掷后的干净测量
    e0 = res.storyboard.entries[0]
    assert e0.condition["seed"] == 1000
    assert e0.condition["pin_gate_mad"] == 1.0
    assert 1000 in vg.seeds
    assert len(calls) >= 2                      # 掷前掷后各测一次


def test_pin_gate_still_tripped_keeps_clip_honestly(tmp_path, monkeypatch):
    res, _vg, _calls = _run(tmp_path, monkeypatch, mads=[13.0, 12.0],
                            pin_gate_mad=8.0)
    acts = [d["action"] for d in res.decisions
            if d.get("stage") == "pin_gate"]
    assert "still_tripped" in acts              # 保留 + 响亮留痕,不空手
    assert res.storyboard.all_generated()


def test_pin_gate_default_off_measures_nothing(tmp_path, monkeypatch):
    res, vg, calls = _run(tmp_path, monkeypatch, mads=[13.0])
    assert calls == []                          # 关着 = 一次都不测
    assert all(d.get("stage") != "pin_gate" for d in res.decisions)
    assert "pin_gate_mad" not in (res.storyboard.entries[0].condition or {})
    assert 1000 not in vg.seeds


def test_pin_gate_measure_failure_is_loud(tmp_path, monkeypatch):
    """对抗核查修正:显式开闸却测不了 → 响亮留痕,绝不静默哑火。"""
    res, _vg, calls = _run(tmp_path, monkeypatch, mads=[None],
                           pin_gate_mad=8.0)
    assert calls, "开闸必须尝试测量"
    acts = [d["action"] for d in res.decisions
            if d.get("stage") == "pin_gate"]
    assert "measure_failed" in acts
    assert res.storyboard.entries[0].condition.get("pin_gate_mad") is None


def test_pin_gate_noise_does_not_inflate_per_seed(tmp_path, monkeypatch):
    """对抗核查修正:pin_gate_mad 是逐候选噪声,不算条件分歧 ——
    两个健康候选(1.21/1.35)不许把 per_seed 撑爆。"""
    res, _vg, _calls = _run(tmp_path, monkeypatch, mads=[1.21, 1.35],
                            pin_gate_mad=8.0, n_candidates=2)
    assert "per_seed" not in (res.storyboard.entries[0].condition or {})
