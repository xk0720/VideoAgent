"""接缝闪烁事故回归(2026-07-29,movie_20260729_150307 现场取证):
根因 = _trim_head 流拷贝在稀疏关键帧的 AI 片上一帧没裁、只改小时长
元数据;帧数-时长说谎的文件进 concat 按谎报时长排偏移 → 接缝两镜帧
交错 = 闪烁。两道修复各自钉死:
(1) _trim_head 解码级精确裁 + 裁后自检(说谎文件绝不放行);
(2) VideoConcatTool 完整性闸(_copy_safe)+ 重编码拼接兜底。
ffmpeg 相关用 skipif 守护;判定逻辑纯离线。"""

import shutil
import subprocess
from pathlib import Path

import pytest

from maestro.pipeline.window_loop import _probe_seconds, _trim_head
from maestro.tools.video_concat import VideoConcatTool, _copy_safe

_HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="no ffmpeg")


def _synth(path: Path, seconds: float, fps: int = 24,
           sparse_keyframes: bool = True) -> Path:
    """合成测试视频;sparse_keyframes=True 用 -g 999 复现事故前提
    (整段只有一个关键帧 —— AI 生成短片的典型形态)。"""
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
           "-i", f"testsrc=duration={seconds}:size=320x240:rate={fps}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if sparse_keyframes:
        cmd += ["-g", "999"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _nb_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


# ── (1) _trim_head:稀疏关键帧下仍精确裁帧 ──────────────────────────

@needs_ffmpeg
def test_trim_head_frame_accurate_on_sparse_keyframes(tmp_path):
    src = _synth(tmp_path / "src.mp4", seconds=4.0, sparse_keyframes=True)
    out = _trim_head(src, 2.0, tmp_path / "trimmed.mp4")
    assert out is not None
    got = _probe_seconds(out)
    assert abs(got - 2.0) < 0.15, f"duration {got} ≠ 2.0"
    # 事故形态 = 帧全在、时长变小;修复后帧数必须真的少了一半
    frames = _nb_frames(out)
    assert abs(frames - 48) <= 2, f"frames {frames} ≠ ~48(未真裁)"


def test_trim_head_self_check_refuses_liar_output(tmp_path, monkeypatch):
    """自检:ffmpeg"成功"但输出时长没有变(= 旧缺陷形态)→ None。"""
    import maestro.pipeline.window_loop as wl

    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00" * 2048)
    monkeypatch.setattr(wl.shutil, "which", lambda x: "/bin/ffmpeg")

    def _fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"\x00" * 2048)   # 产出非空文件
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(wl.subprocess, "run", _fake_run)
    # src 10s,裁 2s,输出仍 10s(说谎)→ 自检拒绝
    monkeypatch.setattr(wl, "_probe_seconds",
                        lambda p: 10.0)
    assert wl._trim_head(src, 2.0, tmp_path / "out.mp4") is None


# ── (2) concat 完整性闸:判定逻辑(纯离线)────────────────────────────

def _info(nb=48, fps=24.0, dur=2.0, **kw):
    d = {"codec": "h264", "w": 320, "h": 240, "pix_fmt": "yuv420p",
         "fps": fps, "nb_frames": nb, "duration": dur, "has_audio": False}
    d.update(kw)
    return d


def test_copy_safe_detects_lying_duration():
    # 事故实测形态:239 帧 @24fps(≈9.96s)但容器声称 7.96s
    ok, reason = _copy_safe([_info(), _info(nb=239, dur=7.958)])
    assert not ok and "frames/duration mismatch" in reason


def test_copy_safe_detects_param_drift_and_passes_uniform():
    ok, reason = _copy_safe([_info(), _info(fps=30.0, dur=1.6)])
    assert not ok and "params differ" in reason
    ok, _ = _copy_safe([_info(), _info()])
    assert ok
    ok, reason = _copy_safe([_info(), None])
    assert not ok and "unprobeable" in reason


# ── (2b) 重编码拼接兜底:参数不齐的输入照样拼出连续成片 ───────────────

@needs_ffmpeg
def test_concat_reencode_fallback_on_mixed_fps(tmp_path):
    a = _synth(tmp_path / "a.mp4", 1.5, fps=24, sparse_keyframes=False)
    b = _synth(tmp_path / "b.mp4", 1.5, fps=30, sparse_keyframes=False)
    out = VideoConcatTool().run([a, b], tmp_path / "joined.mp4")
    dur = _probe_seconds(out)
    assert abs(dur - 3.0) < 0.25, f"joined duration {dur} ≠ ~3.0"
