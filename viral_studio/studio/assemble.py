"""成片装配: 规格统一 → 逐段配音轨 → 拼接。全部走 ffmpeg 子进程。

约定(全部来自 hook_remake 实战):
  - 生成物的分辨率/帧率不可控(实测 std 15fps / pro 25fps / 704x1258 等杂规格)
    → 一律 conform 到成片规格(默认 720x1280@30, 等比缩放+补边);
  - 生成时长 ≥ 目标时长(seedance 最短 4s) → 掐回目标秒数; tpad 克隆尾帧兜底;
  - reuse_motion 段配资产自带的 BGM 切片(借动作必带 BGM); 其余段静音, 由
    全片 BGM 或后期统一铺 —— 音轨缺失时用无声轨占位, 保证 concat 流一致。
"""
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("viral_studio")

W, H, FPS = 720, 1280, 30
V_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p"]
A_ENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]


def _run(cmd: List[str]) -> None:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(map(str, cmd))[:200]}\n{p.stderr[-600:]}")


def probe_duration(path: str) -> float:
    p = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(p.stdout.strip()) if p.stdout.strip() else 0.0


def conform(src: str, dst: str, duration_s: float,
            audio: Optional[str] = None) -> str:
    """统一规格 + 掐到目标时长 + 配音轨(无音轨则铺静音, 保证流一致)。"""
    vf = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
          f"tpad=stop_mode=clone:stop_duration=2")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if audio and Path(audio).exists():
        cmd += ["-i", audio]
    else:                                  # 静音占位轨
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    cmd += ["-map", "0:v:0", "-map", "1:a:0", "-vf", vf,
            "-t", f"{duration_s:.3f}", *V_ENC, *A_ENC, "-shortest", dst]
    _run(cmd)
    return dst


def concat(parts: List[str], dst: str) -> str:
    """按顺序拼接(所有片段已 conform 到同规格, 走 concat demuxer)。"""
    lst = Path(dst).with_suffix(".txt")
    lst.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in parts),
                   encoding="utf-8")
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
          "-i", str(lst), *V_ENC, *A_ENC, dst])
    return dst


def overlay_bgm(video: str, bgm: str, dst: str, volume: float = 0.8,
                keep_original: bool = True) -> str:
    """全片铺 BGM: keep_original=True 时与原音轨混合(口播段保留人声),
    否则整轨替换。BGM 短于视频则循环。"""
    if keep_original:
        fc = (f"[1:a]aloop=loop=-1:size=2e9,volume={volume}[bg];"
              f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[a]")
    else:
        fc = f"[1:a]aloop=loop=-1:size=2e9,volume={volume}[a]"
    _run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", bgm,
          "-filter_complex", fc, "-map", "0:v:0", "-map", "[a]",
          "-c:v", "copy", *A_ENC, "-shortest", dst])
    return dst


def cut_windows(src: str, windows: List[Tuple[float, float]],
                out_dir: Path, stem: str) -> List[str]:
    """把一次 multiwindow 生成的成片按整数窗口切开(供逐窗核验/重组)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (t0, t1) in enumerate(windows):
        dst = out_dir / f"{stem}_w{i}.mp4"
        _run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ss", f"{t0:.3f}",
              "-to", f"{t1:.3f}", "-an", *V_ENC, str(dst)])
        parts.append(str(dst))
    return parts
