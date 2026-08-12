"""切镜与片段加工(全部走 ffmpeg 子进程, 不引 moviepy)。

- 切点检测: PySceneDetect ContentDetector(装不上或检不出 → 等间隔兜底);
- 精切: 解码后按秒切, 帧级准确, 统一去音轨(驱动视频不需要音轨);
- 约束加工: wan2.2-animate 输入视频须 2–30s →
    <min_clip_s 的镜头做回文补齐(正放+倒放循环, 生成后掐回原长);
    >max_clip_s 的镜头等分切块。
"""
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from interfaces import Shot, SourceInfo

log = logging.getLogger("hook_remake")

FF_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
          "-pix_fmt", "yuv420p"]


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(cmd)}\n{p.stderr[-800:]}")


def probe(path: str) -> SourceInfo:
    p = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {path}\n{p.stderr[-400:]}")
    info = json.loads(p.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    return SourceInfo(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=round(fps, 3),
        duration_s=round(float(info["format"]["duration"]), 3),
        has_audio=any(s["codec_type"] == "audio" for s in info["streams"]),
    )


def _detect_cuts(path: str, threshold: float,
                 min_scene_len_frames: int) -> Optional[List[Tuple[float, float]]]:
    """PySceneDetect 检测切点; 未安装/失败返回 None(上层走等间隔兜底)。"""
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        log.warning("未安装 scenedetect(pip install 'scenedetect[opencv]'), "
                    "切镜退化为等间隔切分")
        return None
    try:
        scenes = detect(path, ContentDetector(
            threshold=threshold, min_scene_len=min_scene_len_frames))
    except Exception as e:                          # noqa: BLE001 — 兜底而非装死
        log.warning("scenedetect 运行失败(%s), 退化为等间隔切分", e)
        return None
    spans = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    return spans or None


def _uniform_spans(duration_s: float, interval_s: float) -> List[Tuple[float, float]]:
    n = max(1, math.ceil(duration_s / interval_s))
    step = duration_s / n
    return [(round(i * step, 3), round(min((i + 1) * step, duration_s), 3))
            for i in range(n)]


def split_shots(src: SourceInfo, out_dir: Path, threshold: float,
                min_scene_len_frames: int, fallback_interval_s: float,
                max_clip_s: float) -> List[Shot]:
    """切镜 + 超长切块 + 逐段精切落盘, 返回时间轴顺序的 Shot 列表。"""
    spans = _detect_cuts(src.path, threshold, min_scene_len_frames)
    if spans is None or len(spans) <= 1:
        if spans is None or src.duration_s > fallback_interval_s * 1.5:
            log.info("使用等间隔切分(%.1fs/段)", fallback_interval_s)
            spans = _uniform_spans(src.duration_s, fallback_interval_s)
    log.info("切镜: %d 个镜头", len(spans))

    # 超长镜头等分切块(块继承原镜头的 hook 分配)
    chunked: List[Tuple[float, float, Optional[int]]] = []
    for i, (t0, t1) in enumerate(spans):
        dur = t1 - t0
        if dur <= max_clip_s:
            chunked.append((t0, t1, None))
        else:
            n = math.ceil(dur / max_clip_s)
            step = dur / n
            log.info("镜头 %d 长 %.1fs > %.0fs, 切成 %d 块", i, dur, max_clip_s, n)
            for j in range(n):
                chunked.append((round(t0 + j * step, 3),
                                round(min(t0 + (j + 1) * step, t1), 3), i))

    clip_dir = out_dir / "shots"
    clip_dir.mkdir(parents=True, exist_ok=True)
    shots: List[Shot] = []
    for idx, (t0, t1, chunk_of) in enumerate(chunked):
        clip = clip_dir / f"shot_{idx:03d}.mp4"
        # -ss/-to 放在 -i 之后 = 输出侧裁剪, 帧级准确(短源可接受重复解码)
        _run(["ffmpeg", "-y", "-i", src.path, "-ss", f"{t0:.3f}",
              "-to", f"{t1:.3f}", "-an", *FF_ENC, str(clip)])
        shots.append(Shot(idx=idx, t0=t0, t1=t1,
                          duration_s=round(t1 - t0, 3),
                          clip_path=str(clip), chunk_of=chunk_of))
    return shots


def make_driving_clip(shot: Shot, out_dir: Path, min_clip_s: float) -> Tuple[str, bool, float]:
    """驱动视频加工: ≥min_clip_s 直接用原片段; 否则回文补齐(正放+倒放循环)。

    生成结果的前 shot.duration_s 秒与原始动作逐帧对应, conform 阶段只取
    这一截 —— 补齐部分纯为满足 API 下限, 是计费上的已知浪费(设计文档
    §5 的拼带方案解决它, 测试链路不做)。
    返回 (driving_path, padded, driving_s)。
    """
    if shot.duration_s >= min_clip_s:
        return shot.clip_path, False, shot.duration_s

    pad_dir = out_dir / "padded"
    pad_dir.mkdir(parents=True, exist_ok=True)
    rev = pad_dir / f"shot_{shot.idx:03d}_rev.mp4"
    _run(["ffmpeg", "-y", "-i", shot.clip_path, "-vf", "reverse",
          "-an", *FF_ENC, str(rev)])

    # 正放+倒放交替, 循环到 ≥min_clip_s
    n_pairs = math.ceil(min_clip_s / (2 * shot.duration_s))
    lst = pad_dir / f"shot_{shot.idx:03d}_list.txt"
    lines = []
    for _ in range(n_pairs):
        lines.append(f"file '{Path(shot.clip_path).resolve()}'")
        lines.append(f"file '{rev.resolve()}'")
    lst.write_text("\n".join(lines), encoding="utf-8")

    driving = pad_dir / f"shot_{shot.idx:03d}_drv.mp4"
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
          *FF_ENC, "-an", str(driving)])
    driving_s = round(2 * shot.duration_s * n_pairs, 3)
    log.info("镜头 %d 仅 %.2fs < %.1fs, 回文补齐到 %.2fs",
             shot.idx, shot.duration_s, min_clip_s, driving_s)
    return str(driving), True, driving_s


def conform_clip(in_path: str, out_path: str, src: SourceInfo,
                 duration_s: float) -> None:
    """把片段对齐到成片规格: 原片 fps/分辨率(等比缩放+补边), 掐到原镜头时长。
    tpad 克隆尾帧兜底生成略短的情况, 保证 -t 一定够长。"""
    vf = (f"fps={src.fps},"
          f"scale={src.width}:{src.height}:force_original_aspect_ratio=decrease,"
          f"pad={src.width}:{src.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
          f"tpad=stop_mode=clone:stop_duration=2")
    _run(["ffmpeg", "-y", "-i", in_path, "-vf", vf, "-t", f"{duration_s:.3f}",
          "-an", *FF_ENC, out_path])


def concat_and_mux(conform_paths: List[str], src: SourceInfo, out_dir: Path,
                   audio: str) -> str:
    """按时间轴顺序拼接, 可选铺回原片音轨(BGM 节奏感是复刻的灵魂)。"""
    lst = out_dir / "concat_list.txt"
    lst.write_text("\n".join(f"file '{Path(p).resolve()}'"
                             for p in conform_paths), encoding="utf-8")
    silent = out_dir / "remake_silent.mp4"
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
          *FF_ENC, "-an", str(silent)])

    final = out_dir / "remake.mp4"
    if audio == "original" and src.has_audio:
        _run(["ffmpeg", "-y", "-i", str(silent), "-i", src.path,
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
              "-c:a", "aac", "-shortest", str(final)])
    else:
        silent.rename(final)
    return str(final)
