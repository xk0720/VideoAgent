"""VideoConcatTool — editing category. Concatenate clips into a single file.

Lower-level than AssemblyTool: this is a pure ffmpeg concat with no music / no
manifest fallback dressing. AssemblyTool stays as the high-level pipeline-stage
wrapper. Splitting them mirrors UniVA's separation between "compose a final"
(workflow stage) and "concat files" (atomic editing primitive).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from .base import BaseTool

log = get_logger("concat")


def _stream_info(p: Path) -> Optional[dict]:
    """单文件流参数(拼接完整性判定用);探测失败 → None。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(p)],
            capture_output=True, text=True, timeout=30, check=True).stdout
        data = json.loads(out)
        v = next((st for st in data.get("streams", [])
                  if st.get("codec_type") == "video"), None)
        if v is None:
            return None
        num, _, den = str(v.get("avg_frame_rate", "0/1")).partition("/")
        fps = (float(num) / float(den)) if float(den or 1) else 0.0
        return {
            "codec": v.get("codec_name"), "w": v.get("width"),
            "h": v.get("height"), "pix_fmt": v.get("pix_fmt"),
            "fps": round(fps, 3), "nb_frames": int(v.get("nb_frames") or 0),
            "duration": float(data.get("format", {}).get("duration", 0.0)
                              or 0.0),
            "has_audio": any(st.get("codec_type") == "audio"
                             for st in data.get("streams", [])),
        }
    except Exception:
        return None


def _copy_safe(infos: list[Optional[dict]]) -> tuple[bool, str]:
    """`-c copy` 拼接是否安全(2026-07-29 闪烁事故防线):
    ① 每个文件自身"帧数×帧率 ≈ 容器时长"(不符 = 元数据说谎,concat
      会按谎报时长排偏移 → 接缝处两镜帧交错 = 闪烁);
    ② 所有文件的 编码/分辨率/帧率/像素格式 完全一致。
    任一不满足 → (False, 原因),调用方改走重编码拼接。"""
    if any(i is None for i in infos):
        return False, "unprobeable input"
    for i, info in enumerate(infos):
        if info["fps"] > 0 and info["nb_frames"] > 0 and info["duration"] > 0:
            implied = info["nb_frames"] / info["fps"]
            if abs(implied - info["duration"]) > 0.15:
                return False, (f"clip {i}: frames/duration mismatch "
                               f"({info['nb_frames']}f @ {info['fps']}fps "
                               f"= {implied:.2f}s vs container "
                               f"{info['duration']:.2f}s)")
    keys = ("codec", "w", "h", "fps", "pix_fmt")
    base = {k: infos[0][k] for k in keys}
    for i, info in enumerate(infos[1:], 1):
        diff = {k: info[k] for k in keys if info[k] != base[k]}
        if diff:
            return False, f"clip {i} stream params differ: {diff}"
    return True, ""


class VideoConcatTool(BaseTool):
    name = "video_concat"
    category = "editing"
    description = "Concatenate a list of video files into a single mp4 (lossless when codecs match)."
    side_effects = True

    def run(self, clips: list[str | Path], out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        clip_paths = [Path(p) for p in clips]
        real = bool(shutil.which("ffmpeg")) and all(
            p.exists() and p.stat().st_size > 1024 for p in clip_paths
        )
        if not real:
            # Sandbox fallback: drop a manifest text file (NOT a real mp4); the
            # pipeline's AssemblyTool does the same, kept consistent here.
            out.write_text(
                "MOCK CONCAT\n" + "\n".join(str(p) for p in clip_paths),
                encoding="utf-8",
            )
            return out
        infos = [_stream_info(p) for p in clip_paths]
        safe, reason = _copy_safe(infos)
        if safe:
            listing = out.with_suffix(out.suffix + ".txt")
            listing.write_text(
                "\n".join(f"file '{p.resolve()}'" for p in clip_paths),
                encoding="utf-8",
            )
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(listing), "-c", "copy", str(out)],
                    check=True, capture_output=True, timeout=60,
                )
            finally:
                listing.unlink(missing_ok=True)
            return out
        # ── 完整性闸未过 → 重编码拼接(concat FILTER 全解码重排时间戳,
        # 对元数据说谎/参数不齐的输入免疫;2026-07-29 闪烁事故防线)。
        log.warning("concat: -c copy unsafe (%s) — falling back to "
                    "re-encode concat", reason)
        all_audio = all(i and i["has_audio"] for i in infos)
        cmd = ["ffmpeg", "-y"]
        for p_ in clip_paths:
            cmd += ["-i", str(p_)]
        n = len(clip_paths)
        if all_audio:
            fc = ("".join(f"[{i}:v][{i}:a]" for i in range(n))
                  + f"concat=n={n}:v=1:a=1[v][a]")
            maps = ["-map", "[v]", "-map", "[a]",
                    "-c:a", "aac", "-b:a", "192k"]
        else:
            if any(i and i["has_audio"] for i in infos):
                log.warning("concat: mixed audio presence — re-encode "
                            "concat drops audio (normalize upstream to "
                            "keep it)")
            fc = ("".join(f"[{i}:v]" for i in range(n))
                  + f"concat=n={n}:v=1[v]")
            maps = ["-map", "[v]"]
        cmd += ["-filter_complex", fc, *maps,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return out
