"""Video I/O helpers.

Open-source dependencies:
    • opencv-python  — frame-level read/write
    • ffmpeg-python  — high-level cut/concat (real ffmpeg under the hood)

This module never imports torch / decord, so it stays light. Heavier read
paths (random access at scale) should go through decord in v0.2.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def have_ffmpeg() -> bool:
    """Return True iff an ``ffmpeg`` binary is available on PATH."""
    return shutil.which("ffmpeg") is not None


def _ffprobe(path: Path | str) -> dict:
    """Call system ffprobe and return parsed JSON. Avoids ffmpeg-python dep."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    out = subprocess.check_output(cmd)
    return json.loads(out.decode("utf-8"))


def probe_duration(path: Path | str) -> float:
    info = _ffprobe(path)
    return float(info["format"]["duration"])


def probe_resolution(path: Path | str) -> Tuple[int, int]:
    info = _ffprobe(path)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    return int(stream["width"]), int(stream["height"])


def iter_frames(path: Path | str, stride: int = 1) -> Iterator[np.ndarray]:
    """Yield BGR ndarrays from a video at the given frame stride."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield frame
            idx += 1
    finally:
        cap.release()


def read_frame_at(path: Path | str, t_seconds: float) -> Optional[np.ndarray]:
    """Read one frame at a given time. Returns BGR ndarray or None on miss."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_seconds * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def write_silent_color_clip(
    out_path: Path | str,
    duration_s: float = 5.0,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    color: Tuple[int, int, int] = (40, 40, 200),
) -> Path:
    """Write a deterministic solid-color mp4. Used by tests/fixtures generation."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if have_ffmpeg():
        # color is BGR for cv2 but ffmpeg's color= uses RGB hex; use lavfi color.
        r, g, b = color[2], color[1], color[0]
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"color=c=0x{r:02x}{g:02x}{b:02x}:s={width}x{height}:r={fps}:d={duration_s}",
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        return out
    # Fallback: cv2 VideoWriter (mp4v fallback for environments w/o ffmpeg).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(int(duration_s * fps)):
        vw.write(frame)
    vw.release()
    return out


__all__ = [
    "have_ffmpeg",
    "probe_duration",
    "probe_resolution",
    "iter_frames",
    "read_frame_at",
    "write_silent_color_clip",
]
