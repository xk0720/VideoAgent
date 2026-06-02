"""VideoProbeTool — analysis category. Read duration / fps / size of a video.

UniVA exposes such read-only inspectors under its Analysis tool category. Maestro
uses them BEFORE generation to ground planning in actual asset properties (e.g.
infer shot count from source duration). v0.2.2: probes via `ffprobe` when on
PATH, else falls back to filesystem heuristics so CI / sandbox runs don't crash.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .base import BaseTool


class VideoProbeTool(BaseTool):
    name = "video_probe"
    category = "analysis"
    description = "Inspect duration / fps / resolution / size of a video file."

    def run(self, path: str | Path) -> dict:
        p = Path(path)
        if not p.exists():
            return {"path": str(p), "exists": False, "duration": 0.0}
        info = {"path": str(p), "exists": True, "size": p.stat().st_size}
        if shutil.which("ffprobe"):
            try:
                cmd = [
                    "ffprobe", "-v", "error", "-print_format", "json",
                    "-show_format", "-show_streams", str(p),
                ]
                out = subprocess.run(cmd, capture_output=True, check=True, timeout=10)
                data = json.loads(out.stdout)
                fmt = data.get("format", {})
                vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
                v0 = vstreams[0] if vstreams else {}
                # `avg_frame_rate` is "num/den"; evaluate safely.
                fr = v0.get("avg_frame_rate", "0/1")
                try:
                    num, den = fr.split("/")
                    fps = float(num) / float(den) if float(den) else 0.0
                except Exception:
                    fps = 0.0
                info.update({
                    "duration": float(fmt.get("duration", 0.0)),
                    "fps": fps,
                    "width": int(v0.get("width", 0)),
                    "height": int(v0.get("height", 0)),
                    "codec": v0.get("codec_name", ""),
                    "source": "ffprobe",
                })
                return info
            except Exception:
                pass  # fall through to heuristic
        # Sandbox / mock fallback: estimate duration from filesize (so tests work
        # without ffmpeg on the box). Never returns 0 for a non-empty file.
        info.update({
            "duration": max(1.0, info["size"] / 1_000_000.0),  # crude proxy
            "fps": 0.0, "width": 0, "height": 0,
            "codec": "unknown", "source": "heuristic",
        })
        return info
