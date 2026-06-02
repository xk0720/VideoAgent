"""FrameExtractTool — analysis/editing bridge. Pull frames from a video.

Used by the Refiner's keyframe-edit path when the source is a real video and we
need a specific frame to hand to the image-edit model. UniVA exposes the same
under its analysis/editing tools. Mock-first: if ffmpeg is absent we write
placeholder text frames so downstream calls still have valid Paths.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import BaseTool


class FrameExtractTool(BaseTool):
    name = "frame_extract"
    category = "analysis"
    description = "Extract one or more frames at given timestamps (seconds) from a video."
    side_effects = True

    def run(
        self,
        video: str | Path,
        timestamps: list[float],
        out_dir: str | Path,
        fmt: str = "png",
    ) -> list[Path]:
        v = Path(video)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        have_ffmpeg = bool(shutil.which("ffmpeg")) and v.exists() and v.stat().st_size > 1024
        for i, t in enumerate(timestamps):
            p = out / f"frame_{i:03d}_t{t:.2f}.{fmt}"
            if have_ffmpeg:
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(v),
                         "-frames:v", "1", "-q:v", "2", str(p)],
                        check=True, capture_output=True, timeout=15,
                    )
                except Exception:
                    p.write_text(f"frame@{t}s of {v.name} (ffmpeg failed)\n",
                                 encoding="utf-8")
            else:
                # Sandbox path — write a placeholder so the pipeline can keep
                # threading Paths through without errors.
                p.write_text(f"MOCK FRAME\nvideo={v}\nt={t}\n", encoding="utf-8")
            paths.append(p)
        return paths
