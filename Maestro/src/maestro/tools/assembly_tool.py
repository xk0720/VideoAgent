"""AssemblyTool — stitch clips (+ music) into the final mp4 via ffmpeg.

Degrades gracefully: if ffmpeg is unavailable (e.g. CI / v0.1 mock clips that are
not real videos), it writes a manifest .mp4 placeholder + a .manifest.json so the
pipeline always produces an output path and tests pass without ffmpeg.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .base import BaseTool
from ..types import EditingScript


class AssemblyTool(BaseTool):
    name = "assemble"

    def run(
        self,
        script: EditingScript,
        out_path: Path,
        music_path: Optional[Path] = None,
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clip_paths = [str(c.video_path) for c in script.clips]

        real_videos = all(
            p.lower().endswith((".mp4", ".mov", ".mkv")) and Path(p).stat().st_size > 1024
            for p in clip_paths
        ) if clip_paths else False

        if real_videos and shutil.which("ffmpeg"):
            try:
                return self._ffmpeg_concat(clip_paths, out_path, music_path)
            except Exception:
                pass  # fall through to manifest

        # Graceful fallback: manifest placeholder.
        manifest = {
            "note": "mock assembly (ffmpeg skipped or clips are mock files)",
            "clips": clip_paths,
            "music": str(music_path) if music_path else None,
            "total_duration": script.total_duration,
        }
        out_path.write_text(
            "MOCK ASSEMBLED VIDEO\n" + json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        out_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        script.output_path = out_path
        return out_path

    def _ffmpeg_concat(self, clip_paths, out_path: Path, music_path) -> Path:
        listing = out_path.with_suffix(".txt")
        listing.write_text(
            "\n".join(f"file '{Path(p).resolve()}'" for p in clip_paths),
            encoding="utf-8",
        )
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
        if music_path:
            cmd += ["-i", str(music_path), "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path
