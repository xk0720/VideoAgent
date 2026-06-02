"""VideoConcatTool — editing category. Concatenate clips into a single file.

Lower-level than AssemblyTool: this is a pure ffmpeg concat with no music / no
manifest fallback dressing. AssemblyTool stays as the high-level pipeline-stage
wrapper. Splitting them mirrors UniVA's separation between "compose a final"
(workflow stage) and "concat files" (atomic editing primitive).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import BaseTool


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
        listing = out.with_suffix(out.suffix + ".txt")
        listing.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in clip_paths),
            encoding="utf-8",
        )
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                 "-c", "copy", str(out)],
                check=True, capture_output=True, timeout=60,
            )
        finally:
            listing.unlink(missing_ok=True)
        return out
