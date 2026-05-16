"""AssemblyTool — REAL ffmpeg concat (not mocked).

Open-source dependencies:
    • ffmpeg-python  (https://github.com/kkroening/ffmpeg-python) — wraps ffmpeg.
    • An ffmpeg binary on $PATH.

This is the only tool that is never mocked in v0.1, because part of the
v0.1 acceptance criteria is to produce a real .mp4 file (design doc §11).

Pipeline:
    1. For each EditingSegment:
        • retrieval source: ``ffmpeg -ss <a> -to <b> -i src.mp4 -c copy seg_i.mp4``
          (or transcode if a/b is fractional)
        • generation source: just use gen_video_path as-is
    2. Concat the per-segment intermediates via the ffmpeg concat demuxer.
    3. Optionally overlay music with -filter_complex amix / -map.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    import ffmpeg  # type: ignore
    _HAS_FFMPEG_PY = True
except ImportError:                                            # pragma: no cover
    ffmpeg = None                                              # type: ignore[assignment]
    _HAS_FFMPEG_PY = False

from ..config import AssemblyCfg
from ..logging import logger
from ..types import EditingScript, EditingSegment
from .base import BaseTool


class AssemblyTool(BaseTool):
    name = "assemble_timeline"
    description = "Concat editing segments via ffmpeg into a single mp4 (optionally with music)."

    def __init__(self, cfg: Optional[AssemblyCfg] = None) -> None:
        self.cfg = cfg or AssemblyCfg()

    def run(self, script: EditingScript, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not script.segments:
            raise ValueError("AssemblyTool: script has no segments.")

        with tempfile.TemporaryDirectory(prefix="lva_assemble_") as tmp:
            tmp_dir = Path(tmp)
            seg_files: list[Path] = []
            for i, seg in enumerate(script.segments):
                seg_path = tmp_dir / f"seg_{i:04d}.mp4"
                self._materialize_segment(seg, seg_path)
                seg_files.append(seg_path)

            concat_path = tmp_dir / "concat.txt"
            concat_path.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_files))
            silent_path = tmp_dir / "silent.mp4"
            self._run_ffmpeg_concat(concat_path, silent_path)

            if script.music_path:
                self._mux_music(silent_path, Path(script.music_path), output_path)
            else:
                # shutil.move handles cross-filesystem moves; Path.replace does not.
                shutil.move(str(silent_path), str(output_path))

        script.output_path = output_path
        return output_path

    # ─── helpers ───

    def _materialize_segment(self, seg: EditingSegment, out: Path) -> None:
        if seg.source == "generation":
            if not seg.gen_video_path or not Path(seg.gen_video_path).exists():
                # Generation tool didn't produce a file (e.g. a stub backend);
                # emit a black filler so the timeline survives.
                from ..utils.video_io import write_silent_color_clip
                write_silent_color_clip(out, duration_s=max(0.5, seg.duration),
                                        fps=self.cfg.output_fps, color=(0, 0, 0))
                return
            self._transcode(seg.gen_video_path, out)
            return

        # retrieval — concat clipped windows from each shot.
        with tempfile.TemporaryDirectory(prefix="lva_seg_") as tmp:
            tmp_dir = Path(tmp)
            piece_paths: list[Path] = []
            for j, ((a, b), src_video) in enumerate(zip(seg.shot_trims, seg.source_videos)):
                piece = tmp_dir / f"piece_{j:04d}.mp4"
                self._cut(src_video, a, b, piece)
                piece_paths.append(piece)
            concat_list = tmp_dir / "concat.txt"
            concat_list.write_text("\n".join(f"file '{p.as_posix()}'" for p in piece_paths))
            self._run_ffmpeg_concat(concat_list, out)

    def _cut(self, src: str | Path, start: float, end: float, out: Path) -> None:
        duration = max(0.1, end - start)
        cmd = [
            "ffmpeg", "-y", "-loglevel", self.cfg.loglevel,
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(src),
            "-c:v", self.cfg.output_codec,
            "-pix_fmt", self.cfg.output_pix_fmt,
            "-r", str(self.cfg.output_fps),
            "-an",
            str(out),
        ]
        subprocess.run(cmd, check=True)

    def _transcode(self, src: str | Path, out: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-loglevel", self.cfg.loglevel,
            "-i", str(src),
            "-c:v", self.cfg.output_codec,
            "-pix_fmt", self.cfg.output_pix_fmt,
            "-r", str(self.cfg.output_fps),
            "-an",
            str(out),
        ]
        subprocess.run(cmd, check=True)

    def _run_ffmpeg_concat(self, concat_txt: Path, out: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-loglevel", self.cfg.loglevel,
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", self.cfg.output_codec,
            "-pix_fmt", self.cfg.output_pix_fmt,
            "-r", str(self.cfg.output_fps),
            str(out),
        ]
        subprocess.run(cmd, check=True)

    def _mux_music(self, video_path: Path, music_path: Path, out: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-loglevel", self.cfg.loglevel,
            "-i", str(video_path),
            "-i", str(music_path),
            "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(out),
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:                                  # pragma: no cover
            logger.warning(f"Music mux failed ({e}); shipping silent video.")
            shutil.move(str(video_path), str(out))


__all__ = ["AssemblyTool"]
