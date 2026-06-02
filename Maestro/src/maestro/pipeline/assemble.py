"""Stage 3 — Assembly. Stitch accepted clips (+ music) into the final video."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..tools.assembly_tool import AssemblyTool
from ..types import CandidateClip, EditingScript


def assemble(
    clips: list[CandidateClip],
    output_path: Path,
    music_path: Optional[Path] = None,
    shot_duration: float = 3.0,
) -> EditingScript:
    script = EditingScript(
        clips=clips,
        total_duration=shot_duration * len(clips),
    )
    AssemblyTool().run(script, Path(output_path), music_path)
    return script
