"""Global run state (kept small; AssetMemory is referenced, not inlined)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..types import AssetMemory, EditingScript, ShotSpec


@dataclass
class MaestroState:
    user_prompt: str
    asset_memory: AssetMemory
    cache_dir: Path
    output_path: Path
    shot_specs: list[ShotSpec] = field(default_factory=list)
    script: Optional[EditingScript] = None
    fps: int = 8
