"""Stage 0 — Material Understanding (offline). Builds AssetMemory from user
materials. v0.1 uses mock perception (no GPU); v0.2 swaps in the old repo's
CLIP / shot-detector / InsightFace / all-in-one stack behind the same output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..embeddings import embed_text
from ..types import (
    AssetMemory,
    Identity,
    MusicProfile,
    MusicSection,
    Shot,
    StyleRef,
)


def _mock_music_profile(music: Path) -> MusicProfile:
    bpm = 120.0
    beat_dt = 60.0 / bpm
    beats = [round(i * beat_dt, 3) for i in range(32)]
    sections = [
        MusicSection("intro", 0.0, 4.0, energy_db=-20.0, num_beats=8),
        MusicSection("chorus", 4.0, 12.0, energy_db=-8.0, num_beats=16),
        MusicSection("outro", 12.0, 16.0, energy_db=-18.0, num_beats=8),
    ]
    return MusicProfile(
        audio_path=Path(music), duration=16.0, bpm=bpm,
        beats=beats, downbeats=beats[::4], sections=sections,
    )


def build_asset_memory(
    source_videos: Optional[list[Path]] = None,
    images: Optional[list[Path]] = None,
    music: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> AssetMemory:
    source_videos = source_videos or []
    images = images or []
    mem = AssetMemory()

    for vi, v in enumerate(source_videos):
        stem = Path(v).stem
        for si in range(2):  # mock: 2 shots per source video
            sid = f"{stem}__s{si:03d}"
            cap = f"source footage from {stem}, shot {si}"
            mem.video_shots[sid] = Shot(
                shot_id=sid, source_video=str(v),
                start_time=si * 2.0, end_time=si * 2.0 + 2.0,
                caption=cap, clip_embedding=embed_text(cap),
            )

    for ii, img in enumerate(images):
        stem = Path(img).stem
        iid = f"id_{stem}"
        mem.identity_anchors[iid] = Identity(
            identity_id=iid, name=stem, source=str(img),
            description=f"identity anchor from {stem}",
            embedding=embed_text(stem),
        )
        mem.style_anchors.append(
            StyleRef(style_id=f"style_{stem}", source=str(img),
                     description=f"style ref from {stem}", embedding=embed_text(stem))
        )

    if music:
        mem.music_profile = _mock_music_profile(Path(music))

    return mem
