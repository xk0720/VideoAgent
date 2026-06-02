"""Stage 0 — Material Understanding (offline). Builds AssetMemory from user
materials.

v0.2.2 wiring: when an `ActAgent` is supplied, asset perception routes through
the UniVA-style tool registry — `video_probe` for source-video properties,
`caption` for shot/image description, `detect_objects` for identity bboxes.
That makes the analysis-category tools actually load-bearing in production
(not just unit-tested), and the production trajectory now shows the
Plan→Act handoff (`tool_call` events) end-to-end.

v0.3 will swap the mock tool bodies for real CLIP / shot detector / InsightFace
behind the SAME ToolCall signature — no pipeline change needed.
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


def _act_call(act_agent, name: str, *args, **kwargs):
    """Best-effort tool call: returns the value or None on failure (we never
    want asset perception to crash a generation run)."""
    from ..agents.act import ToolCall

    res = act_agent.call(ToolCall(name=name, args=list(args), kwargs=dict(kwargs)))
    return res.value if res.ok else None


def build_asset_memory(
    source_videos: Optional[list[Path]] = None,
    images: Optional[list[Path]] = None,
    music: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    config: Optional[dict] = None,
    act_agent=None,                    # Optional[ActAgent] (UniVA Plan→Act, v0.2.2)
) -> AssetMemory:
    """Build AssetMemory. When `act_agent` is provided, uses it to invoke the
    analysis tools (video_probe / caption / detect_objects) so the trajectory
    captures the full perception step — back-compatible: no act_agent → pure
    mock behavior, identical to v0.2.1.
    """
    source_videos = source_videos or []
    images = images or []
    mem = AssetMemory()

    for vi, v in enumerate(source_videos):
        stem = Path(v).stem
        # Probe + caption when an ActAgent is wired. Probe gives us the actual
        # duration so we can split into N shots proportional to runtime
        # (instead of always 2). Caption replaces the hard-coded "footage from"
        # placeholder with a tool-grounded description.
        if act_agent is not None:
            probe = _act_call(act_agent, "video_probe", v) or {}
            duration = float(probe.get("duration", 4.0))
            caption_template = _act_call(act_agent, "caption", v, kind="video") or stem
        else:
            duration = 4.0
            caption_template = f"source footage from {stem}"

        # Slice into N shots; cap at 4 to keep the planning surface tractable.
        n_shots = max(1, min(4, int(round(duration / 2.0))))
        seg = duration / n_shots
        for si in range(n_shots):
            sid = f"{stem}__s{si:03d}"
            cap = f"{caption_template} (shot {si})"
            mem.video_shots[sid] = Shot(
                shot_id=sid, source_video=str(v),
                start_time=round(si * seg, 3),
                end_time=round((si + 1) * seg, 3),
                caption=cap, clip_embedding=embed_text(cap),
            )

    for ii, img in enumerate(images):
        stem = Path(img).stem
        iid = f"id_{stem}"
        # Caption + bbox grounding via ActAgent if available.
        if act_agent is not None:
            description = _act_call(act_agent, "caption", img, kind="image") \
                or f"identity anchor from {stem}"
            # detect_objects returns list[dict]; we keep the first bbox as the
            # primary identity locus (real impl would track across frames).
            dets = _act_call(act_agent, "detect_objects", img, stem) or []
            if dets:
                description += f" | bbox={dets[0]['bbox']}"
        else:
            description = f"identity anchor from {stem}"

        mem.identity_anchors[iid] = Identity(
            identity_id=iid, name=stem, source=str(img),
            description=description,
            embedding=embed_text(stem),
        )
        mem.style_anchors.append(
            StyleRef(style_id=f"style_{stem}", source=str(img),
                     description=f"style ref from {stem}", embedding=embed_text(stem))
        )

    if music:
        mem.music_profile = _mock_music_profile(Path(music))

    return mem
