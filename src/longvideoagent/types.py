"""Global type definitions for LongVideoEditAgent.

Every cross-module signature in this package uses dataclasses defined here,
NOT bare dicts. This is the single source of truth referenced by:
    LongVideoEditAgent_DESIGN.md §4 ("Core data structures").

This module depends only on the Python standard library + numpy, so it is
safe to import from anywhere (agents, tools, tests). Heavy frameworks
(torch, faiss, transformers, langgraph) live behind separate wrappers.

Open-source library dependency: **numpy** (https://numpy.org).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Memory primitives  (Stage 1 output schema)
# ─────────────────────────────────────────────────────────────────────

ShotScale = Literal["extreme_close_up", "close_up", "medium", "long", "extreme_long"]
ShotMovement = Literal["static", "pan", "tilt", "zoom", "tracking", "handheld"]
ShotAngle = Literal["eye_level", "low", "high", "dutch", "overhead"]
ShotFraming = Literal["single", "two_shot", "group", "ows", "pov"]


@dataclass
class CinematographyTags:
    """Shot-level cinematography metadata produced by perception.cinematography
    (ShotVL / ShotBench in v0.2, mock-tagged in v0.1)."""

    shot_scale: ShotScale = "medium"
    shot_movement: ShotMovement = "static"
    shot_angle: ShotAngle = "eye_level"
    framing: ShotFraming = "single"


@dataclass
class ShotFeatures:
    """Pre-computed numerical features for one shot.

    All arrays are stored on disk as ``.npz`` next to the SQLite metadata
    record; here they may be lazy-loaded by ``memory.store.MemoryStore``.
    """

    clip_embedding: np.ndarray                        # (D,) keyframe-averaged
    start_flow: Optional[np.ndarray] = None           # (H, W, 2)
    end_flow: Optional[np.ndarray] = None             # (H, W, 2)
    start_saliency: Optional[np.ndarray] = None       # (H, W)
    end_saliency: Optional[np.ndarray] = None         # (H, W)
    avg_flow_magnitude: float = 0.0                   # scalar, feeds metric m6


@dataclass
class Shot:
    """The atomic retrieval unit (typically 1–10 seconds).

    ``features`` is heavy and should be lazy-loaded; everything else is cheap.
    """

    shot_id: str                                      # e.g. "movie_a__s00042"
    source_video: str                                 # absolute path string
    start_time: float                                 # seconds, inclusive
    end_time: float                                   # seconds, exclusive
    caption: str = ""                                 # MLLM-generated
    cinematography: CinematographyTags = field(default_factory=CinematographyTags)
    features: Optional[ShotFeatures] = None           # None until loaded
    character_ids: list[str] = field(default_factory=list)
    dialogue: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Event:
    """A handful of semantically related shots (BaSSL grouping / heuristic)."""

    event_id: str
    shot_ids: list[str]
    summary: str = ""
    start_time: float = 0.0                           # in source video time
    end_time: float = 0.0


@dataclass
class Story:
    """Top-level narrative arc spanning one or more events."""

    story_id: str
    title: str
    event_ids: list[str]
    summary: str = ""
    arc_role: Literal["setup", "rising", "climax", "falling", "resolution"] = "rising"


@dataclass
class Character:
    """An identity entity tracked across all source videos."""

    character_id: str
    name: Optional[str] = None
    face_embedding: Optional[np.ndarray] = None
    voice_embedding: Optional[np.ndarray] = None
    appearance_shot_ids: list[str] = field(default_factory=list)
    profile_summary: str = ""


# ─────────────────────────────────────────────────────────────────────
# Music (MusicProfile produced by perception.music_analyzer)
# ─────────────────────────────────────────────────────────────────────

MusicSectionName = Literal[
    "intro", "verse", "chorus", "bridge", "outro", "instrumental", "drop", "buildup"
]


@dataclass
class MusicSection:
    name: MusicSectionName
    start_time: float
    end_time: float
    energy_db: float = -20.0
    num_beats: int = 0


@dataclass
class MusicProfile:
    audio_path: Optional[Path] = None
    duration: float = 0.0
    bpm: float = 120.0
    beats: list[float] = field(default_factory=list)        # all beat onsets, seconds
    downbeats: list[float] = field(default_factory=list)
    sections: list[MusicSection] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# NarrativeMemory — the full Stage-1 output, consumed by Stages 2 & 3
# ─────────────────────────────────────────────────────────────────────


@dataclass
class NarrativeMemory:
    shots: dict[str, Shot] = field(default_factory=dict)
    events: dict[str, Event] = field(default_factory=dict)
    stories: dict[str, Story] = field(default_factory=dict)
    characters: dict[str, Character] = field(default_factory=dict)
    music_profile: Optional[MusicProfile] = None

    # cheap text summary used by ScreenwriterAgent to avoid context overflow.
    def summarize(self, max_chars: int = 2000) -> str:
        parts: list[str] = []
        parts.append(f"#shots={len(self.shots)}, #events={len(self.events)}, "
                     f"#stories={len(self.stories)}, #characters={len(self.characters)}")
        for s in list(self.stories.values())[:5]:
            parts.append(f"[{s.arc_role}] {s.title}: {s.summary}")
        for e in list(self.events.values())[:8]:
            parts.append(f"event {e.event_id} ({e.start_time:.1f}-{e.end_time:.1f}s): {e.summary}")
        text = "\n".join(parts)
        return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────
# Plan & Script (Stage 2 & 3 outputs)
# ─────────────────────────────────────────────────────────────────────

EnergyLevel = Literal["low", "medium", "high", "extreme"]


@dataclass
class SectionPlan:
    """Per-music-section plan emitted by ScreenwriterAgent."""

    music_section_idx: int
    energy_level: EnergyLevel
    visual_tags: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class GlobalStructuralPlan:
    section_plans: list[SectionPlan] = field(default_factory=list)


@dataclass
class SegmentGuidance:
    """Per-segment guidance from DirectorAgent.

    One instance ≈ one cut on the timeline. EditorAgent consumes a list of these.
    """

    segment_idx: int
    parent_section_idx: int
    semantic_query: str
    editing_heuristic: str                              # name from heuristics/presets.yaml
    rhythmic_pacing: list[int] = field(default_factory=list)   # shot durations in beats
    cinematography_hints: list[str] = field(default_factory=list)
    retrieval_feasibility: float = 1.0                  # 0..1, EditorAgent uses this for routing


@dataclass
class EditingSegment:
    """A composed segment ready for assembly."""

    segment_idx: int
    source: Literal["retrieval", "generation"] = "retrieval"
    duration: float = 0.0

    # Retrieval branch
    shot_ids: list[str] = field(default_factory=list)
    shot_trims: list[tuple[float, float]] = field(default_factory=list)
    source_videos: list[str] = field(default_factory=list)     # parallel to shot_ids

    # Generation branch
    gen_prompt: Optional[str] = None
    gen_video_path: Optional[Path] = None
    gen_conditions: Optional[dict[str, Any]] = None

    # Quality
    metric_scores: dict[str, float] = field(default_factory=dict)
    accepted_by_validator: bool = False
    validator_reasons: list[str] = field(default_factory=list)


@dataclass
class EditingScript:
    """Compiled timeline, fed to tools.assembly_tool.AssemblyTool."""

    segments: list[EditingSegment] = field(default_factory=list)
    total_duration: float = 0.0
    music_path: Optional[Path] = None
    output_path: Optional[Path] = None


# ─────────────────────────────────────────────────────────────────────
# Agent trajectory log
# ─────────────────────────────────────────────────────────────────────


@dataclass
class AgentStep:
    """One agent decision; serialized one-per-line to a JSONL file.

    Future RL training (cf. RAGEN / verl in v0.2) will replay these.
    """

    timestamp: float
    agent_name: str
    state_snapshot: dict[str, Any]
    action: str                                         # tool name or "respond"
    action_input: dict[str, Any]
    observation: dict[str, Any]
    reward: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)
