"""Global type definitions for Maestro.

All cross-module communication uses these dataclasses / enums, never bare dicts.
Mirrors section 8 of REPORT_AND_INSTRUCTIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Literal

try:  # numpy is the only heavy dep; guard so type import never hard-fails
    import numpy as np

    NDArray = np.ndarray
except Exception:  # pragma: no cover
    np = None  # type: ignore
    NDArray = object  # type: ignore


# ─────────────────────────────────────────────────────────────
# Shared cinematography / perception primitives (reused idea from old repo)
# ─────────────────────────────────────────────────────────────
@dataclass
class CinematographyTags:
    shot_scale: Literal[
        "extreme_close_up", "close_up", "medium", "long", "extreme_long"
    ] = "medium"
    shot_movement: Literal[
        "static", "pan", "tilt", "zoom", "tracking", "handheld"
    ] = "static"
    shot_angle: Literal["eye_level", "low", "high", "dutch", "overhead"] = "eye_level"
    framing: Literal["single", "two_shot", "group", "ows", "pov"] = "single"


@dataclass
class Shot:
    """A shot parsed from an uploaded source video (asset memory)."""

    shot_id: str
    source_video: str
    start_time: float
    end_time: float
    caption: str = ""
    cinematography: CinematographyTags = field(default_factory=CinematographyTags)
    clip_embedding: Optional[NDArray] = None
    character_ids: list[str] = field(default_factory=list)


@dataclass
class Identity:
    """A person/object identity anchor from uploaded image or video."""

    identity_id: str
    name: Optional[str] = None
    embedding: Optional[NDArray] = None
    source: str = ""  # path to image/video the anchor came from
    description: str = ""


@dataclass
class StyleRef:
    style_id: str
    embedding: Optional[NDArray] = None
    description: str = ""
    source: str = ""


@dataclass
class MusicSection:
    name: Literal["intro", "verse", "chorus", "bridge", "outro", "instrumental"]
    start_time: float
    end_time: float
    energy_db: float = 0.0
    num_beats: int = 0


@dataclass
class MusicProfile:
    audio_path: Optional[Path] = None
    duration: float = 0.0
    bpm: float = 120.0
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    sections: list[MusicSection] = field(default_factory=list)


@dataclass
class AssetMemory:
    """Structured memory of user-provided multimodal materials (E1)."""

    video_shots: dict[str, Shot] = field(default_factory=dict)
    identity_anchors: dict[str, Identity] = field(default_factory=dict)
    style_anchors: list[StyleRef] = field(default_factory=list)
    music_profile: Optional[MusicProfile] = None

    def summarize(self) -> str:
        parts = [f"{len(self.video_shots)} source shots"]
        if self.identity_anchors:
            names = [i.name or i.identity_id for i in self.identity_anchors.values()]
            parts.append("identities: " + ", ".join(names))
        if self.style_anchors:
            parts.append(f"{len(self.style_anchors)} style refs")
        if self.music_profile:
            parts.append(
                f"music {self.music_profile.bpm:.0f}bpm / "
                f"{len(self.music_profile.sections)} sections"
            )
        return "; ".join(parts)


# ─────────────────────────────────────────────────────────────
# Physics (C1 / E2)  — the differentiation core
# ─────────────────────────────────────────────────────────────
class PhysFailureMode(str, Enum):
    PENETRATION = "penetration"            # 穿模
    GRAVITY_INERTIA = "gravity_inertia"    # 重力 / 惯性
    COLLISION = "collision"
    FLUID = "fluid"
    OBJECT_PERMANENCE = "object_permanence"  # 物体恒存
    DEFORMATION = "deformation"
    CONSERVATION = "conservation"          # 守恒律 (公认最弱项)


@dataclass
class PhysEntity:
    name: str
    mass: float = 1.0
    init_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # y is height above ground
    forces: list[str] = field(default_factory=lambda: ["gravity"])
    restitution: float = 0.6                  # bounciness on contact, 0..1


@dataclass
class PhysInteraction:
    kind: Literal["collision", "support", "fluid", "constraint"]
    entities: list[str] = field(default_factory=list)


@dataclass
class PhysicsSketch:
    """C1 sketch layer: lightweight physical representation + control signal."""

    entities: list[PhysEntity] = field(default_factory=list)
    interactions: list[PhysInteraction] = field(default_factory=list)
    control_signal: Optional[Path] = None       # trajectory/depth/flow image fed to gen
    expected_modes: list[PhysFailureMode] = field(default_factory=list)  # to watch


@dataclass
class PhysicsVerdict:
    """PhysicsCritic output: localizable -> actionable."""

    mode: PhysFailureMode
    frame_range: tuple[int, int]
    severity: float                # 0-1, higher = worse
    suggested_intervention: str


# ─────────────────────────────────────────────────────────────
# Event Graph IR (GEST-style; borrows Event-Graph arXiv:2604.10383)
# ─────────────────────────────────────────────────────────────
@dataclass
class EventNode:
    """One event: an action by actors on objects, in a time window.
    GEST = Graph of Events in Space and Time."""

    event_id: str
    action: str
    actors: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0


@dataclass
class EventEdge:
    """Typed relation between events (the semantic/logical edges GEST allows)."""

    src: str
    dst: str
    relation: Literal["before", "after", "during", "causes", "enables"] = "before"


@dataclass
class EventGraph:
    nodes: list[EventNode] = field(default_factory=list)
    edges: list[EventEdge] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Planning products
# ─────────────────────────────────────────────────────────────
@dataclass
class ShotSpec:
    shot_idx: int
    duration: float
    prompt: str
    cinematography: CinematographyTags = field(default_factory=CinematographyTags)
    identity_refs: list[str] = field(default_factory=list)
    style_refs: list[str] = field(default_factory=list)
    rhythmic_pacing: list[int] = field(default_factory=list)  # shot durations in beats
    physics_sketch: Optional[PhysicsSketch] = None
    event_graph: Optional[EventGraph] = None
    injected_lessons: list[str] = field(default_factory=list)  # C4
    matched_skill: Optional["Skill"] = None                    # C7 (v0.3)


# ─────────────────────────────────────────────────────────────
# Review / self-improvement (C2 / C3)
# ─────────────────────────────────────────────────────────────
ChecklistKind = Literal["semantic", "physics", "consistency", "rhythm"]


@dataclass
class ChecklistItem:
    question: str                  # verifiable yes/no question
    kind: ChecklistKind
    passed: bool = False
    fix_instruction: str = ""      # populated by critic/refiner when failed


@dataclass
class Checklist:
    items: list[ChecklistItem] = field(default_factory=list)

    @property
    def failed_items(self) -> list[ChecklistItem]:
        return [i for i in self.items if not i.passed]

    @property
    def pass_rate(self) -> float:
        if not self.items:
            return 1.0
        return sum(1 for i in self.items if i.passed) / len(self.items)


@dataclass
class CandidateClip:
    shot_idx: int
    video_path: Path
    keyframes: list[Path] = field(default_factory=list)
    metric_scores: dict[str, float] = field(default_factory=dict)
    physics_verdicts: list[PhysicsVerdict] = field(default_factory=list)
    checklist: Checklist = field(default_factory=Checklist)
    accepted: bool = False
    revision: int = 0
    skipped_items: list[str] = field(default_factory=list)  # escape-hatched

    @property
    def aggregate_score(self) -> float:
        if not self.metric_scores:
            return 0.0
        return sum(self.metric_scores.values()) / len(self.metric_scores)


@dataclass
class Lesson:
    """C4 cross-task experience entry.

    v0.3: extended with A-MEM-style attributes (lesson_id / keywords /
    linked_lesson_ids / confidence) so the LessonLibrary can self-organise
    into an evolving network rather than a flat list. Old JSONL files load
    fine — missing fields take their defaults; `lesson_id` is regenerated
    deterministically from (trigger, fix, mode).
    """

    trigger: str
    fix: str
    failure_mode: Optional[PhysFailureMode] = None
    embedding: Optional[NDArray] = None
    lesson_id: str = ""                       # stable hash of content (filled on load/add)
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    linked_lesson_ids: list[str] = field(default_factory=list)   # A-MEM evolution links
    revised_by: list[str] = field(default_factory=list)          # supersession chain
    confidence: float = 1.0                   # decays with disuse, grows with re-confirmation
    uses: int = 0                             # times retrieved
    born_task_id: str = ""                    # which episodic trace produced it


# ─────────────────────────────────────────────────────────────
# v0.3 — Procedural skill + cross-run entities + preferences
# (see RESEARCH_MEMORY_SKILL.md §4 for design rationale)
# ─────────────────────────────────────────────────────────────
@dataclass
class Skill:
    """C7 PhysicsTyped Skill — a *compiled* shot recipe that an HSI Tier-0
    convergence proved works on non-trivial physics.

    Distinct from Voyager (executable code) and SkillWeaver (web API):
    a Skill is a structured plan template — entities + interactions +
    cinematography + acceptance thresholds — keyed on the set of physical
    failure modes it resolves.
    """

    skill_id: str                                                # stable hash
    name: str
    physical_signature: list[PhysFailureMode] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)            # keyword cues
    entities: list[PhysEntity] = field(default_factory=list)     # parametric template
    interactions: list[PhysInteraction] = field(default_factory=list)
    cinematography_preset: CinematographyTags = field(default_factory=CinematographyTags)
    checklist_template: list[ChecklistItem] = field(default_factory=list)
    acceptance_thresholds: dict[str, float] = field(default_factory=dict)
    coupled_lesson_ids: list[str] = field(default_factory=list)  # auto-inject on retrieve
    embedding: Optional[NDArray] = None                          # fallback text retrieval
    perf_score: float = 0.0                                      # EMA of weighted_total
    uses: int = 0
    last_used_ts: float = 0.0
    parent_id: str = ""                                          # versioning chain


@dataclass
class PersistentEntity:
    """C8 Tier-4 cross-run entity (character / prop / location).

    VideoMemory's Dynamic Memory Bank is per-run; ours is cross-run so the
    same hero on Day 2 reuses the Day-1 face / style / physics profile.
    """

    entity_id: str
    canonical_name: str
    embedding: Optional[NDArray] = None
    source_paths: list[str] = field(default_factory=list)
    style_descriptors: dict[str, str] = field(default_factory=dict)
    appearance_log: list[dict] = field(default_factory=list)     # (task_id, bbox, ctx)
    physics_profile: dict[str, float] = field(default_factory=dict)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0


@dataclass
class UserPreference:
    """C8 Tier-5 per-user cinematic / physics-strictness preferences."""

    user_id: str = "default"
    cinematic_priors: dict[str, str] = field(default_factory=dict)
    style_priors: list[str] = field(default_factory=list)
    physics_strictness: float = 1.0                              # multiplier on p1/p2 weights
    endorsed_lesson_ids: list[str] = field(default_factory=list)
    rejected_lesson_ids: list[str] = field(default_factory=list)


@dataclass
class EpisodicTrace:
    """C8 Tier-1 episodic record — a summary of one finished task run.

    The full trajectory remains a JSONL on disk; this lightweight record is
    what the multi-layer memory indexes for "show me similar past tasks".
    """

    task_id: str
    user_prompt: str
    timestamp: float
    trajectory_path: str
    n_shots: int = 0
    total_revisions: int = 0
    escalations: int = 0
    converged: bool = True
    final_weighted_total: float = 0.0
    lessons_distilled: list[str] = field(default_factory=list)
    skills_distilled: list[str] = field(default_factory=list)
    embedding: Optional[NDArray] = None


@dataclass
class EditingScript:
    clips: list[CandidateClip] = field(default_factory=list)
    total_duration: float = 0.0
    output_path: Optional[Path] = None


# ─────────────────────────────────────────────────────────────
# Trajectory (E4) — clean interface for future RL / reward model
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentStep:
    timestamp: float
    agent_name: str
    state_snapshot: dict
    action: str
    action_input: dict
    observation: dict
    reward: Optional[float] = None
