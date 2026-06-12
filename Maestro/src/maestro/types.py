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
# Physics (C1 / C6) — physics-from-pixels VERIFICATION (no sketch, no sim)
#
# v0.4 repositioning: the old "sketch → simulate → control signal" line is
# dead (a frozen video model cannot be controlled by a synthetic sketch, and
# comparing against ONE simulated rollout presumes parameters we don't know).
# What remains physical about a shot is an ANNOTATION: which entities should
# move, what CLASS of motion each implies, and which failure modes to watch.
# Verification happens reference-free, on the OBSERVED pixels (physics/laws.py).
# ─────────────────────────────────────────────────────────────
class PhysFailureMode(str, Enum):
    PENETRATION = "penetration"            # 穿模
    GRAVITY_INERTIA = "gravity_inertia"    # 重力 / 惯性
    COLLISION = "collision"
    FLUID = "fluid"
    OBJECT_PERMANENCE = "object_permanence"  # 物体恒存
    DEFORMATION = "deformation"
    CONSERVATION = "conservation"          # 守恒律 (公认最弱项)


MotionClass = Literal["ballistic", "rigid", "fluid", "agentive", "static"]


@dataclass
class PhysEntity:
    """A movable entity named in the prompt, plus the CLASS of motion the
    prompt implies. The class decides HOW the entity can be verified (which
    router tier), never how generation should be conditioned."""

    name: str
    motion_class: MotionClass = "rigid"


@dataclass
class PhysInteraction:
    kind: Literal["collision", "support", "fluid", "constraint"]
    entities: list[str] = field(default_factory=list)


@dataclass
class PhysicsAnnotation:
    """What is physically at stake in a shot — verification seeds, not control.

    Consumed by (a) the VerifiabilityRouter (which tier can check each entity),
    (b) the track extractor (which entities to seed/track), (c) C7 skill
    retrieval (expected_modes is the physical signature key). `strictness`
    tightens the law-residual threshold on HSI tier-1 replans."""

    entities: list[PhysEntity] = field(default_factory=list)
    interactions: list[PhysInteraction] = field(default_factory=list)
    expected_modes: list[PhysFailureMode] = field(default_factory=list)  # to watch
    strictness: float = 1.0          # >1.0 = tighter verification thresholds


@dataclass
class PhysicsVerdict:
    """Physics critic output: localizable -> actionable.

    `source` separates evidence kinds so the metric suite can score them on
    distinct axes: "vlm" = judged (PhysicsCritic), "law_verifier" = measured
    (PhysicsConsistencyCritic, reference-free law checks)."""

    mode: PhysFailureMode
    frame_range: tuple[int, int]
    severity: float                # 0-1, higher = worse
    suggested_intervention: str
    source: str = "vlm"


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
    physics_annotation: Optional[PhysicsAnnotation] = None
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
    """Unified skill entry — the agent's ONLY learnable substrate (training-free).

    v0.3 shipped this as the C7 PhysicsTyped creation recipe; the unified
    abstraction (INNOVATION_PLAN_2026_06.md §2) generalises it to three
    classes sharing ONE lifecycle (distill → admission → retrieve → execute
    → evaluate → evolve/evict):

      • "creation" — a compiled shot recipe an HSI Tier-0 convergence proved
        works on non-trivial physics (distinct from Voyager's executable code
        and SkillWeaver's web APIs: it is a structured plan template keyed on
        the physical failure modes it resolves);
      • "review"   — a verification capability (the C6 physics tiers:
        measurement / world_model / vlm) registered so router choices are
        recordable as skill usage;
      • "memory"   — a memory-management policy entry (MemSkill-style:
        write gating / retention), auditable instead of implicit.

    `admission` records the "skill CI" verdict (skill_admission.py) that let
    the entry into the library — AutoSkill (arXiv:2603.01145) consolidates
    unverified habits; we persist only verified, versioned entries.
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
    skill_class: str = "creation"                                # "creation" | "review" | "memory"
    version: int = 1                                             # bumped on re-distill
    admission: dict = field(default_factory=dict)                # {passed, judge, score, reasons}


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


# ─────────────────────────────────────────────────────────────
# Dual-register entity memory (v0.4 — survey_memory_2026_06.md Angles 1+2)
#
# OUR increment over the 2026 entity-memory line: EntityMem (arXiv:2605.15199)
# freezes reference features and cannot evolve state; VideoMemory
# (arXiv:2601.03655) and StoryMem (arXiv:2512.19539) update descriptors freely
# and drift. We factorize every entity into an IMMUTABLE identity register ⊕ a
# MUTABLE state register that changes only through typed, logged transitions —
# continuity becomes auditable instead of implicit.
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EntityIdentity:
    """Canonical identity register — IMMUTABLE after registration.

    Holds what never changes about an entity: its id, name, verified
    reference image paths and description. The dataclass is frozen AND the
    EntityStore exposes no API to mutate it post-`register` — this is the
    EntityMem-style anchor, minus EntityMem's inability to evolve state
    (state lives in the separate EntityState register).
    """

    entity_id: str
    name: str
    reference_paths: list[str] = field(default_factory=list)
    description: str = ""
    created_ts: float = 0.0


@dataclass
class EntityState:
    """Mutable state register — appearance/location/emotion/… as a flat dict.

    Changes ONLY through StateTransition entries applied by the EntityStore
    (committed or correction status); `version` increments once per applied
    transition so every state version is traceable to a log entry.
    """

    attributes: dict[str, str] = field(default_factory=dict)
    shot_idx: int = -1            # shot of last applied transition
    version: int = 0              # == number of applied transitions


@dataclass
class StateTransition:
    """One typed, logged state change — the audit unit of entity continuity.

    Lifecycle: "proposed" (authored at planning time) → "committed" (the
    verification gate confirmed the change in the ACCEPTED rendered clip) or
    "rejected" (gate found no evidence / clip not accepted). "correction" is
    the explicit, auditable path for when the render contradicts the
    proposal and memory must follow the pixels (Angle 1's discrepancy entry).
    `evidence` records what the gate actually saw — never left empty on a
    gated decision.
    """

    entity_id: str
    shot_idx: int
    field: str                    # which state attribute changed
    old: str
    new: str
    cause: str                    # why the change was proposed
    status: str = "proposed"      # "proposed" | "committed" | "rejected" | "correction"
    evidence: str = ""            # what the verification gate saw
    ts: float = 0.0


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
