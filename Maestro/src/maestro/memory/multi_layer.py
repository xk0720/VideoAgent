"""MultiLayerMemory — C8 façade over the 6 memory tiers (v0.3).

Tiers (see RESEARCH_MEMORY_SKILL.md §4.2):
    Tier 0  Working      (transient, owned by the pipeline state — not stored)
    Tier 1  Episodic     EpisodicStore
    Tier 2  Semantic     LessonLibrary  (extended with A-MEM evolution)
    Tier 3  Procedural   SkillLibrary    (C7)
    Tier 4  Entity       EntityStore
    Tier 5  Preference   PreferenceStore

The façade gives one entry point for the pipeline ("MLM"), and offers a
HippoRAG-style associative query that lights up nodes in multiple tiers
simultaneously (lightweight v0.3: cosine-rank across tiers; full PPR over
a typed graph is reserved for v0.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..embeddings import cosine, embed_text
from ..types import PhysFailureMode, UserPreference
from .episodic_store import EpisodicStore
from .lesson_library import LessonLibrary
from .entity_store import EntityStore
from .preference_store import PreferenceStore
from .skill_admission import SkillAdmission
from .skill_library import SkillLibrary


@dataclass
class AssociativeHit:
    """One result of an associative query — anchors back to its source tier."""

    tier: str           # "episodic" | "semantic" | "procedural" | "entity"
    score: float
    payload: object     # EpisodicTrace | Lesson | Skill | PersistentEntity


@dataclass
class MultiLayerMemory:
    """6-tier memory bundle. Each tier is independently swappable / disable-able.

    Construction is via `MultiLayerMemory.open(base_dir, user_id=...)` which
    wires JSONL files at consistent paths so persistence across runs works
    out of the box.
    """

    lessons: LessonLibrary
    skills: SkillLibrary
    entities: EntityStore
    preferences: PreferenceStore
    episodes: EpisodicStore
    user_id: str = "default"
    base_dir: Optional[Path] = None
    enabled: dict[str, bool] = field(default_factory=lambda: {
        "lessons": True, "skills": True, "entities": True,
        "preferences": True, "episodes": True,
    })

    @classmethod
    def open(
        cls,
        base_dir: Optional[Path] = None,
        user_id: str = "default",
        lesson_path: Optional[Path] = None,
        enable_skills: bool = True,
        enable_entities: bool = True,
        enable_preferences: bool = True,
        enable_episodes: bool = True,
        skill_admission: Optional[SkillAdmission] = None,
        # Operator-tunable skill lifecycle constants (configs/default.yaml
        # memory.skill_* keys, threaded by pipeline/run.py). None = library
        # class defaults.
        skill_distill_severity_threshold: Optional[float] = None,
        skill_perf_ema_alpha: Optional[float] = None,
        skill_eviction_floor: Optional[float] = None,
    ) -> "MultiLayerMemory":
        base = Path(base_dir) if base_dir else None
        # Allow an explicit lesson path override for back-compat with v0.2.2
        # callers that passed `lesson_path=...` to `run_maestro`.
        lp = Path(lesson_path) if lesson_path else (
            base / "lessons.jsonl" if base else None
        )
        sp = (base / "skills.jsonl") if base else None
        ep = (base / "entities.jsonl") if base else None
        pp = (base / "preferences.json") if base else None
        ep_tr = (base / "episodes.jsonl") if base else None
        skill_kwargs = {}
        if skill_distill_severity_threshold is not None:
            skill_kwargs["distill_severity_threshold"] = skill_distill_severity_threshold
        if skill_perf_ema_alpha is not None:
            skill_kwargs["perf_ema_alpha"] = skill_perf_ema_alpha
        if skill_eviction_floor is not None:
            skill_kwargs["eviction_floor"] = skill_eviction_floor
        mlm = cls(
            lessons=LessonLibrary(lp),
            skills=SkillLibrary(sp, admission=skill_admission, **skill_kwargs),
            entities=EntityStore(ep),
            preferences=PreferenceStore(pp),
            episodes=EpisodicStore(ep_tr),
            user_id=user_id,
            base_dir=base,
            enabled={
                "lessons": True,
                "skills": enable_skills,
                "entities": enable_entities,
                "preferences": enable_preferences,
                "episodes": enable_episodes,
            },
        )
        if mlm.enabled["skills"]:
            # Unified skill abstraction: the library's own retention policy is
            # a declared MEMORY skill — the EMA/eviction constants become an
            # auditable library entry instead of implicit code behavior.
            mlm.skills.register_memory_skill(
                "skill_ema_retention",
                params={
                    "perf_ema_alpha": mlm.skills.perf_ema_alpha,
                    "eviction_floor": mlm.skills.eviction_floor,
                },
            )
        return mlm

    # ── Associative query (HippoRAG-style, lightweight v0.3) ─────────────
    def query(
        self,
        text: str,
        expected_modes: Optional[list[PhysFailureMode]] = None,
        top_k_per_tier: int = 3,
        threshold: float = 0.15,
    ) -> list[AssociativeHit]:
        """Light up nodes across tiers for the same query — Returns hits
        sorted by score across all tiers. v0.4: replace with PPR over a
        cross-tier typed graph (A-MEM links + signature edges).
        """
        hits: list[AssociativeHit] = []
        q = embed_text(text)

        # Semantic (lessons)
        if self.enabled["lessons"]:
            for l in self.lessons.retrieve(text, top_k=top_k_per_tier,
                                            threshold=threshold):
                hits.append(AssociativeHit(
                    tier="semantic", score=cosine(q, l.embedding), payload=l,
                ))
        # Procedural (skills) — only meaningful if expected_modes given.
        if self.enabled["skills"] and expected_modes:
            for s in self.skills.retrieve(text, expected_modes,
                                           top_k=top_k_per_tier):
                hits.append(AssociativeHit(
                    tier="procedural",
                    score=cosine(q, s.embedding) if s.embedding is not None else 0.0,
                    payload=s,
                ))
        # Episodic (past similar tasks)
        if self.enabled["episodes"]:
            for tr in self.episodes.similar_tasks(text, top_k=top_k_per_tier,
                                                    threshold=threshold):
                hits.append(AssociativeHit(
                    tier="episodic",
                    score=cosine(q, tr.embedding) if tr.embedding is not None else 0.0,
                    payload=tr,
                ))
        # Entities are name-keyed; we skip them here (queried directly by
        # `find_or_create` in understand.py).

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    # ── Convenience: load the active user's preference ───────────────────
    def my_preferences(self) -> UserPreference:
        return self.preferences.get(self.user_id)

    def __len__(self) -> int:
        return (len(self.lessons) + len(self.skills) + len(self.entities)
                + len(self.episodes))
