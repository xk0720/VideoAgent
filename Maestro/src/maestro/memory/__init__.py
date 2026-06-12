"""Maestro memory package (v0.3 — Multi-Layer Memory).

  Tier 1  Episodic    EpisodicStore
  Tier 2  Semantic    LessonLibrary  (extended with A-MEM evolution)
  Tier 3  Procedural  SkillLibrary    ← C7 (v0.3)
  Tier 4  Entity      EntityStore     ← C8 (v0.3)
  Tier 5  Preference  PreferenceStore ← C8 (v0.3)

See `RESEARCH_MEMORY_SKILL.md` for full design rationale and citations.
"""
from .lesson_library import LessonLibrary
from .skill_admission import AdmissionVerdict, MockSkillJudge, SkillAdmission
from .skill_library import SkillLibrary
from .entity_store import EntityStore
from .preference_store import PreferenceStore
from .episodic_store import EpisodicStore
from .multi_layer import AssociativeHit, MultiLayerMemory

__all__ = [
    "LessonLibrary",
    "SkillLibrary",
    "SkillAdmission",
    "AdmissionVerdict",
    "MockSkillJudge",
    "EntityStore",
    "PreferenceStore",
    "EpisodicStore",
    "MultiLayerMemory",
    "AssociativeHit",
]
