"""SkillLibrary — C7 PhysicsTyped Skill Library (v0.3).

A skill is a *compiled shot recipe* — physics annotation + cinematography preset +
checklist + acceptance thresholds + lesson pointers — that the HSI Verifier
already accepted under non-trivial physics. Distinct from:

  • Voyager (executable Minecraft code, env-reward distillation)
  • SkillWeaver (web APIs, rehearsal-repeatability distillation)
  • SkillFoundry / SkillOps (no physics typing)

Maestro's four differentiators (see RESEARCH_MEMORY_SKILL.md §4.1):

  1. Physics-typed retrieval (PhysFailureMode signature × text similarity).
  2. Verifier-monotonic distillation (HSI Tier-0 convergence + non-trivial
     initial severity is what BORNS a skill).
  3. Lesson coupling — every skill carries pointers to LessonLibrary IDs.
  4. Time-axis composition is enabled via the sketch_template shape (not
     implemented here; the planner can chain skills).

Unified skill abstraction (INNOVATION_PLAN_2026_06.md §2, training-free):
the library now holds THREE skill classes ("creation" / "review" / "memory")
under one lifecycle — distill → ADMISSION (skill CI, skill_admission.py) →
retrieve → execute → evaluate (EMA) → evolve/evict. When an admission
reviewer is configured, rejected entries are never persisted (AutoSkill
arXiv:2603.01145 consolidates unverified habits; we don't), and
re-distillation bumps `version` so the library stays auditable/rollbackable.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from ..embeddings import cosine, embed_text
from ..logging_utils import get_logger
from ..types import (
    CinematographyTags,
    PhysEntity,
    PhysFailureMode,
    PhysInteraction,
    PhysicsAnnotation,
    Skill,
)
from .skill_admission import SkillAdmission

log = get_logger(__name__)

# The C6 physics verification tiers, registered as review skills by the
# pipeline (pipeline/run.py) so VerifiabilityRouter choices are recordable
# as skill usage. Mirrors physics/router.py tier names.
PHYSICS_REVIEW_TIERS = ("measurement", "world_model", "vlm")


def _stable_skill_id(name: str, signature: list[PhysFailureMode]) -> str:
    sig = ",".join(sorted(m.value for m in signature))
    h = hashlib.md5(f"{name}|{sig}".encode("utf-8")).hexdigest()
    return f"S{h[:12]}"


class SkillLibrary:
    """Persistent skill store with typed retrieval + lifecycle (SkillOps-style).

    The on-disk format is JSONL, one skill per line. Re-loading is idempotent
    via content-hashed `skill_id`. Skills track `perf_score` (EMA of accepted
    weighted_total) and `uses` for SkillOps-style aging.
    """

    # — Distillation thresholds (also configurable via configs/default.yaml) —
    DEFAULT_SEVERITY_THRESHOLD = 0.5    # initial worst verdict must be ≥ this
    PERF_EMA_ALPHA = 0.3                # rolling avg for perf_score
    AGE_DECAY = 0.95                    # per epoch when unused
    EVICTION_FLOOR = 0.4                # drop skill once perf_score falls below

    def __init__(self, path: Optional[Path] = None,
                 admission: Optional[SkillAdmission] = None):
        self.path = Path(path) if path else None
        self.skills: list[Skill] = []
        self._by_id: dict[str, Skill] = {}
        # Admission reviewer ("skill CI"). None = legacy v0.3 behavior:
        # distill inserts unconditionally. When set, distill() runs the
        # three-gate review BEFORE insertion and returns None on rejection.
        self.admission = admission
        if self.admission is not None and self.admission.library is None:
            self.admission.library = self   # regression gate needs the bar
        if self.path and self.path.exists():
            self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sig = [PhysFailureMode(m) for m in d.get("physical_signature", [])]
            entities = [
                PhysEntity(name=e["name"],
                           motion_class=e.get("motion_class", "rigid"))
                for e in d.get("entities", [])
            ]
            interactions = [PhysInteraction(**i) for i in d.get("interactions", [])]
            cinema = CinematographyTags(**d.get("cinematography_preset", {}))
            skill = Skill(
                skill_id=d["skill_id"],
                name=d["name"],
                physical_signature=sig,
                triggers=d.get("triggers", []),
                entities=entities,
                interactions=interactions,
                cinematography_preset=cinema,
                acceptance_thresholds=d.get("acceptance_thresholds", {}),
                coupled_lesson_ids=d.get("coupled_lesson_ids", []),
                embedding=embed_text(d["name"] + " " + " ".join(d.get("triggers", []))),
                perf_score=float(d.get("perf_score", 0.0)),
                uses=int(d.get("uses", 0)),
                last_used_ts=float(d.get("last_used_ts", 0.0)),
                parent_id=d.get("parent_id", ""),
                # v0.3 JSONL records lack the unified-abstraction fields;
                # they load as version-1 creation skills with no admission
                # record (grandfathered, back-compat).
                skill_class=d.get("skill_class", "creation"),
                version=int(d.get("version", 1)),
                admission=dict(d.get("admission") or {}),
            )
            self.skills.append(skill)
            self._by_id[skill.skill_id] = skill

    def _persist(self) -> None:
        """Atomically rewrite the JSONL — needed because we mutate perf_score
        / uses in-place, and append-only would diverge from the in-memory state.
        """
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for s in self.skills:
                f.write(json.dumps({
                    "skill_id": s.skill_id,
                    "name": s.name,
                    "physical_signature": [m.value for m in s.physical_signature],
                    "triggers": s.triggers,
                    "entities": [
                        {"name": e.name, "motion_class": e.motion_class}
                        for e in s.entities
                    ],
                    "interactions": [
                        {"kind": i.kind, "entities": i.entities}
                        for i in s.interactions
                    ],
                    "cinematography_preset": {
                        "shot_scale": s.cinematography_preset.shot_scale,
                        "shot_movement": s.cinematography_preset.shot_movement,
                        "shot_angle": s.cinematography_preset.shot_angle,
                        "framing": s.cinematography_preset.framing,
                    },
                    "acceptance_thresholds": s.acceptance_thresholds,
                    "coupled_lesson_ids": s.coupled_lesson_ids,
                    "perf_score": round(s.perf_score, 4),
                    "uses": s.uses,
                    "last_used_ts": s.last_used_ts,
                    "parent_id": s.parent_id,
                    "skill_class": s.skill_class,
                    "version": s.version,
                    "admission": s.admission,
                }, ensure_ascii=False) + "\n")

    # ── distillation ─────────────────────────────────────────────────────
    def should_distill(
        self,
        escalations: int,
        converged: bool,
        initial_severity_max: float,
        severity_threshold: float = DEFAULT_SEVERITY_THRESHOLD,
    ) -> bool:
        """The Maestro-specific distillation rule (§4.1 (b)).

        A skill is born when ALL of:
          • HSI never had to escalate past Tier 0 (cheap recipe worked);
          • the loop converged (no escape hatch left defects behind);
          • initial worst-verdict severity ≥ threshold (the recipe handled
            something non-trivial — we don't want to memorise "no physics
            verdict ever appeared" as a skill).
        """
        return (
            escalations == 0
            and converged
            and initial_severity_max >= severity_threshold
        )

    def distill(
        self,
        name: str,
        spec_prompt: str,
        annotation: PhysicsAnnotation,
        cinematography: CinematographyTags,
        thresholds: dict[str, float],
        coupled_lesson_ids: Optional[list[str]] = None,
        weighted_total: float = 0.0,
        skill_class: str = "creation",
        evidence: Optional[dict] = None,
    ) -> Optional[Skill]:
        """Freeze a successful HSI outcome into a Skill. Idempotent on
        (name, physical_signature) — re-distillation bumps `version` and
        reconfirms perf_score.

        When an admission reviewer is configured (constructor param), the
        entry passes "skill CI" BEFORE insertion: `evidence` carries the HSI
        episode outcome {weighted_total, escalations, resolved_modes,
        converged} (the metric/verifier signals, not constants). Rejected
        entries are NOT persisted and None is returned.
        """
        sig = list(annotation.expected_modes)
        skill_id = _stable_skill_id(name, sig)
        if skill_id in self._by_id:
            existing = self._by_id[skill_id]
            if self.admission is not None:
                # Re-run admission against the NEW evidence + thresholds. The
                # candidate keeps the same skill_id; the regression gate
                # compares against the previous version still in the library,
                # so the bar is monotone across versions.
                candidate = dataclasses.replace(
                    existing,
                    acceptance_thresholds=dict(thresholds),
                    version=existing.version + 1,
                )
                verdict = self.admission.review(candidate, evidence)
                if not verdict.passed:
                    log.info(
                        "skill admission REJECTED re-distill of %s (v%d): %s",
                        skill_id, existing.version + 1, "; ".join(verdict.reasons),
                    )
                    return None
                existing.acceptance_thresholds = dict(thresholds)
                existing.admission = self.admission.as_record(verdict)
            existing.version += 1
            # Confirm: rolling-average the new weighted_total in.
            existing.perf_score = (
                self.PERF_EMA_ALPHA * weighted_total
                + (1 - self.PERF_EMA_ALPHA) * existing.perf_score
            )
            existing.uses += 1
            existing.last_used_ts = time.time()
            self._persist()
            return existing
        skill = Skill(
            skill_id=skill_id,
            name=name,
            physical_signature=sig,
            triggers=[t for t in spec_prompt.lower().split() if len(t) > 3][:6],
            entities=[
                PhysEntity(name=e.name, motion_class=e.motion_class)
                for e in annotation.entities
            ],
            interactions=[
                PhysInteraction(kind=i.kind, entities=list(i.entities))
                for i in annotation.interactions
            ],
            cinematography_preset=CinematographyTags(
                shot_scale=cinematography.shot_scale,
                shot_movement=cinematography.shot_movement,
                shot_angle=cinematography.shot_angle,
                framing=cinematography.framing,
            ),
            acceptance_thresholds=dict(thresholds),
            coupled_lesson_ids=list(coupled_lesson_ids or []),
            embedding=embed_text(name + " " + spec_prompt),
            perf_score=weighted_total,
            uses=1,
            last_used_ts=time.time(),
            skill_class=skill_class,
        )
        if self.admission is not None:
            verdict = self.admission.review(skill, evidence)
            if not verdict.passed:
                log.info("skill admission REJECTED %s ('%s'): %s",
                         skill_id, name, "; ".join(verdict.reasons))
                return None
            skill.admission = self.admission.as_record(verdict)
        self.skills.append(skill)
        self._by_id[skill_id] = skill
        self._persist()
        return skill

    # ── registered (non-distilled) skill classes ─────────────────────────
    def _register_builtin(self, skill_class: str, name: str, tag: str,
                          params: Optional[dict[str, float]] = None) -> Skill:
        """Idempotent registration for review/memory skills. These are not
        distilled from trajectories — they are pipeline capabilities declared
        as first-class skills so their usage is recordable in the same
        lifecycle ledger; the admission record states that provenance."""
        skill_id = _stable_skill_id(f"{skill_class}::{name}", [])
        if skill_id in self._by_id:
            return self._by_id[skill_id]
        skill = Skill(
            skill_id=skill_id,
            name=name,
            physical_signature=[],          # not physics-typed-retrievable
            triggers=[tag],
            acceptance_thresholds=dict(params or {}),
            embedding=embed_text(name + " " + tag),
            last_used_ts=time.time(),
            skill_class=skill_class,
            admission={
                "passed": True,
                "judge": "registration",
                "score": 1.0,
                "reasons": [f"built-in {skill_class} skill registered by the pipeline"],
            },
        )
        self.skills.append(skill)
        self._by_id[skill_id] = skill
        self._persist()
        return skill

    def register_review_skill(self, name: str, tier: str,
                              params: Optional[dict[str, float]] = None) -> Skill:
        """Declare a verification tier (PHYSICS_REVIEW_TIERS) as a review
        skill so VerifiabilityRouter choices become skill usage records."""
        return self._register_builtin("review", name, tier, params)

    def register_memory_skill(self, name: str,
                              params: Optional[dict[str, float]] = None) -> Skill:
        """Declare a memory-management policy (write gating / retention) as a
        memory skill — MemSkill-style, auditable instead of implicit."""
        return self._register_builtin("memory", name, "memory_policy", params)

    def find_review_skill(self, tier: str) -> Optional[Skill]:
        for s in self.skills:
            if s.skill_class == "review" and tier in s.triggers:
                return s
        return None

    def by_class(self, skill_class: str) -> list[Skill]:
        return [s for s in self.skills if s.skill_class == skill_class]

    # ── retrieval ────────────────────────────────────────────────────────
    def retrieve(
        self,
        prompt: str,
        expected_modes: list[PhysFailureMode],
        top_k: int = 1,
        min_signature_overlap: int = 1,
    ) -> list[Skill]:
        """Physics-typed retrieval (§4.1 (a)).

        Score = signature_overlap × (0.6 + 0.4 × text_cosine × confidence_proxy).
        The signature term DOMINATES so a physics-mode-matching skill beats a
        text-only-similar skill — the differentiator from Voyager/SkillWeaver.
        Skills with signature_overlap < min_signature_overlap are filtered out.
        """
        if not self.skills:
            return []
        q = embed_text(prompt)
        target = set(expected_modes)
        scored = []
        for s in self.skills:
            if s.skill_class != "creation":
                continue   # review/memory skills are not shot recipes
            overlap = len(set(s.physical_signature) & target)
            if overlap < min_signature_overlap:
                continue
            text_sim = cosine(q, s.embedding) if s.embedding is not None else 0.0
            # perf_score is already in [0,1] for normal weighted_total; clip.
            perf = max(0.0, min(1.0, s.perf_score))
            score = overlap * (0.6 + 0.4 * (0.5 * text_sim + 0.5 * perf))
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [s for _, s in scored[:top_k]]
        # Promote (uses++ + ts touch) on retrieval; lifecycle will apply
        # AGE_DECAY only to skills NOT touched in this epoch.
        for s in hits:
            s.uses += 1
            s.last_used_ts = time.time()
        if hits:
            self._persist()
        return hits

    def mark_used(self, skill_id: str) -> None:
        """Record one execution of a skill (uses++ + ts touch) without a
        retrieval — how review/memory skill usage is ledgered."""
        s = self._by_id.get(skill_id)
        if s is None:
            return
        s.uses += 1
        s.last_used_ts = time.time()
        self._persist()

    def record_outcome(self, skill_id: str, weighted_total: float) -> None:
        """Update perf_score after a downstream HSI run that used this skill."""
        s = self._by_id.get(skill_id)
        if s is None:
            return
        s.perf_score = (
            self.PERF_EMA_ALPHA * weighted_total
            + (1 - self.PERF_EMA_ALPHA) * s.perf_score
        )
        self._persist()

    def age_and_evict(self, now: Optional[float] = None,
                      idle_seconds: float = 60.0 * 60.0 * 24.0 * 90.0) -> int:
        """SkillOps lifecycle (§4.1 lifecycle). Returns # evicted.

        Strategy:
          • decay perf_score by AGE_DECAY for skills not used in this run;
          • evict skills with perf_score < EVICTION_FLOOR AND uses > 5,
            OR last_used_ts older than idle_seconds.
        """
        now = now if now is not None else time.time()
        kept: list[Skill] = []
        evicted = 0
        for s in self.skills:
            if now - s.last_used_ts > idle_seconds:
                evicted += 1
                continue
            # perf_score is an EMA of episode outcomes — only creation skills
            # have one; registered review/memory skills are exempt from the
            # perf floor (their perf is not measured by this signal).
            if (s.skill_class == "creation"
                    and s.uses > 5 and s.perf_score < self.EVICTION_FLOOR):
                evicted += 1
                continue
            kept.append(s)
        if evicted:
            self.skills = kept
            self._by_id = {s.skill_id: s for s in self.skills}
            self._persist()
        return evicted

    def __len__(self) -> int:
        return len(self.skills)
