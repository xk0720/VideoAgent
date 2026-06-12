"""SkillAdmission — "skill CI" for the unified skill library (training-free).

The unified-skill claim (INNOVATION_PLAN_2026_06.md §2): the agent's ONLY
learnable substrate is a typed, verified, versioned skill library. Distillation
is an LLM summary of a successful trajectory (zero gradients); ADMISSION is the
gate between that summary and the library. The skill-library line never built
this gate for perceptual domains: AutoSkill (arXiv:2603.01145) consolidates
unverified habits straight into memory, AWM (arXiv:2409.07429) induces
workflows from cheap repeated trajectories, Voyager/SkillWeaver rely on an
execution-success oracle that video does not have — video tool calls are
expensive, stochastic, and judged perceptually (survey_skills_audio_2026_06.md
SYNTHESIS (a)). OUR increment is a three-gate, inference-time admission review:

  a) evidence gate    — the REAL signals from the HSI episode: the metric
                        suite's `weighted_total` plus the Verifier-driven
                        convergence / escalation record. The signal source is
                        the episode's metric/verifier outcome, NOT a constant
                        stamped on at distill time.
  b) regression gate  — a new skill may not LOWER the library's accepted
                        acceptance_thresholds for the same physical_signature
                        ("skill CI" regression check, the analogue of a test
                        suite that a new commit must not weaken).
  c) judge gate       — an MLLM judge (BaseMLLMClient-style; deterministic
                        mock here, real VLM plugs in behind the same
                        interface) reviews the ENTRY's internal coherence:
                        non-empty trigger cues, physical signature consistent
                        with the entities' motion classes.

Rejection means the entry is never persisted. No gradients anywhere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..types import PhysFailureMode, Skill

if TYPE_CHECKING:  # avoid a runtime cycle; SkillLibrary imports this module
    from .skill_library import SkillLibrary


# Which entity motion classes can plausibly exhibit each failure mode. A skill
# whose signature watches a mode NO entity could exhibit is incoherent (e.g. a
# FLUID signature over purely ballistic entities). PENETRATION and
# OBJECT_PERMANENCE are unconstrained: static occluders/surfaces participate.
_MODE_MOTION_COMPAT: dict[PhysFailureMode, frozenset[str]] = {
    PhysFailureMode.FLUID: frozenset({"fluid"}),
    PhysFailureMode.GRAVITY_INERTIA: frozenset({"ballistic", "rigid", "agentive"}),
    PhysFailureMode.COLLISION: frozenset({"ballistic", "rigid", "agentive"}),
    PhysFailureMode.CONSERVATION: frozenset({"ballistic", "rigid", "agentive"}),
    PhysFailureMode.DEFORMATION: frozenset({"rigid", "agentive", "fluid"}),
}


@dataclass
class AdmissionVerdict:
    """Outcome of one admission review — persisted onto the skill entry so
    every library member carries an auditable record of WHY it was admitted."""

    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)


class BaseSkillJudge(ABC):
    """Judge interface for the admission gate (c). Mirrors the
    BaseMLLMClient pattern (models/mllm.py): deterministic mock by default,
    a real VLM judge plugs in behind the same signature."""

    name: str = "skill-judge"

    @abstractmethod
    def review_entry(self, skill: Skill) -> tuple[float, list[str]]:
        """Return (coherence_score in [0,1], list of problems found)."""


class MockSkillJudge(BaseSkillJudge):
    """Deterministic entry-coherence checks — what a real VLM judge would be
    prompted to verify, computed symbolically so tests need no network/GPU."""

    COHERENCE_PENALTY = 0.5     # score deduction per problem found

    def __init__(self, name: str = "mock-skill-judge"):
        self.name = name

    def review_entry(self, skill: Skill) -> tuple[float, list[str]]:
        problems: list[str] = []
        if not skill.triggers:
            problems.append("entry has no trigger cues (unretrievable by text)")
        classes = {e.motion_class for e in skill.entities}
        if classes:
            for mode in skill.physical_signature:
                compat = _MODE_MOTION_COMPAT.get(mode)
                if compat is not None and not (classes & compat):
                    problems.append(
                        f"signature mode '{mode.value}' is inconsistent with "
                        f"the entities' motion classes {sorted(classes)}"
                    )
        score = max(0.0, 1.0 - self.COHERENCE_PENALTY * len(problems))
        return score, problems


class SkillAdmission:
    """Three-gate admission reviewer ("skill CI"). Strictly training-free:
    every check is deterministic arithmetic over episode evidence plus one
    inference-time judge call."""

    # — Gate thresholds (constants, overridable per-instance) —
    MIN_WEIGHTED_TOTAL = 0.5    # evidence gate: episode metric floor
    MAX_ESCALATIONS = 0         # evidence gate: Tier-0-only episodes qualify
    JUDGE_PASS_FLOOR = 0.75     # judge gate: even ONE coherence problem
                                # (score 1.0 - COHERENCE_PENALTY = 0.5) fails
    REGRESSION_EPS = 1e-6       # float tolerance for the bar comparison
    # — Verdict score weights (evidence + regression + judge sum to 1.0) —
    W_EVIDENCE = 0.4
    W_REGRESSION = 0.2
    W_JUDGE = 0.4

    def __init__(
        self,
        judge: Optional[BaseSkillJudge] = None,
        library: Optional["SkillLibrary"] = None,
        min_weighted_total: float = MIN_WEIGHTED_TOTAL,
        max_escalations: int = MAX_ESCALATIONS,
        judge_floor: float = JUDGE_PASS_FLOOR,
    ):
        self.judge = judge or MockSkillJudge()
        # Bound by SkillLibrary.__init__ when passed as its constructor param;
        # needed for the regression gate (current bar per signature).
        self.library = library
        self.min_weighted_total = min_weighted_total
        self.max_escalations = max_escalations
        self.judge_floor = judge_floor

    # ── gates ────────────────────────────────────────────────────────────
    def _evidence_reasons(self, evidence: dict) -> list[str]:
        """Gate (a). `evidence` carries the HSI episode outcome:
        {weighted_total, escalations, resolved_modes, converged}. Missing
        evidence fails the gate — no evidence, no admission."""
        reasons: list[str] = []
        if not bool(evidence.get("converged", False)):
            reasons.append("evidence: episode did not converge "
                           "(escape hatch may have hidden defects)")
        weighted_total = float(evidence.get("weighted_total", 0.0))
        if weighted_total < self.min_weighted_total:
            reasons.append(
                f"evidence: weighted_total {weighted_total:.3f} below "
                f"admission floor {self.min_weighted_total:.3f}"
            )
        escalations = int(evidence.get("escalations", 0))
        if escalations > self.max_escalations:
            reasons.append(
                f"evidence: {escalations} escalation(s) past Tier 0 "
                f"(max {self.max_escalations})"
            )
        return reasons

    def _library_bar(self, skill: Skill) -> dict[str, float]:
        """Current best acceptance_thresholds among creation skills with the
        SAME physical_signature. Deliberately includes a previous version of
        the same skill_id, so re-distillation is bar-monotone too."""
        if self.library is None:
            return {}
        target = set(skill.physical_signature)
        bar: dict[str, float] = {}
        for s in self.library.skills:
            if s.skill_class != "creation":
                continue
            if set(s.physical_signature) != target:
                continue
            for key, value in s.acceptance_thresholds.items():
                bar[key] = max(bar.get(key, value), value)
        return bar

    def _regression_reasons(self, skill: Skill) -> list[str]:
        """Gate (b). Only keys the candidate actually declares are compared;
        an entry that declares no threshold for a key asserts nothing and
        cannot replace the stricter incumbent at retrieval time."""
        reasons: list[str] = []
        bar = self._library_bar(skill)
        for key, best in bar.items():
            ours = skill.acceptance_thresholds.get(key)
            if ours is not None and ours + self.REGRESSION_EPS < best:
                reasons.append(
                    f"regression: '{key}' threshold {ours:.3f} would lower "
                    f"the library bar {best:.3f} for this physical signature"
                )
        return reasons

    # ── public API ───────────────────────────────────────────────────────
    def review(self, skill: Skill, evidence: Optional[dict]) -> AdmissionVerdict:
        """Run all three gates; pass only if every gate passes."""
        reasons: list[str] = []

        evidence_reasons = self._evidence_reasons(evidence or {})
        reasons += evidence_reasons

        regression_reasons = self._regression_reasons(skill)
        reasons += regression_reasons

        judge_score, problems = self.judge.review_entry(skill)
        reasons += [f"judge: {p}" for p in problems]
        judge_ok = judge_score >= self.judge_floor

        passed = (not evidence_reasons) and (not regression_reasons) and judge_ok
        score = (
            (self.W_EVIDENCE if not evidence_reasons else 0.0)
            + (self.W_REGRESSION if not regression_reasons else 0.0)
            + self.W_JUDGE * judge_score
        )
        return AdmissionVerdict(passed=passed, score=round(score, 4),
                                reasons=reasons)

    def as_record(self, verdict: AdmissionVerdict) -> dict:
        """Serialize a verdict into the Skill.admission dict shape."""
        return {
            "passed": verdict.passed,
            "judge": self.judge.name,
            "score": verdict.score,
            "reasons": list(verdict.reasons),
        }
