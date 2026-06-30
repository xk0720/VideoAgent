"""DefectReport — the review, re-expressed as LOCALIZED, actionable defects.

The brain's old failure mode was that it read a flat blob of verdicts + failed
checklist items and reached for one coarse whole-clip action. This module turns
that same review into a structured GUIDE: a sorted list of `Defect`s, each of
which names WHICH entity, WHICH frame span, HOW severe, and WHICH fix modality
(motion / presence / content) is appropriate — so the brain can target the worst
LOCALIZED problem with a LOCALIZED tool instead of rerolling everything.

Two evidence kinds become defects:

  • physics verdicts (measured/judged) → a physics Defect carrying the verdict's
    entity + frame_range + severity; its `fix_modality` is derived from the
    failure mode (motion vs. presence);
  • failed SEMANTIC checklist items → a content Defect over the whole clip
    (semantic critics give no frame localization), entity parsed from the
    question text when possible.

Nothing here is heavy: pure dataclasses + a compact `to_brain_json()` the
orchestrator drops into its prompt. Training-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..types import CandidateClip, PhysFailureMode, ShotSpec

# Physics failure mode → fix modality. "motion" defects are span-localized
# trajectory problems a flf2v/i2v re-anchor can repair; "presence" defects are
# about an object existing/persisting across the span (object permanence).
_MODE_TO_MODALITY: dict[str, str] = {
    PhysFailureMode.GRAVITY_INERTIA.value: "motion",
    PhysFailureMode.COLLISION.value: "motion",
    PhysFailureMode.CONSERVATION.value: "motion",
    PhysFailureMode.PENETRATION.value: "motion",
    PhysFailureMode.OBJECT_PERMANENCE.value: "presence",
}


def _modality_for_mode(mode: PhysFailureMode) -> str:
    """gravity_inertia/collision/conservation/penetration → "motion";
    object_permanence → "presence"; anything else (fluid/deformation/
    unexplained) defaults to "motion" (a trajectory/dynamics re-gen is the most
    plausible localized repair we can offer)."""
    return _MODE_TO_MODALITY.get(mode.value, "motion")


@dataclass
class Defect:
    """One localized, actionable defect distilled from the review."""

    kind: str                         # "physics" | "semantic"
    entity: str                       # which object (best-effort; "" if unknown)
    frame_range: tuple[int, int]
    severity: float                   # 0-1
    fix_modality: str                 # "motion" | "presence" | "content"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "entity": self.entity,
            "frame_range": list(self.frame_range),
            "severity": round(float(self.severity), 3),
            "fix_modality": self.fix_modality,
            "note": self.note,
        }


@dataclass
class DefectReport:
    """The review, localized: a severity-sorted list of `Defect`s + n_frames."""

    defects: list[Defect] = field(default_factory=list)
    n_frames: int = 0

    def __bool__(self) -> bool:
        return bool(self.defects)

    def sorted_by_severity(self) -> list[Defect]:
        return sorted(self.defects, key=lambda d: d.severity, reverse=True)

    def worst(self) -> Optional[Defect]:
        if not self.defects:
            return None
        return max(self.defects, key=lambda d: d.severity)

    def to_brain_json(self) -> list[dict]:
        """Compact, worst-first defect list for the brain's prompt."""
        return [d.to_dict() for d in self.sorted_by_severity()]


# A small vocabulary the semantic-question parser uses to recover an entity
# noun. Cheap and deterministic — we only need a hint for the brain, not NER.
def _entity_from_question(question: str, spec: ShotSpec) -> str:
    """Best-effort entity name from a semantic checklist question.

    Prefer a name from the shot's physics annotation that literally appears in
    the question; otherwise "" (the brain still gets the whole-clip content
    defect, just unbound to a single object)."""
    q = (question or "").lower()
    ann = spec.physics_annotation
    if ann is not None:
        for ent in ann.entities:
            if ent.name and ent.name.lower() in q:
                return ent.name
    return ""


def build_defect_report(
    clip: CandidateClip, spec: ShotSpec, fps: int = 8
) -> DefectReport:
    """Distill the reviewed clip into a localized, worst-first DefectReport.

    physics_verdicts → physics Defects (entity + frame_range + severity; modality
    from the mode). Failed SEMANTIC checklist items → content Defects spanning the
    whole clip (semantic critics localize to no frame range), entity parsed from
    the question when possible. Other checklist kinds (consistency/rhythm) are
    left to the existing metric machinery — they are not per-segment repairable,
    so surfacing them as localized defects would mislead the brain.
    """
    n_frames = max(1, int(round(spec.duration * fps)))
    defects: list[Defect] = []

    for v in clip.physics_verdicts:
        fr = (int(v.frame_range[0]), int(v.frame_range[1])) if v.frame_range \
            else (0, n_frames)
        defects.append(Defect(
            kind="physics",
            entity=v.entity or "",
            frame_range=fr,
            severity=float(v.severity),
            fix_modality=_modality_for_mode(v.mode),
            note=f"{v.mode.value} ({v.source})",
        ))

    for item in clip.checklist.failed_items:
        if item.kind != "semantic":
            continue
        defects.append(Defect(
            kind="semantic",
            entity=_entity_from_question(item.question, spec),
            frame_range=(0, n_frames),
            severity=0.5,   # semantic fails carry no measured severity; mid-rank
            fix_modality="content",
            note=item.question[:80],
        ))

    report = DefectReport(defects=defects, n_frames=n_frames)
    report.defects = report.sorted_by_severity()
    return report
