"""PhysicsFromPixelsVerifier — the C6 v0.4 verification stack, assembled.

    annotation ──► router (which tier can check each entity)
                       │ measurement tier
                       ▼
    clip ──► track extractor ──► reliability gate ──► law checks ──► reports
                                     (certify)         (laws.py)

Output is interpretable and localized (entity, frame range, failure mode,
severity) so the critic can emit actionable verdicts, the HSI loop can target
its repair, and the lesson/skill layers can index the failure. Tiers the
measurement path cannot cover (fluids, agentive motion → world_model;
semantic violations → vlm) are reported as EXPLICIT deferrals, never silently
skipped — partial-verification transparency is part of the contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..types import CandidateClip, ShotSpec
from .laws import LawReport, analyze_track
from .reliability import TrackCertificate, certify
from .router import RouteDecision, coverage_summary, route
from .tracks import BaseTrackExtractor, MockTrackExtractor


@dataclass
class EntityVerification:
    entity: str
    tier: str
    certificate: Optional[TrackCertificate] = None
    report: Optional[LawReport] = None       # only for certified measurement

    @property
    def measured(self) -> bool:
        return self.report is not None


@dataclass
class VerificationResult:
    entities: list[EntityVerification] = field(default_factory=list)
    coverage: dict[str, list[str]] = field(default_factory=dict)

    @property
    def measured_reports(self) -> list[LawReport]:
        return [e.report for e in self.entities if e.report is not None]

    @property
    def worst_violation(self) -> float:
        return max((r.violation for r in self.measured_reports), default=0.0)

    @property
    def uncertified(self) -> list[str]:
        return [
            e.entity for e in self.entities
            if e.tier == "measurement" and e.certificate is not None
            and not e.certificate.certified
        ]


class PhysicsFromPixelsVerifier:
    """Route → extract → certify → fit laws. Training-free, reference-free."""

    def __init__(self, extractor: Optional[BaseTrackExtractor] = None):
        self.extractor = extractor or MockTrackExtractor()

    def verify(
        self, clip: CandidateClip, spec: ShotSpec, fps: int = 8
    ) -> Optional[VerificationResult]:
        """None = nothing to verify at all (no annotation, or the clip could
        not be read for the measurement tier)."""
        annotation = spec.physics_annotation
        decisions: list[RouteDecision] = route(annotation)
        if not decisions:
            return None
        result = VerificationResult(coverage=coverage_summary(decisions))

        measurable = [d for d in decisions if d.tier == "measurement"]
        tracked: dict = {}
        if measurable:
            wanted = {d.entity for d in measurable}
            entities = [e for e in annotation.entities if e.name in wanted]
            observed = self.extractor.extract(clip, spec, entities, fps)
            if observed is None and len(measurable) == len(decisions):
                return None              # clip unreadable and nothing else to say
            tracked = observed or {}

        for d in decisions:
            if d.tier != "measurement":
                result.entities.append(EntityVerification(d.entity, d.tier))
                continue
            track = tracked.get(d.entity)
            cert = certify(track or [], fps)
            report = analyze_track(d.entity, track, fps) if (
                cert.certified and track
            ) else None
            result.entities.append(
                EntityVerification(d.entity, d.tier, certificate=cert, report=report)
            )
        return result
