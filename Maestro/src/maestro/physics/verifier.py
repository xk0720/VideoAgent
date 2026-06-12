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

Decertified measurement entities are DEMOTED to the "vlm" tier (the track
cannot be trusted, but a VLM judge can still look at the pixels), and the
coverage report is rebuilt from the FINAL per-entity tiers so it reflects
what actually got verified, not the initial routing intent. The one
exception is "clip_unreadable": when the extractor cannot read the clip at
all there are no pixels for a VLM either, so the entity stays an
uncertified measurement deferral.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..types import CandidateClip, ShotSpec
from .laws import LawReport, analyze_track
from .reliability import TrackCertificate, certify
from .router import RouteDecision, route
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
        """Entities that were ROUTED to measurement but whose track failed
        certification — now demoted to "vlm" (or, for clip_unreadable, left
        as an uncertified measurement deferral). A certificate only ever
        exists for measurement-routed entities."""
        return [
            e.entity for e in self.entities
            if e.certificate is not None and not e.certificate.certified
        ]


class PhysicsFromPixelsVerifier:
    """Route → extract → certify → fit laws. Training-free, reference-free."""

    def __init__(self, extractor: Optional[BaseTrackExtractor] = None):
        self.extractor = extractor or MockTrackExtractor()

    def verify(
        self, clip: CandidateClip, spec: ShotSpec, fps: int = 8
    ) -> Optional[VerificationResult]:
        """None = nothing to verify at all (no annotation, or the clip could
        not be read AND every routed entity was measurement-tier). When the
        clip is unreadable but other tiers exist, measurement entities are
        reported as uncertified "clip_unreadable" deferrals instead."""
        annotation = spec.physics_annotation
        decisions: list[RouteDecision] = route(annotation)
        if not decisions:
            return None
        result = VerificationResult()

        measurable = [d for d in decisions if d.tier == "measurement"]
        tracked: dict = {}
        clip_unreadable = False
        if measurable:
            wanted = {d.entity for d in measurable}
            entities = [e for e in annotation.entities if e.name in wanted]
            observed = self.extractor.extract(clip, spec, entities, fps)
            if observed is None and len(measurable) == len(decisions):
                return None              # clip unreadable and nothing else to say
            clip_unreadable = observed is None
            tracked = observed or {}

        for d in decisions:
            if d.tier != "measurement":
                result.entities.append(EntityVerification(d.entity, d.tier))
                continue
            track = tracked.get(d.entity)
            if clip_unreadable:
                # The extractor could not read the clip at all — calling
                # this "too_short" would be a lie about the track. There is
                # also nothing for a VLM to look at in this degenerate path,
                # so the entity is NOT demoted: it stays an uncertified
                # measurement deferral.
                cert = TrackCertificate(False, 0.0, "clip_unreadable")
            elif track is None:
                # Clip was readable but the extractor returned no track for
                # this entity (e.g. it never found the object).
                cert = TrackCertificate(False, 0.0, "no_track")
            else:
                cert = certify(track, fps)
            report = analyze_track(d.entity, track, fps) if (
                cert.certified and track
            ) else None
            # S2 demotion: a decertified track must never produce a measured
            # verdict, but the entity is still checkable by a VLM judge —
            # route it to the fallback tier (except clip_unreadable, above).
            tier = d.tier
            if not cert.certified and cert.reason != "clip_unreadable":
                tier = "vlm"
            result.entities.append(
                EntityVerification(d.entity, tier, certificate=cert, report=report)
            )

        # Coverage from the FINAL entity tiers (post-demotion), not the
        # initial route decisions — the transparency report must be truthful.
        for e in result.entities:
            result.coverage.setdefault(e.tier, []).append(e.entity)
        return result
