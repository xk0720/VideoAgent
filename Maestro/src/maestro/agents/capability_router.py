"""CapabilityRouter — WHICH generation capability THIS shot needs (Phase-2).

The three-layer model this file lives in:

  • adapter        (HOW to call a model)        = the backend methods
                                                  (models/video_gen_backends.py)
  • provider binding (WHICH backend backs a cap) = config (models.video_gen.*)
  • **capability routing (WHICH cap THIS shot needs) = THIS module — a SKILL**

Capability routing is procedural knowledge, not static config: the right
capability for a shot depends on what the shot is trying to do and what
succeeded for similar shots before. So the decision is SKILL-DRIVEN —

  (a) PRIMARY PATH — skill reuse: when planning matched a creation skill for
      this shot AND that skill recorded a capability the backend offers, we
      return the skill's recorded (gen_capability, gen_params). "The skill
      decides which model." This closes the gap UniVA leaves: UniVA re-decides
      routing via its Act-LLM every run, ephemerally; Maestro distills the
      decision into the skill and reuses it.

  (b) COLD-START — deterministic intent heuristic: before any skill exists for
      a signature, a cheap, training-free heuristic over the shot/assets picks
      the capability. This is the BOOTSTRAP that the learned skill replaces
      over time. (A real deployment could swap this heuristic for an LLM
      director; the contract — `route(...) -> (capability, params)` — is the
      same.)

  (c) HONESTY — never claim a capability the backend lacks: the mock backend
      only offers {t2v, i2v}, so a shot that "wants" edit/flf2v DEGRADES to
      i2v (if a reference exists) else t2v, and the downgrade is REPORTED in
      the result (no silent capability claims).

Training-free, deterministic, no LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..logging_utils import get_logger
from ..types import AssetMemory, ShotSpec

log = get_logger(__name__)


@dataclass
class RouteDecision:
    """The routing verdict — recordable in the trajectory."""

    capability: str                                  # chosen cap, ALWAYS in `available`
    params: dict = field(default_factory=dict)       # capability-specific args
    source: str = "heuristic"                        # "skill" | "heuristic"
    reason: str = ""                                 # human-readable rationale
    downgraded_from: str = ""                         # cap we WANTED but couldn't serve


class CapabilityRouter:
    """Deterministic, training-free capability router (see module docstring)."""

    def route(
        self,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory],
        available: set[str],
    ) -> RouteDecision:
        """Pick (capability, params) for `spec`, choosing only among `available`.

        Order: (a) skill reuse → (b) intent heuristic → (c) downgrade to keep
        the choice inside `available`. Never returns a capability not offered
        by the backend.
        """
        available = set(available) or {"t2v"}

        # (a) PRIMARY — the matched skill's recorded routing decision wins,
        #     provided the backend can still serve it.
        skill = spec.matched_skill
        if skill is not None and getattr(skill, "gen_capability", None):
            want = skill.gen_capability
            if want in available:
                return RouteDecision(
                    capability=want,
                    params=dict(getattr(skill, "gen_params", {}) or {}),
                    source="skill",
                    reason=f"matched_skill {skill.skill_id} recorded {want}",
                )
            # Skill recorded a capability this backend lacks — fall through to
            # the heuristic, but remember the intent so we can report it.
            wanted_by_skill = want
        else:
            wanted_by_skill = ""

        # (b) COLD-START — deterministic intent heuristic over shot + assets.
        want, params, reason = self._heuristic(spec, asset_memory)
        if wanted_by_skill:
            reason = (
                f"matched_skill wanted {wanted_by_skill} (not offered) → "
                f"heuristic: {reason}"
            )

        # (c) HONESTY — keep the choice inside `available`, log any downgrade.
        if want in available:
            return RouteDecision(capability=want, params=params,
                                 source="heuristic", reason=reason)
        return self._downgrade(want, params, spec, asset_memory, available, reason)

    # ── intent heuristic (cold-start bootstrap) ──────────────────────────
    def _heuristic(
        self, spec: ShotSpec, asset_memory: Optional[AssetMemory]
    ) -> tuple[str, dict, str]:
        """Map shot/asset intent → (capability, params, reason). Priority order
        is most-specific-first; only the FIRST matching branch fires."""
        gp = spec.gen_params or {}

        # 1) An editable SOURCE video is supplied for this shot (the director
        #    marked a source clip for editing) → "edit".
        source_video = gp.get("source_video")
        if source_video:
            params = {"source_video": source_video,
                      "backend": gp.get("backend", "runway"),
                      "task": gp.get("task", "depth")}
            return "edit", params, f"source video marked for editing: {source_video}"

        # 2) Both a start AND end keyframe/reference are available → "flf2v".
        first_frame = gp.get("first_frame")
        last_frame = gp.get("last_frame")
        if first_frame and last_frame:
            return ("flf2v",
                    {"first_frame": first_frame, "last_frame": last_frame},
                    "start+end keyframes available")

        # 3) An identity/reference image anchor is available → "i2v".
        if self._has_identity_anchor(spec, asset_memory):
            return "i2v", {}, "identity/reference anchor available"

        # 4) Default → text-to-video.
        return "t2v", {}, "no source/keyframe/anchor intent — default t2v"

    @staticmethod
    def _has_identity_anchor(
        spec: ShotSpec, asset_memory: Optional[AssetMemory]
    ) -> bool:
        if not spec.identity_refs or asset_memory is None:
            return False
        return any(
            rid in asset_memory.identity_anchors for rid in spec.identity_refs
        )

    # ── downgrade (capability not offered) ───────────────────────────────
    def _downgrade(
        self,
        want: str,
        params: dict,
        spec: ShotSpec,
        asset_memory: Optional[AssetMemory],
        available: set[str],
        reason: str,
    ) -> RouteDecision:
        """Degrade an unavailable capability to one the backend offers: i2v if
        a reference anchor exists (and i2v is offered), else t2v. The downgrade
        is LOGGED and reported — never a silent capability claim."""
        if self._has_identity_anchor(spec, asset_memory) and "i2v" in available:
            fallback = "i2v"
        elif "t2v" in available:
            fallback = "t2v"
        else:
            fallback = next(iter(available))
        log.info(
            "capability downgrade shot %d: wanted '%s' (not in %s) → '%s'",
            spec.shot_idx, want, sorted(available), fallback,
        )
        return RouteDecision(
            capability=fallback,
            params={} if fallback != want else params,
            source="heuristic",
            reason=f"{reason}; downgraded {want}→{fallback} (not offered)",
            downgraded_from=want,
        )
