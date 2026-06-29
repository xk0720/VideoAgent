"""RepairRouter — the Review→Execute bridge (multi-action, verdict-routed).

UniVA leaves "review verdict → repair" to an ephemeral Act-LLM that re-decides
the repair tool every run (and effectively only re-prompts/regenerates). Maestro
turns the SAME bridge into a DETERMINISTIC, training-free routing decision over a
RICHER tool set: edit-clip / retrieve-replace / extend-clip / keyframe-edit /
regenerate-with-hint. The choice is driven by the WORST review verdict on the
clip and gated by what the configured backend + uploaded assets actually support.

Design contract (the part that keeps every existing test green):
  • Pure logic + a tiny dataclass. NO LLM call, NO I/O — a real deployment could
    swap in an LLM repair-planner behind this exact `choose` signature.
  • A chosen action is NEVER returned if its capability/asset is missing — it
    falls through to "regenerate_hint", which is always available and IS the
    current Tier-1 behaviour. So on the mock pipeline (caps={t2v,i2v}, no source
    shots) `choose` ALWAYS returns "regenerate_hint" → existing behaviour intact.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..physics.failure_modes import suggest_intervention
from ..types import CandidateClip, PhysFailureMode, ShotSpec

# Physics modes whose defect is a MOTION error living IN the rendered clip — an
# edit pass can repair these in place (preserving the good parts) rather than
# rerolling the whole shot.
_MOTION_MODES = {
    PhysFailureMode.GRAVITY_INERTIA,
    PhysFailureMode.COLLISION,
    PhysFailureMode.CONSERVATION,
    PhysFailureMode.PENETRATION,
}

# Cues (in the prompt or a failed item) that the clip is INCOMPLETE / too short
# / loses an object that should persist — an extend pass adds the missing tail.
_EXTEND_CUES = ("incomplete", "too short", "cut off", "longer", "continue", "extend")


@dataclass
class RepairAction:
    """One routed repair decision. `action` is the tool to invoke; `hint` is the
    verdict-derived instruction handed to that tool; `reason` is the audit trail."""

    action: str   # regenerate_hint | keyframe_edit | edit_clip | retrieve_replace | extend_clip
    reason: str
    hint: str


def _worst_verdict(clip: CandidateClip):
    if not clip.physics_verdicts:
        return None
    return max(clip.physics_verdicts, key=lambda v: v.severity)


def _semantic_miss(clip: CandidateClip):
    """The first failed SEMANTIC checklist item (a 'missing element' defect),
    or None. Real uploaded footage can ground exactly this kind of failure."""
    for item in clip.checklist.failed_items:
        if item.kind == "semantic":
            return item
    return None


class RepairRouter:
    """Map a review verdict → a concrete, tool-backed repair ACTION.

    Deterministic and verdict-driven; no learning. `capabilities` is the
    backend's `video_gen.capabilities()` and `has_source_shots` is whether
    AssetMemory holds any uploaded source shots to retrieve from.
    """

    def choose(
        self,
        clip: CandidateClip,
        spec: ShotSpec,
        *,
        capabilities: set[str],
        has_source_shots: bool,
    ) -> RepairAction:
        worst = _worst_verdict(clip)

        # (a) PHYSICS motion verdict + an edit-capable backend → fix the motion
        #     IN the existing clip (cheaper, preserves the good parts vs a reroll).
        if worst is not None and worst.mode in _MOTION_MODES and "edit" in capabilities:
            return RepairAction(
                action="edit_clip",
                reason=f"physics motion verdict '{worst.mode.value}' + edit capability",
                hint=worst.suggested_intervention or suggest_intervention(worst.mode),
            )

        # (b) Semantic 'missing element' failure + uploaded footage → ground the
        #     missing content in real source shots instead of re-imagining it.
        miss = _semantic_miss(clip)
        if miss is not None and has_source_shots:
            return RepairAction(
                action="retrieve_replace",
                reason="semantic miss + source shots available",
                hint=miss.fix_instruction or miss.question,
            )

        # (c) object_permanence / 'incomplete'/'too short' signal + extend-capable
        #     backend → continue the clip past its last frame to add what's missing.
        if "extend" in capabilities and self._wants_extend(clip, spec, worst):
            return RepairAction(
                action="extend_clip",
                reason="incomplete/object-permanence signal + extend capability",
                hint=(worst.suggested_intervention if worst is not None else "")
                or "continue the shot; keep all entities present and on a "
                "single continuous trajectory",
            )

        # (d) Fallback — always available; the current Tier-1 behaviour and the
        #     guaranteed degrade target when no richer tool/asset applies.
        from ..pipeline.generate_loop import _tier1_hint

        return RepairAction(
            action="regenerate_hint",
            reason="no richer tool/asset applies — degrade to prompt-regenerate",
            hint=_tier1_hint(clip),
        )

    @staticmethod
    def _wants_extend(clip: CandidateClip, spec: ShotSpec, worst) -> bool:
        if worst is not None and worst.mode == PhysFailureMode.OBJECT_PERMANENCE:
            return True
        haystacks = [spec.prompt or ""]
        haystacks += [i.fix_instruction for i in clip.checklist.failed_items]
        haystacks += [i.question for i in clip.checklist.failed_items]
        blob = " ".join(haystacks).lower()
        return any(cue in blob for cue in _EXTEND_CUES)
