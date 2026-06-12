"""VerificationWriteGate — "commit only what rendered" (v0.4, Angle 1).

OUR increment over the 2026 generation-memory line (survey_memory_2026_06.md
§5 gap 1): VideoMemory (arXiv:2601.03655) writes unverified LLM descriptions
into its memory bank, so write errors compound across shots; StoryMem
(arXiv:2512.19539) keyframe writes are heuristic; EntityMem (arXiv:2605.15199)
sidesteps the problem by never updating at all. Nobody runs continuous
perceptual verification AS THE WRITE POLICY. This gate closes the read–write
loop: a proposed StateTransition is committed into the entity state register
only after the accepted rendered clip is confirmed to actually show the change.

SIGNAL SOURCE (stated exactly, no fake confidence): in mock mode the
perceptual check degrades to prompt-echo + checklist — the only real signals
the mock pipeline produces are (a) the clip's accept/reject verdict,
(b) the prompt text the mock generator echoed into the clip body/keyframes,
and (c) the consistency checklist items the ReviewBoard appended. The real
backend swaps in an MLLM look at the decoded frames — same contract
(`confirm(transition, clip, spec) -> (bool, evidence)`), different evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.mllm import BaseMLLMClient
from ..types import CandidateClip, ShotSpec, StateTransition


def _tokens(text: str) -> set[str]:
    return {w.strip(".,;:!?'\"()") for w in text.lower().split()}


class VerificationWriteGate:
    """Gate between proposed and committed entity-state transitions.

    confirm() returns (confirmed, evidence_str). The evidence string is
    persisted on the transition either way, so every commit/reject decision
    in the log carries WHAT the gate saw, not just the verdict.

    Mock-honest decision rule (all three must hold):
      1. the clip was ACCEPTED by the HSI loop (`clip.accepted`) — we never
         write state from a clip that was not kept;
      2. the transition's `new` value (or its key terms) appears in the
         rendered prompt evidence (clip body / keyframes / spec.prompt) —
         the mock generator echoes the prompt into the artifact, which is
         the mock's stand-in for "visible in pixels";
      3. no consistency checklist item failed on the clip — if the identity
         reviewer flagged drift, no state write is trustworthy.

    A real MLLM client (BaseMLLMClient pattern) can be injected; if it
    exposes `confirm_state_change(transition, clip, spec)` the perceptual
    check is delegated to it (frames in, verdict + evidence out) while
    conditions 1 and 3 still apply.
    """

    def __init__(self, mllm: Optional[BaseMLLMClient] = None):
        self.mllm = mllm

    def confirm(
        self,
        transition: StateTransition,
        clip: CandidateClip,
        spec: ShotSpec,
    ) -> tuple[bool, str]:
        # 1. Only accepted renders may write memory.
        if not clip.accepted:
            return False, (
                f"clip shot{clip.shot_idx} not accepted — "
                f"unrendered proposals never write memory"
            )

        # 3. (checked before 2: it invalidates ALL evidence from this clip)
        failed_consistency = [
            i for i in clip.checklist.items
            if i.kind == "consistency" and not i.passed
        ]
        if failed_consistency:
            return False, (
                "consistency checklist failed on this clip: "
                f"'{failed_consistency[0].question}' — identity drift "
                "invalidates state evidence"
            )

        # Real path: delegate the perceptual look to the injected MLLM.
        if self.mllm is not None and hasattr(self.mllm, "confirm_state_change"):
            return self.mllm.confirm_state_change(transition, clip, spec)

        # 2. Mock perceptual signal: prompt echo in the rendered artifact.
        evidence_text = spec.prompt.lower()
        try:
            evidence_text += "\n" + Path(clip.video_path).read_text(
                encoding="utf-8", errors="ignore").lower()
        except OSError:
            pass
        toks = _tokens(evidence_text)
        terms = [w for w in transition.new.lower().split() if len(w) > 2] \
            or [transition.new.lower()]
        found = [t for t in terms if t in toks]
        if found:
            return True, (
                f"accepted clip; '{found[0]}' present in rendered prompt echo "
                f"(mock perceptual signal: clip body + spec.prompt); "
                f"consistency checklist clean"
            )
        return False, (
            f"accepted clip but '{transition.new}' absent from rendered "
            f"prompt echo (clip body + spec.prompt) — proposal contradicted "
            f"by the render; not committed"
        )
