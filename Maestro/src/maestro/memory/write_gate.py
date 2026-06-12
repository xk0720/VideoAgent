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
perceptual check degrades to render-artifact echo + loop status + checklist.
The evidence text is the CLIP BODY ONLY (the mock generator's render
artifact) — `spec.prompt` is deliberately NOT mixed in, because confirming a
prompt-derived transition against the prompt itself is a tautology. FULL
HONESTY: the mock generator still echoes the prompt into the clip body, so
the echo check remains prompt-shaped in mock mode; the residual REAL
discriminators the mock pipeline produces are (a) the HSI loop's
`converged` verdict, (b) the consistency-checklist outcome, and (c) the
escape-hatch skip records. The real backend swaps in an MLLM look at the
decoded frames — same contract (`confirm(transition, clip, spec, converged)
-> (bool, evidence)`), different evidence.
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

    Mock-honest decision rule (ALL must hold):
      1. the clip was ACCEPTED by the HSI loop (`clip.accepted`) — we never
         write state from a clip that was not kept;
      2. the loop CONVERGED on it (`converged` param, threaded from the HSI
         result) — `accepted` alone only means "the clip we ship", not
         "defect-free", and a non-converged render is untrusted evidence;
      3. the escape hatch left no skip records on the clip: any
         `skipped_items` entry starting with "consistency" (identity check
         waved through) or "physics:" (a physics defect dropped unfixed)
         invalidates ALL state evidence from this render;
      4. no consistency checklist item failed on the clip — if the identity
         reviewer flagged drift, no state write is trustworthy;
      5. the transition's `new` value (or its key terms) appears in the
         rendered clip body — the mock generator echoes the prompt into the
         artifact, which is the mock's stand-in for "visible in pixels"
         (see module docstring for what this does and does not prove).

    A real MLLM client (BaseMLLMClient pattern) can be injected; if it
    exposes `confirm_state_change(transition, clip, spec)` the perceptual
    check (5) is delegated to it (frames in, verdict + evidence out) while
    conditions 1-4 still apply.
    """

    def __init__(self, mllm: Optional[BaseMLLMClient] = None):
        self.mllm = mllm

    def confirm(
        self,
        transition: StateTransition,
        clip: CandidateClip,
        spec: ShotSpec,
        converged: bool = True,
    ) -> tuple[bool, str]:
        # 1. Only accepted renders may write memory.
        if not clip.accepted:
            return False, (
                f"clip shot{clip.shot_idx} not accepted — "
                f"unrendered proposals never write memory"
            )

        # 2. `accepted` = "the clip we ship"; only a CONVERGED loop outcome
        # is trusted evidence for state writes.
        if not converged:
            return False, (
                f"clip shot{clip.shot_idx} accepted but HSI loop did not "
                f"converge — non-converged renders never write memory"
            )

        # 3. Escape-hatched defects = untrusted render: a consistency skip or
        # ANY dropped physics defect invalidates all evidence from this clip.
        hatched = [
            s for s in clip.skipped_items
            if s.startswith("consistency") or s.startswith("physics:")
        ]
        if hatched:
            return False, (
                "escape hatch fired on this clip "
                f"({hatched[0]!r}{' …' if len(hatched) > 1 else ''}) — "
                "hatched defects make the render untrusted; no state write"
            )

        # 4. (checked before 5: it invalidates ALL evidence from this clip)
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

        # 5. Mock perceptual signal: echo in the rendered artifact (clip body
        # ONLY — spec.prompt is excluded; see module docstring).
        try:
            evidence_text = Path(clip.video_path).read_text(
                encoding="utf-8", errors="ignore").lower()
        except OSError:
            evidence_text = ""
        toks = _tokens(evidence_text)
        terms = [w for w in transition.new.lower().split() if len(w) > 2] \
            or [transition.new.lower()]
        found = [t for t in terms if t in toks]
        if found:
            return True, (
                f"accepted+converged clip; '{found[0]}' present in rendered "
                f"clip body (mock perceptual signal — the body echoes the "
                f"prompt, see write_gate docstring); consistency checklist "
                f"clean; no escape-hatch skips"
            )
        return False, (
            f"accepted clip but '{transition.new}' absent from rendered "
            f"clip body — proposal contradicted by the render; not committed"
        )
