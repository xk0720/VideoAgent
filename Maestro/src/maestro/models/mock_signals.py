"""Content-derived mock signals — read the ARTIFACT, never the revision counter.

THE PRINCIPLE
  A mock may simulate the WORLD — a generator that responds to repair
  instructions. But critics and metrics must read the ARTIFACT (the clip's
  content), never the revision counter. If a critic's verdict is
  `f(revision)`, the whole self-improve loop is a clock, not a feedback
  system: regenerating WITHOUT applying the fix would still "improve", which
  is exactly the failure mode documented in ../docs/CRITICAL_REVIEW.md
  (parent repo) §meta-error-1.

WHAT THE MOCK ARTIFACT LOOKS LIKE (models/video_gen.py MockVideoGenClient):

    MOCK VIDEO
    model=mock-video-gen
    prompt=<spec.prompt>[ | plan-fix: <hint>][ | constraints: <l1>; <l2>][ | fix: <f1> | <f2> ...]
    duration=...
    fps=...
    first_frame=<path or None>
    reference_images=<[paths] or None>
    seed=<n>

The generator (agents/generator.py) builds the effective prompt as
`spec.prompt` + `" | constraints: "`-joined injected lessons + one
`" | fix: "` clause carrying this round's repair instructions (RefinerAgent
joins several with " | "; the tier-1 hint may be " | "-joined too).
`| plan-fix:` is the tier-2 spec-level hint the Director bakes into
spec.prompt itself. The body is therefore an honest record of WHICH repairs
were actually applied to THIS clip — the only quality signal a mock
critic/metric is allowed to consume. Real backends replace these parsers
with real perception; the contract (artifact in, signal out) stays.
"""
from __future__ import annotations

from pathlib import Path

from ..types import CandidateClip

_PROMPT_KEY = "prompt="
_FIX_MARK = "fix:"
_PLAN_FIX_MARK = "plan-fix:"
_CONSTRAINTS_MARK = "constraints:"


def read_clip_body(clip: CandidateClip) -> str:
    """Tolerant artifact reader: the mock clip body, or "" when the clip is
    missing/unreadable (real binary video also lands here — real backends
    bring their own perception and never call these parsers)."""
    try:
        return Path(clip.video_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def applied_fixes(clip: CandidateClip) -> list[str]:
    """The repair instructions ACTUALLY applied to this clip, parsed from the
    artifact's `prompt=` record.

    Returns one entry per individual instruction:
      • each " | "-separated item of the `| fix:` clause (refiner/tier-1),
      • each ";"-separated lesson of the `| constraints:` clause (C4 lessons
        ARE applied repair knowledge — a clip conditioned on them is honestly
        better-informed),
      • the `| plan-fix:` tier-2 hint.
    [] for unreadable bodies or fix-free prompts. len(applied_fixes(clip)) is
    the mock's content-derived "repair depth".
    """
    body = read_clip_body(clip)
    if not body:
        return []
    for line in body.splitlines():
        if line.startswith(_PROMPT_KEY):
            prompt = line[len(_PROMPT_KEY):]
            break
    else:
        return []

    fixes: list[str] = []
    segments = prompt.split(" | ")
    in_fix_clause = False
    for seg in segments[1:]:               # segments[0] is the base prompt
        seg = seg.strip()
        if seg.startswith(_PLAN_FIX_MARK):
            in_fix_clause = False
            rest = seg[len(_PLAN_FIX_MARK):].strip()
            if rest:
                fixes.append(rest)
        elif seg.startswith(_CONSTRAINTS_MARK):
            in_fix_clause = False
            rest = seg[len(_CONSTRAINTS_MARK):]
            fixes.extend(p.strip() for p in rest.split(";") if p.strip())
        elif seg.startswith(_FIX_MARK):
            in_fix_clause = True
            rest = seg[len(_FIX_MARK):].strip()
            if rest:
                fixes.append(rest)
        elif in_fix_clause and seg:
            # continuation of a " | "-joined fix clause
            fixes.append(seg)
    return fixes


def _meta_value(clip: CandidateClip, key: str) -> str:
    body = read_clip_body(clip)
    prefix = key + "="
    for line in body.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def first_frame_anchored(clip: CandidateClip) -> bool:
    """Was this clip actually conditioned on a first-frame keyframe anchor?
    Read from the artifact's `first_frame=` record (None ⇒ unanchored)."""
    value = _meta_value(clip, "first_frame")
    return bool(value) and value != "None"


def reference_images_present(clip: CandidateClip) -> bool:
    """Was this clip actually conditioned on identity/style reference images?
    Read from the artifact's `reference_images=` record (None/[] ⇒ no)."""
    value = _meta_value(clip, "reference_images")
    return bool(value) and value not in ("None", "[]")
