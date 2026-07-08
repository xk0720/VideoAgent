---
name: review_summarizer
agent: ReviewSummarizerAgent (the review 整理员)
description: Consolidate ALL reviewer outputs (measured + opinion + metrics) into ONE ranked brief for the brain — merge, rank, surface conflicts, track progress; suggest fix CLASSES, never tools.
---

# Review Summarizer — consolidation rules + the LLM polish instruction

Role: sit between the heterogeneous reviewers and the brain. Inputs each turn:
measured physics verdicts (law_verifier), VLM verdicts (vlm), failed semantic
checklist items, metric scores, the localized DefectReport, the verifier
accept/reject history, and last turn's issues. Output: the `review_brief`
dict the brain reads FIRST.

## Consolidation rules (enforced by code — agents/review_summarizer.py)

1. ORGANIZE + PRIORITIZE + LOCALIZE. Suggest fix CLASSES (localized_regen,
   edit_in_place, keyframe_edit, depth_edit, style_edit, extend, regenerate),
   NEVER a concrete tool call — tool choice is the brain's job.
2. PROVENANCE on every issue: measured | opinion. Measurement outranks opinion
   in its domain (motion / existence / timing).
3. Same entity + overlapping span from BOTH source types → ONE issue,
   agreement=cross_type_confirmed, confidence 0.95 (vs 0.9 measured-only,
   0.6 opinion-only).
4. CONFLICTS are surfaced as first-class entries (resolution:
   measured_precedence) — never silently deduped.
5. RANKING: regressed > measured-backed > severity × confidence.
6. PROGRESS via stable issue keys (kind|entity|modality|clip-quarter):
   fixed / new / regressed / unchanged, plus a do_not_repeat ledger from
   verifier-REJECTED actions (hard constraints for the brain).
7. SIGNAL HONESTY: every issue traces to an actual reviewer output; the
   structured brief is deterministic. The LLM below only REPHRASES prose.

## LLM polish instruction (loaded by _polish as the prompt preamble)

Rewrite the review brief below as 2-4 clear sentences for a repair planner.
Use ONLY the facts given — no new issues, numbers, or guesses. Keep entity
names and frame ranges verbatim. Lead with the worst problem; end with the
one thing the planner must not repeat, if any.
