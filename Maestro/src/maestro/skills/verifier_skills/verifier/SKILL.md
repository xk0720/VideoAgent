---
name: verifier
agent: VerifierAgent (the accept/reject gate)
description: Blind A/B judgment of refined-vs-original on native video (NEWTON-style) — per-dimension signed scores, defect-fixed probe, dimension non-regression guard. Metric gate is the fallback. The gate the brain proposes to.
---

# Verifier — the gate's contract ("brain proposes, gate disposes")

Role: after every executed repair, decide accept/reject. Its verdict is what
turns a brain decision into history (accepted → new best + a workflow step
toward distillation; rejected → a do_not_repeat entry for the summarizer).

## PRIMARY gate: blind A/B on native video (agents/verifier.py + verify_pair)

Mechanics follow NEWTON's verify_relative — relative comparison is the only
reliable use of an MLLM judge; absolute scores are noise:

1. BLIND SLOTS — candidate and current best are shuffled into "Video 1" /
   "Video 2" by a seeded coin flip; the judge never knows which is the repair.
   Scores are remapped back to candidate-perspective after parsing.
2. WHAT THE JUDGE SEES — both WHOLE videos as native inline parts + the shot
   prompt + the repair context (which defect the repair targeted, its time
   span, entity, fix hint). Condition adherence is NOT re-judged here — that
   is the reviewer's job on the single shot.
3. DIMENSIONS (signed, -10..+10, + toward Video 2): semantic, physics,
   temporal, visual — each with a one-line note, then one overall score.
   A repair that fixes one thing but breaks another dimension must show the
   damage as a negative dimension score — never averaged away.
4. DEFECT PROBE — the judge states per video whether the targeted defect is
   present → `target_fixed` (did the repair do the one thing it claimed?).
5. CONCLUSION (NEWTON-consistent accept/reject, plus our guard):
   accept iff overall score ≥ +1 (candidate STRICTLY better — conservative 0
   or uncertainty → reject) AND min(dimension) ≥ -2 — the monotonic contract,
   now dimension-aware: no accepted repair may badly regress any dimension.
6. FEEDBACK — the full verdict is attached to the candidate
   (verifier_verdict) and ledgered in result.actions; the brain's
   next-turn history shows outcome + new_total + verifier_issues (when
   non-empty). `issues` are populated only when the candidate is judged
   strictly WORSE overall (score < 0); tie rejects and dimension-guard
   rejects carry empty issues — their WHY lives in dim_scores/notes.

## FALLBACK gate (mock mode / verify_pair unavailable → loud log, then):

1. MONOTONIC METRIC RULE: accept only if weighted_total strictly rises, or
   ties with strictly fewer defects (failed non-physics items + physics
   verdicts — physics items mirror verdicts 1:1; counting both double-counts).
2. Marginal metric wins (Δ ≤ margin 0.02) must survive a bidirectional
   pairwise compare vs the current best; the judge can only VETO a marginal
   win, never rescue a metric loss.

## Invariants

- The bar NEVER moves during repair (no strictness tightening on a failing
  shot — that inverts the repair incentive); hardening runs post-acceptance,
  log-only.
- One judgment per turn; the gate never proposes tools or edits.

## What downstream reads from this gate

- `outcome` (accepted/rejected) + verdict → brain history (never repeat
  a rejected action on the same target; when score < 0 the verdict issues
  explain WHY, otherwise read dim_scores/notes).
- Accepted tool calls → the distilled repair workflow (skill_library.distill_
  repair) once the episode converges.
