---
name: verifier
agent: VerifierAgent (the monotonic gate)
description: Accept a candidate only if STRICTLY better than the current best; blind pairwise confirmation for marginal wins (veto-only). The gate the brain proposes to.
---

# Verifier — the gate's contract ("brain proposes, gate disposes")

Role: after every executed repair, decide accept/reject. Its verdict is what
turns a brain decision into history (accepted → new best + a workflow step
toward distillation; rejected → a do_not_repeat entry for the summarizer).

## Rules (enforced by code — agents/verifier.py)

1. MONOTONIC HARD RULE: accept only if weighted_total strictly rises, or ties
   with strictly fewer defects (failed non-physics items + physics verdicts —
   physics items mirror verdicts 1:1, so counting both would double-count).
2. BLIND PAIRWISE CONFIRMATION (NEWTON-style, when a judge MLLM is wired):
   a MARGINAL metric win (Δ ≤ margin, default 0.02 — VLM absolute-score noise)
   must also survive a bidirectional pixel comparison vs the current best:
   compare(candidate, best) and compare(best, candidate), order-debiased.
   The judge can only VETO a marginal win. It can NEVER rescue a metric loss,
   and it is NEVER consulted on a decisive win (no wasted VLM calls).
3. The bar NEVER moves during repair (no strictness tightening on a failing
   shot — that inverts the repair incentive); hardening runs post-acceptance,
   log-only.

## What downstream reads from this gate

- `outcome` (accepted/rejected) + `new_total` → brain history (never repeat a
  rejected action on the same target).
- Accepted tool calls → the distilled repair workflow (skill_library.distill_
  repair) once the episode converges.
