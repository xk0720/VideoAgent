# Survey — Consolidating Heterogeneous Reviewer Feedback for a Tool-Calling Planner (July 2026)

> Raw survey, collected 2026-07-02 via WebSearch + WebFetch (12+ sources
> fetched in full). Purpose: evidence-back the design of Maestro's
> `agents/review_summarizer.py` (the review 整理员 between the critics and the
> brain) and position it for the paper. Design questions Q1–Q4 answered first.

---

## 0. TL;DR — the four design answers

- **Q1 (organize-only vs suggest fixes?)** Middle position wins: the summarizer
  should ORGANIZE + PRIORITIZE + LOCALIZE — localization is the single most
  valuable payload (LLMs correct errors well WHEN GIVEN the location but find
  them poorly; Tyen et al. ACL Findings 2024) — plus NON-BINDING fix-direction
  hints (fix CLASS, not tool call): Self-Refine's ablation shows actionable
  ("pinpoints an issue AND suggests a clear improvement") beats descriptive
  feedback. It must NOT emit the concrete tool call (judge/planner separation:
  Kamoi TACL 2024; Agentless FSE 2025).
- **Q2 (format?)** BOTH: a ranked machine-stable JSON issue list + a short NL
  rationale (≤200 words). No universally optimal format exists (Zhang et al.
  2411.10541: small models swing up to 40% by format; frontier models robust) —
  field CONTENT matters: location, severity, provenance, cross-reviewer
  agreement, turn-over-turn status. Curate/filter — Agentless: agents fed ALL
  raw feedback get confused; put the ranked summary at the context top
  (lost-in-the-middle, Liu 2307.03172).
- **Q3 (measured vs opinion?)** "Measurement outranks opinion" is SUPPORTED
  within the verifier's competence domain: reliable EXTERNAL verifiers are the
  differentiator for self-correction (Kamoi); a small trained classifier beat
  LLMs at error localization (Tyen); deterministic test/static-analysis
  feedback lifted Meta's repair agent 28.5%→43.9% solve rate (2507.18755); SLD
  treats detector boxes as ground truth the LLM plans against. Rule:
  domain-gated precedence + agreement boosting — measured dominates
  existence/count/geometry/timing; MLLM opinion dominates semantics/aesthetics;
  same entity+span from both types → merge with confidence boost; conflicts
  stay VISIBLE (independent aggregation beats discussion-consensus, which
  converges sycophantically — Meta-Judges 2504.17087).
- **Q4 (progress?)** Verbal per-issue diff status recomputed by re-running the
  SAME verifiers (Agentless regression pattern) + an explicit action ledger:
  Reflexion's episodic verbal-failure memory prevents repeating failed
  strategies; the brief carries `do_not_repeat` (action, target, failure
  reason) so the planner is TOLD, not left to infer.

---

## 1. Self-correction with structured feedback

- **Self-Refine** — arXiv:2303.17651 (NeurIPS 2023). Ablation: actionable vs
  generic vs none — Code Opt 27.5/26.0/24.8; Sentiment Reversal 43.2/31.2/0.
  Actionable = pinpoint + suggested improvement. Multi-aspect tasks need
  per-aspect scores or one dimension regresses while another is fixed.
- **Reflexion** — arXiv:2303.11366 (NeurIPS 2023). Verbal reflections of
  failures in an episodic memory buffer consulted on later trials — the
  canonical `do_not_repeat` evidence.
- **Kamoi et al., self-correction survey** — arXiv:2406.01297 (TACL 2024).
  Self-correction works essentially only with reliable EXTERNAL feedback;
  intrinsic self-critique unreliable.
- **Tyen et al.** — arXiv:2311.08516 (ACL Findings 2024). LLMs can't FIND
  errors but can FIX them given the location; a small trained classifier beat
  LLMs at locating. → localization is job #1; non-LLM locators are legitimate.
- **LLMRefine** — arXiv:2311.09336 (NAACL 2024). Location + error type +
  severity feedback → best revisions.
- Failure modes: Huang 2310.01798 (ICLR 2024, unaided self-correction
  degrades); Self-Correction Bench 2507.02778 (blind spot); sycophancy drift →
  provenance tags let the planner discount opinion-only critiques.

## 2. Multi-critic aggregation

- **ChatEval** — arXiv:2308.07201 (ICLR 2024): panels with DIVERSE roles beat
  single judges; output is a verdict, not a repair brief.
- **Meta-Judges** — arXiv:2504.17087 (2025): majority voting best (61.71%→
  77.26% precision); PANEL DISCUSSION WORST — "opinions converge over time".
  → aggregate critics independently; resolve by weight/vote, not negotiation.
- **Agent-as-a-Judge** — arXiv:2410.10934 (Meta 2024): judges should emit
  intermediate, step-localized feedback → issue-level, not video-level briefs.
- **Bias amplification** — arXiv:2505.19477 (2025): multi-agent judging can
  AMPLIFY shared biases → anchor the panel with a non-AI measured signal; mark
  opinion-only consensus differently from cross-type agreement.

## 3. Visual/video generation refinement loops (novelty check)

| System | Critics | Consolidator? | Planner action space |
|---|---|---|---|
| VISTA (2510.15831, Google 2025) | 9 MLLM judges (3 dims × normal/adversarial/meta) | YES — meta judges + Deep Thinking Prompting Agent (≤8-score trigger; prompt-vs-model root-cause attribution) | prompt rewrite ONLY |
| VideoRepair (2411.15115, CVPR 2025) | MLLM QA + Molmo/SAM grounding | no — fixed pipeline stages | 1 op (localized spatial regen) |
| SLD (2311.16090, CVPR 2024) | OWL-ViT detector (measured) | controller = consolidator+planner in one | 4 latent ops (images) |
| Idea2Img (2310.08541) / AIGVE-MACS (2507.01255) | single (M)LLM critic | no | prompt/instruction rewrite |
| UniVA (2511.08521) | none described (planner self-reflects) | no | full MCP tool space, NO reviewer layer |
| MovieAgent / Mora / Anim-Director / VideoAgent (2410.10076) | none / self-reflect / execution feedback | no | feed-forward or self-loop |

**Verdict:** no surveyed system inserts a dedicated review-consolidation agent
between HETEROGENEOUS critics (MLLM opinion + non-AI pixel-track measurement +
metrics) and a tool-calling planner with a wide repair palette. Closest to cite
and differentiate: VISTA (consolidation exists — homogeneous MLLM critics,
prompt-rewrite-only) and VideoRepair/SLD (heterogeneous signals — hard-wired
pipeline, single/tiny repair space). Maestro's combination appears novel as of
2026-07. UNVERIFIED residual: closed-source/unindexed systems.

## 4. Fusing measured signals with LLM critique (other domains)

- **Agentless** — arXiv:2407.01489 (FSE 2025): hierarchical localization →
  ±10-line windows to the fixer; regression + reproduction tests gate patches;
  documented warning that agents given ALL raw feedback amplify wrong steps →
  CURATE before the planner sees it.
- **Meta neuro-symbolic repair** — arXiv:2507.18755 (2025): deterministic
  feedback injected as short structured messages after each action; ablation
  ReAct 28.5% → +static analysis 34.1% → +tests 43.9%; representation matters
  (search/replace beats unified diff by 23 pts).
- **CRITIC** — arXiv:2305.11738 (ICLR 2024): tool-grounded critique works where
  pure self-critique fails.
- **Prompt formatting** — arXiv:2411.10541 (2024): no universal best format;
  JSON helps some tasks, hurts others → JSON for parsing stability + prose for
  salience. **Lost in the middle** — arXiv:2307.03172 (TACL 2024): ranked list
  at the top; raw dumps behind references.

## 5. What Maestro implements (`agents/review_summarizer.py`)

- Deterministic structured brief (signal honesty: every issue traces to a real
  reviewer output); optional REAL LLM only rephrases `brief_nl` (mock LLMs
  skipped; template fallback).
- Issues: merged by entity+overlapping span; evidence provenance
  (`measured` = law_verifier | `opinion`); `cross_type_confirmed` agreement →
  confidence 0.95 vs 0.9 measured-only vs 0.6 opinion-only; conflicts surfaced
  with `measured_precedence`.
- Ranking: regressed > measured-backed > severity × confidence.
- `fix_classes` = non-binding class hints mapped from fix_modality (never tool
  calls); `do_not_repeat` from verifier-rejected history; `progress`
  (fixed/new/regressed/unchanged) via stable cross-turn issue keys
  (kind|entity|modality|clip-quarter).
- Wired per turn in `generate_shot_orchestrated` before `orchestrator.decide`;
  the brain's skill file (`prompts/orchestrator.txt`) documents the brief as
  "read this first", closing the loop.
