# CSA — Cut · Score · Arc

> A differentiated framework for long-video editing agents.
>
> Motivation lives in [`docs/CRITICAL_REVIEW.md`](./CRITICAL_REVIEW.md): 14 rounds
> of buffet-integrating SOTA papers produced a codebase whose every self-loop
> signal is downstream of a constant I wrote. The way out isn't another paper,
> it's a **change of primitive and a change of scale**.
>
> This document defines the framework. The minimum implementation is the new
> dataclasses in `src/longvideoagent/types.py` (CutEvent, ArcContext) plus the
> Arc-level judge in `src/longvideoagent/tools/metric_tool.py::arc_coherence`.
> Nothing else is added — no new agents, no new RL stages, no new mocks.

---

## 1. The four challenges (each falsifiable)

The framework targets these specific gaps in the long-video-editing-agent
literature (DIRECT, FilmAgent, MovieAgent, CineAgents, GLANCE, LongVideoAgent,
Sima 1.0, EVA). All of those operate on the *segment* primitive; all evaluate
at the *segment* scale; all treat retrieve-vs-generate as a top-down agent
decision.

### Challenge C1 · Edit and generation are coupled in API but not in objective

In DIRECT (arXiv 2409.13560) and our v0.2 design doc §5.3, the
`GenerationTool` exposes anchor-frame conditioning, but the *decision* to
generate vs retrieve is made by the agent *before* the candidate exists. The
two operations are still two separate objects. **The framework should let
the editor express "I need shot of X" and have retrieve-vs-generate fall
out as a side effect of which source can supply X.**

* Falsification: if a future implementation that unifies them under one
  intent-matching call cannot reproduce both DIRECT's retrieve-only result
  and MovieAgent's generate-only result by varying only a source-bank
  saturation parameter, the unification is broken.

### Challenge C2 · Timing is treated as a constraint, not a decision

m5 (beat-sync) in DIRECT and in our retrieval metrics is a *score* that's
computed *after* the timing is decided by Director's `rhythmic_pacing`. In
real editing, "when to cut" is the decision; "what to cut to" is conditioned
on it. **Timing should be an actively-decided variable at a scale separate
from the segment.**

* Falsification: if we cannot construct two scripts with identical
  SegmentGuidances but different cut-timing patterns and show that the
  Score-level judge prefers the rhythmically correct one, the scale
  separation isn't doing work.

### Challenge C3 · No representation of editorial intent

Across DIRECT (semantic_query string), MovieAgent (character bank),
CineAgents (rolling caption buffer), and our SegmentGuidance, "what should
this cut accomplish" is represented as either a short text string or a
post-hoc collection of metric scores. Neither captures: (a) the
protagonist's emotional state, (b) the implied off-screen geography, (c) the
intended contrast with the previous cut. **A structured editorial intent
representation is the open research problem.**

* Falsification (and honest acknowledgement): CSA does *not* solve this.
  CSA names it as the open problem the framework's `ArcContext` is a
  placeholder for. The contribution is making it a named slot rather than
  letting "semantic_query: 'a cinematic shot'" pretend to be the answer.

### Challenge C4 · No judge operates at the whole-sequence scale

Every existing judge — m1..m6, MLLM-as-judge, EnsembleRewardModel, OPD
teacher — looks at *one segment at a time*. The narrative arc, the rise and
fall of tension, the implicit promises made at minute 1 and paid off at
minute 5 — these are properties of the *whole* script, not of any segment.
**A judge at the Arc scale is missing and needs to be invented.**

* Falsification: if a two-script test (same segments reshuffled, just
  different ordering) does not produce different `arc_coherence` scores,
  the Arc judge is degenerate.

---

## 2. The framework: three scales, three loops

```
                         ╔════════════════════════════════╗
                         ║          ARC (whole)            ║
                         ║  narrative shape, promise/payoff║
                         ║  scale = entire script          ║
                         ║  judge = arc_coherence(...)     ║
                         ╚═════════════╤══════════════════╝
                                       │ constrains
                                       ▼
                         ╔════════════════════════════════╗
                         ║         SCORE (rhythm)          ║
                         ║  when to cut, beat alignment    ║
                         ║  scale = sequence of cut times  ║
                         ║  judge = score_match(...)       ║
                         ╚═════════════╤══════════════════╝
                                       │ constrains
                                       ▼
                         ╔════════════════════════════════╗
                         ║          CUT (local)            ║
                         ║  what to show at this moment    ║
                         ║  scale = a single cut event     ║
                         ║  judge = m1..m6 (existing)      ║
                         ╚════════════════════════════════╝
```

Each scale has:

1. Its own **representation** (dataclass).
2. Its own **judge** (a function/agent that scores at that scale).
3. Its own **self-loop** (could be revise locally without re-deciding higher
   levels).

The three scales **compose by constraint**, not by aggregation. The Arc
constraints the Score (the rise-and-fall pattern in narrative tension
implies a tempo curve in cut frequency). The Score constraints the Cut (a
chosen cut time + length narrows the candidate set retrievable from the
source bank).

This is the architectural difference vs current literature, which uses
"plan → execute" (top-down, executed once) rather than "constraint
propagation across scales" (iterative, scale-aware).

### Where edit + generation actually unify

At the **Cut** level, the decision is:

> "Given the constraints from Score (when, how long) and Arc (what intent),
> what's the best concrete material to put in this cut slot?"

The answer is a `CutEvent.candidate_source ∈ {retrieval, generation,
no_op}`, and which one wins is **a lookup into the unified candidate pool**
— not an agent's top-down decision. Retrieval first; if no candidate above
threshold and generation is feasible, generate with the retrieval-near
anchor; if neither, mark the cut as `no_op` and let the Arc judge see the
gap.

This is the v0.2 architecture *promised* but the v0.2 code never *enforced*:
v0.2 has `EditorAgent` decide first, then call the tool. CSA inverts —
the tools (retrieve + generate) are queried in parallel against the intent;
the cheaper-or-better wins. This is what the design doc §0 actually
described and v0.2 never implemented.

---

## 3. How CSA addresses each challenge

| Challenge | CSA's answer |
|---|---|
| C1 edit/gen coupled in API not objective | Cut-level lookup is intent-first, source-second. Retrieve and generate are two implementations of the same query, not two top-level decisions. |
| C2 timing as constraint | Score lives at a separate scale with its own judge. It can be optimised independently of what's cut to. |
| C3 no editorial intent representation | `ArcContext` is the named slot; CSA acknowledges this is the open problem and stops pretending semantic_query is the answer. |
| C4 no whole-sequence judge | `arc_coherence(script)` is the minimum implementation: a non-segment-level scorer that takes the *whole* script. Composes with existing m1..m6 (which stays for Cut-level judgement). |

---

## 4. Minimum implementation (this round)

To prove CSA isn't another paper-stack, the implementation in this round is
deliberately small:

* `src/longvideoagent/types.py` — add `CutEvent` and `ArcContext` dataclasses.
* `src/longvideoagent/tools/metric_tool.py` — add `arc_coherence(script)`
  that computes an Arc-level score from the whole `EditingScript`.
* `tests/unit/test_arc_coherence.py` — pin the C4 falsification: two scripts
  with reshuffled segments produce different arc_coherence scores.

No new agent, no new pipeline stage, no new RL module, no new docs that
"refer to a future implementation". Either the dataclasses + function carry
their weight or they get removed.

What CSA does **not** do this round (deliberately):

* Implement an Editor that takes Arc + Score + Cut and runs them as nested
  loops. That's R17 if R16 is approved.
* Build the "unified intent-driven candidate lookup" replacing
  `RetrievalTool` + `GenerationTool`. That requires real perception backends
  to be meaningful.
* Claim CSA "outperforms baseline". It doesn't, because there's no real
  baseline yet (see `docs/BASELINE_v0_2.md`).

---

## 5. Relation to literature (rigorous)

Each existing line of work occupies a specific point in the
(primitive × scale) plane. CSA adds a new row by introducing the Arc scale
as a separate slot with its own representation and judge.

| Work | Primitive | Judge scale |
|---|---|---|
| DIRECT (arXiv 2409.13560) | segment (shot beam) | segment (m1..m6) |
| FilmAgent (arXiv 2501.12909) | segment (scene) | segment (CCV per scene) |
| MovieAgent (arXiv 2412.06185 / showlab) | segment (generated shot) | segment (character consistency) |
| CineAgents | segment + rolling caption | segment + narrative-validation pass |
| GLANCE (arXiv 2604.05076) | segment + bi-loop | segment + verify |
| LongVideoAgent (arXiv 2512.20618) — same name, different task | reasoning step | episode (RL reward) |
| EVA (arXiv 2603.22918) | tool-call step | episode (BoN reward) |
| **CSA (this doc)** | **cut event** | **cut + score + arc, composed by constraint** |

The Arc-level slot is not a new judge model — it's a new *kind* of judge
(operates on the whole script, not on segments). When implemented with real
backends it could use any of:

* a long-context MLLM (Qwen3-VL 256K, InternVL3) reading the script's
  summary + sampled frames
* a learned narrative-tension RM (analogous to AgentPRM arXiv 2511.08325
  but at the script level)
* a structural-coherence heuristic (e.g. dramatic-arc fit via a few hand-
  designed features — what we do in this round's minimum implementation)

The Score-level slot, similarly, is not a new judge model — it's the
recognition that beat-alignment is a separate decision-scale. In real
backends:

* All-In-One (mir-aidj) gives us the music structure
* RL or beam-search at the cut-time grid (separate from the cut-content
  grid)
* The decision dimension is *which beat to land a cut on*, scored
  independently of what shot lands there

The Cut-level slot is what already exists in v0.2: m1..m6 + the
neighbour-anchored generation contract from design doc §5.3 / §7.2.

CSA's claim isn't novel judges — it's the **scale separation**, and the
**constraint propagation** between the three.

---

## 6. What this lets us say differently

After CSA is in place, when someone asks "what's the difference between
your framework and DIRECT?" the honest answer becomes:

> "DIRECT operates at the segment scale only. It has nothing to say about
> the rhythm of cuts as a separable decision, and nothing to say about
> whole-sequence narrative coherence. CSA adds two scales — Score and Arc —
> with their own representations and judges. The Cut scale is the same as
> DIRECT's; we don't claim improvement there. The differentiation is that
> the segment scale alone is *not enough* and that adding Score + Arc is the
> minimum addition that makes long-video editing as a discipline coherent."

That's a defensible claim. None of the previous 14 rounds gave us a
defensible claim.

---

## 7. Roadmap, no commitments

Honest staging if R16 is approved:

* **R17** — implement a Score-level scorer (`score_match`) that operates on
  the sequence of cut times. Compose it with `arc_coherence` from R16.
* **R18** — refactor EditorAgent to take constraints from Score + Arc
  judges and pass them down to the Cut-level decision. This is where the
  unified intent-driven candidate lookup goes.
* **R19** — once R17/R18 work in mock, decide whether to wire real backends
  (Qwen3-VL for arc judging, All-In-One for music, HunyuanVideo for
  generation). This is the gate where mock-pipeline numbers stop being
  fake.

If R16's two dataclasses + one function don't seem like enough work — that's
the point. The previous 14 rounds added lots of code and zero
differentiation. R16's job is to add little code and define what
differentiation looks like.

---

## Appendix · Rigorous references (verified May 2026)

| Reference | Where it's load-bearing |
|---|---|
| DIRECT (arXiv 2409.13560) — beam search + 6 metrics for music-driven retrieval | §1 C1, §5 |
| FilmAgent (HITsz-TMG, arXiv 2501.12909) — multi-agent video generation w/ CCV | §1 C1, §5 |
| MovieAgent (showlab, arXiv 2412.06185) — generation w/ character bank | §1 C1, §5 |
| CineAgents — narrative iterative planning, rolling caption buffer | §1 C3, §5 |
| GLANCE: Music-Grounded Non-Linear Video Editing (arXiv 2604.05076, 2026) | §5 |
| LongVideoAgent (arXiv 2512.20618, 2025) — long-video VQA, name clash | §5 |
| Sima 1.0 (arXiv 2604.07721, 2026) — documentary multi-agent | §5 |
| EVA (arXiv 2603.22918, 2026) — SFT+KTO+GRPO for video agents | §5 |
| AgentPRM (arXiv 2511.08325, 2025) — process RM for LLM agents | §5 |
| All-In-One (mir-aidj) — music structure analysis | §2 Score scale |
| Qwen3-VL (arXiv 2511.21631) / InternVL3 — long-context MLLMs | §5 |
| HunyuanVideo (Tencent, Dec 2024) — open T2V w/ anchor conditioning | §5, §7 |
| Our `LongVideoEditAgent_DESIGN.md` §0, §5.3, §7.2 | §1 C1, §4 |
| Our [`docs/CRITICAL_REVIEW.md`](./CRITICAL_REVIEW.md) | §1 (motivation) |
