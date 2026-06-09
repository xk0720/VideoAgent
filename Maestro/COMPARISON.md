# Maestro vs reference frameworks — capabilities, gaps, and the effect of our six innovations

This document situates Maestro against every agentic video framework we
borrowed *patterns* from, so an outside reader can see which axes we
deliberately *match* (operational maturity), which axes we deliberately
*differ on* (the depth of self-improvement and physics grounding), and what
concrete effect each of our six core innovations has on a generation run.

Cited works:

- **UniVA** — *Universal Video Agent* (Plan-Act + MCP tool servers).
  arXiv:[2511.08521](https://arxiv.org/abs/2511.08521) ·
  [github](https://github.com/univa-agent/univa)
- **CutClaw** — *Agentic Hours-Long Video Editing via Music Synchronization*
  (Playwriter / Editor / Reviewer agents; coarse-to-fine montage).
  arXiv:[2603.29664](https://arxiv.org/abs/2603.29664)
- **VISTA** — *Test-Time Self-Improving Video Generation Agent* (multi-critic
  whole-segment regeneration). arXiv:[2510.15831](https://arxiv.org/abs/2510.15831)
- **M3** — *High-fidelity T2I via Multi-Modal, Multi-Agent, Multi-Round Visual
  Reasoning* (Checklist + Verifier + Refiner on STATIC images).
  arXiv:[2602.06166](https://arxiv.org/abs/2602.06166)
- **VideoAgent (HKUDS)** — *all-in-one understanding & editing* with an
  agentic graph router. [github](https://github.com/HKUDS/VideoAgent)
- **ViMax (HKUDS)** — Director/Editor/Producer/Generator agents for
  idea→long-video. [github](https://github.com/HKUDS/ViMax)
- **Event-Graph** — *Text → executable event graphs* (3D engine renders).
  arXiv:[2604.10383](https://arxiv.org/abs/2604.10383)

---

## 1. Capability matrix

| Property | UniVA | CutClaw | VISTA | M3 | VideoAgent | ViMax | Event-Graph | **Maestro v0.2.2** |
|---|---|---|---|---|---|---|---|---|
| Primary task | omni video agent (gen+edit) | hours-long *editing* | T2V *generation* | T2I *image* | understanding/edit | idea→long video | text → GEST → engine | **multimodal video *generation*** |
| Training-free | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-agent | Plan + Act | Playwriter + Editor + Reviewer | Planner + 3-dim critic + rewriter | Planner + Checker + Refiner + Verifier + Editor | graph router | Director + Editor + Producer + Generator | Director + SceneBuilder + Relations | **10 agents** (incl. C5 HSI tiers) |
| Tool registry / MCP | ✓ MCP servers | ✗ (in-paper agents) | ✗ | ✗ | partial (graph) | ✗ | n/a (engine) | **✓ ToolRegistry + 4-category UniVA-style taxonomy + 9 default tools** |
| Self-improvement loop | workflow-level reflection | ✗ (one-shot edit) | ✓ whole-segment, multi-critic | ✓ image checklist + verifier | binary work-flow exec eval | TODO per repo README | ✗ (built-by-construction) | **✓ HSI: keyframe → physics-sketch → spec → escape; monotonic Verifier at every tier** |
| Physics grounding | ✗ | ✗ | soft VLM critic only | ✗ (static) | ✗ | ✗ | hard engine, no neural pixels | **first-class** (sim oracle; p1 native modes + p2 trajectory-L2; optional world-model reward) |
| Sim-oracle verification | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | implicit in engine | **✓ TrajectoryOracle + PhysicsConsistencyCritic (C6); test-time best-of-N search** |
| Cross-task memory | per-user prefs | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ LessonLibrary (C4) distills resolved failure modes** |
| Long-form structure | ✓ via prompts | **✓ hierarchical decomposition + music sync** | ✗ (single segment) | ✗ | ✓ | ✓ | ✓ via event chain | ✓ (music-driven, but length is shot-bounded; v0.3 → hours via CutClaw-style hierarchical decomposition planned) |
| Deployment surface | ✓ FastAPI on `/health` + Next.js frontend | n/a (paper) | n/a | n/a | git only | git only | git only | **✓ `maestro serve` (UniVA-compatible `/health` + `/tools` + `/generate` + `/jobs/{id}`) + Dockerfile + HEALTHCHECK** |
| Eval / benchmark | UniVA-Bench (multi-step) | own metrics | tournament + early-stop | own | exec success rate | ✗ (no paper) | physical-validity 58 % vs 20-25 % neural | tournament + 7-dim metric suite (incl. **p1 / p2 physics split**); pluggable into VideoPhy-2 / PhyGenBench / Physics-IQ |

---

## 2. Where we deliberately match a reference (operational maturity)

| We copy from | What we adopt | Why |
|---|---|---|
| **UniVA** | MCP-style **self-describing tools + ToolRegistry** + 4-category taxonomy + `Plan` / `Act` separation + `/health` + Dockerfile | Gives Maestro the *breadth* and *deployability* UniVA has, without our own NIH "tool calling" syntax. An operator deploying Maestro doesn't have to learn a new contract. |
| **CutClaw** | hierarchical multimodal decomposition for *long-form* structure (v0.3 roadmap) | When we extend beyond shot-scale, CutClaw's coarse→fine narrative+music anchor pattern is the right shape, but bolted onto our generation loop (CutClaw is purely editing). |
| **VISTA** | bidirectional **binary Tournament** judge + a multi-critic Review Board | De-biasing through pairwise swap is more robust than a single multi-way pick (ViMax). |
| **M3** | **Checklist → local edit → monotonic Verifier → escape hatch** | The cleanest "single-image" repair recipe; we lift it to keyframes. |
| **Event-Graph** | **GEST event graph as the IR** between Screenwriter and PhysicsPlanner | Executable-by-construction; lets PlanValidator catch ungroundable plans before any pixel is rendered. |

---

## 3. Where we deliberately differ from every reference

These are the moves that make Maestro a *different framework*, not "UniVA with
extra commits".

### D1. **Physics as a verification oracle for test-time search** (vs. UniVA / CutClaw / VISTA which treat physics as a soft VLM critic at best, vs. Event-Graph which sacrifices photorealism)

> **Repositioned in v0.3** (see `PHYSICS_LITERATURE_REVIEW.md`). The original
> claim — "we condition the frozen generator on a sketch trajectory" — was
> dropped: trajectory conditioning of a frozen video model is an unvalidated,
> training-heavy path and trajectories under-determine physics. The load-bearing
> mechanism is **verification + search**, not control.

A lightweight rigid-body **simulator** (ground/wall collision, restitution,
contact events) computes the **expected** motion; a track extractor recovers the
**observed** motion from the generated video; we score the deviation
(PISA-style normalized Trajectory-L2) and let **best-of-N tournament + monotonic
Verifier (+ optional world-model reward)** search for the most physical
candidate. Conditioning, if any, is a best-effort bonus — the oracle works on a
fully black-box generator (even text-only API models).

| Prior | Physics signal | Mechanism | Photorealism |
|---|---|---|---|
| UniVA / CutClaw / ViMax | ✗ | — | ✓ neural |
| VISTA | soft VLM "commonsense" score | whole-segment re-prompt | ✓ neural |
| Event-Graph | hard engine | engine renders the pixels | **✗ engine-rendered** |
| **Maestro** | **sim oracle: observed-vs-expected Trajectory-L2 (p1/p2) + optional world-model reward** | **best-of-N + monotonic Verifier test-time search; localized verdict → HSI repair** | ✓ neural |

Why this is stronger than the old "control" claim: it is **honest** (concedes
control is unreliable) yet still novel — no prior agentic video framework uses a
*physics simulator as a localized verification oracle inside a test-time search
loop*. It also composes with the V-JEPA-2 / WMReward line (a learned physics
oracle) as a second, ensemble reward.

### D2. **Adaptive scope of self-improvement (HSI)** (vs. M3 always local / VISTA always whole / CutClaw zero loop)

Self-improvement is hierarchical, scope-elastic, cost-amortized:

```
Tier 0   keyframe-local edit   ⭐    (M3-style)
Tier 1   physics-sketch replan ⭐⭐   (Maestro)
Tier 2   ShotSpec rewrite      ⭐⭐⭐  (bounded VISTA-style)
Tier 3   escape hatch          ⭐    (M3)
```

Verifier's monotonic-improvement rule applies at every tier; no other framework
escalates *scope* under critic-driven feedback.

### D3. **Cross-task capability memory** (LessonLibrary) — vs. *every* reference

VISTA/M3/ViMax/Event-Graph self-reflect within a single task. UniVA has *user
preference* memory but not *fix recipe* memory. Maestro's `LessonLibrary`
distills the failure-mode → successful-fix pairs that the loop *actually
resolved* (not naïve "expected_modes[0]") and re-injects them into future plans
via `DirectorAgent.run`, so the system gets better the more you use it.

### D4. **A second self-improvement loop at the plan stage** (PlanValidator's Critique-Correct-Verify) — vs. UniVA/VISTA/M3 (one loop, all at generation)

Maestro has *two* concentric self-improvement loops:

```
  Plan-level     Director ─► PlanValidator ─► Director.revise ─► retry
                 (cheap; catches ungroundable refs / broken GEST early)
  Generation     Generator ↔ ReviewBoard ↔ Verifier ↔ Refiner (HSI)
                 (expensive; only runs on validated plans)
```

Stopping a broken plan *before* generating saves orders of magnitude over
catching it post-hoc with a VLM critic.

### D5. **Differentiation matrix per agent** — vs. CutClaw's edit-only stance

CutClaw has Playwriter, Editor, Reviewer — all editing-side. Maestro adds:

| Maestro agent | Borrowed | Maestro twist |
|---|---|---|
| Screenwriter | ViMax / FilmAgent | music-section count drives n_shots (E1) |
| Director | ViMax | identity/style anchor binding + lesson injection (C4) |
| **PhysicsPlanner** | none | C1 sketch building + replan (C5 Tier-1) |
| **PlanValidator** | FilmAgent CCV | event-graph validation + ref grounding |
| Generator | VISTA + I2V | first-frame from sketch / image-edit (C1+C2) |
| **PhysicsCritic** | PhyGenEval | per-mode localizable verdict (C1) |
| **PhysicsConsistencyCritic** | none | closed-loop sketch verify (C6) |
| SemanticCritic | M3 Checker | checklist + fix instruction |
| ConsistencyCritic | old VideoAgent | identity/style across frames (E1) |
| RhythmCritic | old m5/m6 | beat-sync (E1) |
| Verifier | M3 | monotonic improvement gate |
| Refiner | M3 | keyframe-local edit (C2) |
| **ActAgent** | UniVA | generic tool-call executor (v0.2.2 wiring) |

---

## 4. Concrete effect of each innovation — measured on a live run

Run probe (mock pipeline, CPU, ~10 ms per shot):

```
prompt = "a ball is thrown and bounces off a wall; water pours from a cup"
```

| Innovation | Concrete effect on this run | Evidence in trajectory / report |
|---|---|---|
| **C1 physics oracle** | each shot's simulator emits a real bouncing trajectory (ground/wall collision + contact-frame events) as the oracle's expected motion; `TrajectoryOracle` compares it to observed motion | `build_sketch` action × 3; trajectory JSON has non-trivial `tracks` + `events` |
| **C1 critic layer** | per-mode verdicts (GRAVITY_INERTIA / COLLISION / PENETRATION / FLUID) at revision 0 with severity ~0.8, decaying to <0.3 by revision 2 | `review` action × 75 = 3 shots × 5 candidates × 5 critics |
| **C2 keyframe-local edit** | Refiner picks `edit_keyframe_idx` + `edit_instruction` *per revision*; image_edit feeds the result back as `first_frame` of the next gen | `plan_fix` action × 9 (3 shots × 3 revisions) |
| **C3 multi-agent review × metric** | every candidate scored on 7 dims (m1/m2/p1/p2/id1/m5/aesthetic); `weighted_total` drives Verifier | `final_metrics` in report |
| **C4 cross-task memory** | 3 lessons distilled, one per resolved failure mode; persisted to `lessons.jsonl`; next run's Director will retrieve them at planning time | `lessons_learned: 3` in report; `lessons.jsonl` has 3 records |
| **C5 HSI** | this prompt converges at Tier 0; under stubborn judges the loop escalates Tier 0→1→2 (unit-tested) | `tier_used: [0,0,0]`, `escalations: 0`; stress test in `test_hsi_and_consistency.py` shows `tier_used: [2]` when Tier 0 cannot fix the issue |
| **C6 sketch-as-oracle** | `TrajectoryOracle` recovers observed motion (mock extractor by default; `cotracker`/`tapir` on real frames) and scores observed-vs-expected Trajectory-L2; deviation > threshold → localized CONSERVATION verdict → HSI repair. Degrades silently on non-video (mock) clips; fails loudly if a real backend is misconfigured | `p2_sketch_consistency` in `final_metrics`; `tests/unit/test_track_extractor.py` (9) |
| **UniVA tool registry** | analysis tools fire during Stage 0; the trajectory now logs every tool call with category | `tool_call` × 4 (`video_probe`, `caption` × 2, `detect_objects`) — agents: `ActAgent` |
| **Server shim** | `maestro serve` → `/health` returns UniVA-shape JSON; `/generate` enqueues a job; `/jobs/{id}` polls | `test_server.py` 4/4 green |

Score history per shot (this run): `[0.42, 0.61, 0.88, 0.96]` — monotonic by
construction (Verifier rejects regressions), converged in 3 revisions.

---

## 4b. Deep-path verification (v0.2.2 final audit)

Beyond "do unit tests pass" — we verify the *promised behavior* of each
innovation holds end-to-end. All four checks are in
`tests/integration/test_deep_paths.py` and run on CPU.

| Property under test | What we actually run | Outcome |
|---|---|---|
| **C4 LessonLibrary really improves next run** | Run #1 distills a lesson to JSONL → Run #2 starts from the SAME path → Director's `expand_shotspecs` trajectory entry must show `lessons_injected > 0` | ✓ — lesson auto-retrieved and threaded into ShotSpec.injected_lessons |
| **HSI never deadlocks even when every tier fails** | Synthetic `_AlwaysFailingMLLM` rejects all candidates; HSI runs `max_revisions=3, k_retries=1`. Loop must exit, `tier_used` must contain 3 (escape), `score_history` still monotonic | ✓ — `3 in tier_used`, `escalations ≥ 2`, clip accepted |
| **Server boots under real uvicorn** (not TestClient) | Subprocess `python -m maestro.cli serve --port <free>` → poll `GET /health` → assert UniVA-compatible JSON | ✓ — boots in <5 s, `/health` returns `status=ok, service=maestro, n_tools≥7` |
| **No import-time side effects, no circular imports** | Fresh subprocess `python -c "import …"` of the full public surface (10 agents, 9 tools, server, pipeline) | ✓ — clean import in fresh interpreter |

**Test count after v0.2.2 deep paths: 58 passed in 1.21 s.**

## 4c. Internal-component audits (v0.2.2 follow-up)

Properties that don't surface as end-to-end pass/fail but silently rot the
framework if they regress. In `tests/integration/test_internal_audits.py`.

| Property under test | What we check | Outcome |
|---|---|---|
| **Embeddings deterministic + L2-normalized** | `embed_text(s) == embed_text(s)`; `‖v‖₂ = 1` for non-empty input | ✓ |
| **Embeddings safe on empty input** | `cosine(embed_text(""), v) == 0` (no NaN) | ✓ |
| **Embeddings rank semantic overlap higher** | `cos(q,near) − cos(q,far) > 0.1` for English same-domain text | ✓ — verifies LessonLibrary.retrieve actually picks relevant lessons (not just deterministic noise) |
| **Cosine in [0,1] for BoW vectors** | non-negative buckets → bounded cosine | ✓ |
| **Tournament neutralizes position bias** | a "first-arg always wins" judge → bidirectional swap yields a tie (bias cancelled), not a spurious win | ✓ — proves the VISTA-style debias actually de-biases, not just claims to |
| **Tournament picks the strongest under an honest judge** | argmax-by-weighted_total across arbitrary list positions | ✓ |
| **(mock oracle) control_signal plumbed to generator metadata** | the MOCK track extractor reads `control_signal=<path>` from the clip metadata to stand in for "did the generator follow physics"; the REAL `cotracker`/`tapir` path instead recovers motion from pixels | ✓ — mock oracle mechanism verified |
| **(mock oracle) no control_signal when sketch absent → oracle silent** | metadata says `control_signal=None`, the cue for the mock extractor to stay silent | ✓ |
| **PlanValidator CCV converges** | a plan with `id_real + id_ghost` refs runs through `plan_shots` with `max_plan_iters=3`; after Validate→Correct→Verify the bogus ref is gone and the second validate pass succeeds | ✓ — proves the plan-level self-improvement loop actually self-improves, not just runs |

**Test count after internal audits: 67 passed.**

## 4d. Cross-cutting fixes uncovered during the audit

| Bug found by the audit | Fix | Why it matters |
|---|---|---|
| `.env.example` promises `MAESTRO_SANDBOX=1` to refuse side-effecting tools, but no code honored it | `ActAgent.call` now checks the env var and refuses tools whose `spec.side_effects=True` with a clear error (read-only tools still go through) | An operator following `.env.example` was running un-sandboxed despite asking for sandbox; documented promise now matches behavior |
| `embeddings.embed_text` collapsed an entire Chinese prompt to one hash bucket (regex `\w+` swallowed all CJK as one token) — C4 LessonLibrary retrieval for CJK users degenerated to literal-string matching | Mixed tokenizer: ASCII/Latin via `[A-Za-z0-9_]+`, CJK Han / Hiragana / Katakana / Hangul **per character** | Bilingual prompts (the user's actual usage) now retrieve relevant lessons; verified by `test_embedding_cjk_per_character_tokenization` |
| `scripts/run_pipeline.py` (the demo entry script) did not surface HSI `tier_used` / `escalations`, `p2_sketch_consistency`, `tool_call` events, or the tool registry banner | rewrote stdout panel to expose every innovation; `tests/integration/test_deep_paths.py::test_pipeline_script_exposes_every_innovation_in_stdout` locks the contract | An operator running one demo command can now visually verify C1-C6 + UniVA wiring without grepping the JSON report |

**Test count after v0.2.2: 72 passed.**

## 4e. v0.3 — Memory + Skill (C7 + C8)

Adds two more innovations on top of C1-C6 without touching the model layer.
See `RESEARCH_MEMORY_SKILL.md` for the survey + design rationale.

| Innovation | What it does | Distinguishing axis vs prior work |
|---|---|---|
| **C7 PhysicsTyped SkillLibrary** | distills "compiled shot recipes" when HSI converges at Tier 0 with non-trivial initial severity (≥ 0.5); retrieves by `PhysFailureMode` signature × text cosine; couples each skill to its lessons | Voyager / SkillWeaver / SkillFoundry distil by env reward / rehearsal repeatability and retrieve by text — none use **Verifier-monotonic acceptance** as the distillation signal nor **physical-mode signature** as the retrieval key |
| **C8 Multi-Layer Memory** | 6 tiers — Working / Episodic (+ replay) / Semantic (A-MEM-extended LessonLibrary) / Procedural (= C7) / Entity (cross-run) / Preference; `MultiLayerMemory` façade with cross-tier associative query | A-MEM operates on semantic memory only; VideoMemory's Dynamic Memory Bank is per-run; Me-Agent has preference + working only — Maestro is the first to combine *all six* and tie them through a HippoRAG-style associative graph |

### v0.3 deep-path verification

| Property under test | Outcome |
|---|---|
| SkillLibrary distillation rule (Tier-0 + severity ≥ 0.5 + converged) | ✓ — `test_skill_distill_when_tier0_converges_on_nontrivial_severity` |
| Skill JSONL persistence roundtrip | ✓ |
| Skill distill is idempotent and EMA-updates `perf_score` | ✓ |
| Typed retrieval prefers physical-signature match over text-only | ✓ — direct contrast with Voyager-style retrieval |
| Skill lifecycle: evicts perf < 0.4 with uses > 5 | ✓ |
| EntityStore cross-run reuse — Day 2's "hero" gets Day 1's entity_id | ✓ |
| PreferenceStore JSON roundtrip + lazy user creation | ✓ |
| EpisodicStore appends + retrieves similar past tasks | ✓ |
| Episodic retrieval down-weights diverged (escalated) runs | ✓ |
| A-MEM bidirectional lesson linking on related-add | ✓ |
| Lesson idempotence reconfirms confidence | ✓ |
| MultiLayerMemory associative query lights up ≥ 2 tiers | ✓ |
| **E2E run distills a skill + writes episodic trace** | ✓ |
| **Round 2 of same prompt retrieves the round-1 skill (closed loop)** | ✓ |
| **Episodic store finds the prior run for a semantically-close prompt** | ✓ |

### v0.3 physics repositioning (sketch-as-oracle)

| Property under test | Outcome |
|---|---|
| Simulator: dropped ball bounces, decaying amplitude, never penetrates floor | ✓ — `test_simulator_ball_bounces_off_ground` |
| Simulator: one-sided wall rebound + contact events recorded | ✓ — `test_simulator_wall_bounce_records_event` |
| Simulator: support constraint keeps object at rest (no fall-through) | ✓ |
| `TrajectoryOracle` observed-vs-expected normalized Trajectory-L2 | ✓ — `physics/oracle.py` |
| Track-extractor factory: mock default, lazy `cotracker`/`tapir` dispatch | ✓ — `test_track_extractor.py` |
| Real extractor returns None (silent) on non-video mock clip — no torch needed | ✓ |
| Real extractor **fails loudly** on a decodable video when the model is unwired | ✓ — never emits a fake-perfect p2 |
| magic-byte sniff blocks text placeholders before any decoder runs | ✓ |

**Final test count: 113 passed in ~1.8 s** (CPU, no GPU, no API keys).
0 regressions across the v0.3 line.

---

## 5. One-paragraph summary

Maestro = **UniVA's deployability + CutClaw's hierarchical structuring + M3's
monotonic-improvement Verifier + Event-Graph's executable IR + VISTA's
de-biased tournament — all stitched onto a new core (physics-as-verification-
oracle for test-time search, adaptive-scope HSI self-improvement, cross-task
LessonLibrary, and a six-tier memory + PhysicsTyped skill library)**. We
deliberately match the operational surface of the most production-ready peers
(UniVA's `/health`, MCP-style tool registry)
while differentiating on the *depth* of test-time self-improvement and the
*directness* of physical grounding — the two axes everyone else either skips
or sacrifices.
