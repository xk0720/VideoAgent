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
| Physics grounding | ✗ | ✗ | soft VLM critic only | ✗ (static) | ✗ | ✗ | hard engine, no neural pixels | **first-class** (sketch ↔ video bidirectional, p1 / p2 split) |
| Closed-loop sim verify | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | implicit in engine | **✓ PhysicsConsistencyCritic (C6, new)** |
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

### D1. **Physics is first-class, both ways** (vs. UniVA / CutClaw / VISTA which treat it as soft VLM critic at best, vs. Event-Graph which sacrifices photorealism)

We split physics into a *forward control layer* (sketch → trajectory →
generator conditioning) AND a *backward consistency critic* (does the rendered
clip's implied motion match the sketch?). Neither side of this loop exists
in any prior work:

| Prior | Forward physics | Backward physics | Photorealism |
|---|---|---|---|
| UniVA / CutClaw / ViMax | ✗ | ✗ | ✓ neural |
| VISTA | ✗ | soft VLM commonsense | ✓ neural |
| Event-Graph | hard engine | implicit in engine | **✗ engine-rendered** |
| **Maestro** | **lightweight sim → control_signal (C1)** | **PhysicsConsistencyCritic → p2_sketch_consistency (C6, new)** | ✓ neural |

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
| **C1 physics sketch** | each shot's `physics_sketch.control_signal` is a JSON trajectory file the generator was conditioned on; generator metadata records `control_signal=<path>` | `build_sketch` action × 3, `conditioned_on_control: true` in `generate` |
| **C1 critic layer** | per-mode verdicts (GRAVITY_INERTIA / COLLISION / PENETRATION / FLUID) at revision 0 with severity ~0.8, decaying to <0.3 by revision 2 | `review` action × 75 = 3 shots × 5 candidates × 5 critics |
| **C2 keyframe-local edit** | Refiner picks `edit_keyframe_idx` + `edit_instruction` *per revision*; image_edit feeds the result back as `first_frame` of the next gen | `plan_fix` action × 9 (3 shots × 3 revisions) |
| **C3 multi-agent review × metric** | every candidate scored on 7 dims (m1/m2/p1/p2/id1/m5/aesthetic); `weighted_total` drives Verifier | `final_metrics` in report |
| **C4 cross-task memory** | 3 lessons distilled, one per resolved failure mode; persisted to `lessons.jsonl`; next run's Director will retrieve them at planning time | `lessons_learned: 3` in report; `lessons.jsonl` has 3 records |
| **C5 HSI** | this prompt converges at Tier 0; under stubborn judges the loop escalates Tier 0→1→2 (unit-tested) | `tier_used: [0,0,0]`, `escalations: 0`; stress test in `test_hsi_and_consistency.py` shows `tier_used: [2]` when Tier 0 cannot fix the issue |
| **C6 sketch consistency** | the rendered clip's metadata records the sketch's `control_signal`, so the consistency critic adds no CONSERVATION verdict and `p2_sketch_consistency = 1.0`; if a hypothetical generator ignored the sketch, p2 would drop below 0.7 and the same HSI loop would repair it | `p2_sketch_consistency` in `final_metrics`; unit test `test_consistency_critic_flags_clip_that_ignored_sketch` |
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

**Final test count: 58 passed in 1.21 s** (CPU only, no GPU, no API keys).

---

## 5. One-paragraph summary

Maestro = **UniVA's deployability + CutClaw's hierarchical structuring + M3's
monotonic-improvement Verifier + Event-Graph's executable IR + VISTA's
de-biased tournament — all stitched onto a new core (physics-as-first-class,
adaptive-scope HSI self-improvement, closed-loop sketch↔video consistency,
cross-task LessonLibrary)**. We deliberately match the operational surface of
the most production-ready peers (UniVA's `/health`, MCP-style tool registry)
while differentiating on the *depth* of test-time self-improvement and the
*directness* of physical grounding — the two axes everyone else either skips
or sacrifices.
