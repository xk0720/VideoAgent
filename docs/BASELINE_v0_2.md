# v0.2 Baseline Measurement — Plan & Hybrid-Claim Test

> **Status: framework in place, numbers pending.**
> The experiment is fully defined and the runner is in
> [`scripts/measure_baseline.py`](../scripts/measure_baseline.py). To produce
> the numbers in §3 below, run the command in §4. This file then becomes the
> *with-data* baseline report (the runner emits an auto-populated
> `outputs/baseline/report.md` that you can paste in).

---

## 1. What we are measuring (and why this matters)

The whole framework's central architectural claim is:

> "剪辑过程通过 ... agent 思维链中**联合调度 retrieval（从源视频找镜头）和
> generation（用视频生成模型补素材）** 完成"
> — `LongVideoEditAgent_DESIGN.md` §0

Design doc §5.3 / §7.2 commit to a stronger version of this: **generation is
not a fallback — it is conditioned on the previous segment's end-frame,
optical-flow tail, and character anchors**, so retrieve→generate and
generate→retrieve transitions are supposed to feel continuous, not bolted-on.

For 14 rounds nobody put this to the test on real numbers. Before adding any
more agents / training stages / reward models, we run the simplest possible
falsification experiment.

## 2. Three regimes, one knob

| Regime | What it forces | How we force it |
|---|---|---|
| **R-only** | EditorAgent always picks `retrieve` | `compose.generation.fallback_threshold = -1.0` (no SegmentGuidance.retrieval_feasibility can ever fall below) |
| **G-only** | EditorAgent always picks `generate` | `compose.generation.fallback_threshold = 2.0` (every guidance feasibility falls below) |
| **Hybrid** | Routing decides per segment | `compose.generation.fallback_threshold = 0.4` (the framework default) |

Only this one knob varies between conditions. Everything else (memory,
prompts, music profile, agents, model backends) is identical. This isolates
the routing decision so that any difference in output quality is causally
attributable to it.

## 3. What we measure

### 3.1 Per-regime aggregates (the headline)

Computed by ``measure_baseline.aggregate(...)``:

* `mean_segments` per run, `mean_pct_generation`, `mean_accepted_rate`
* mean m1–m6 across all segments × all runs in the regime
* mean validator score (1–10 scale, MockReward / EnsembleReward)
* wall-clock `mean_elapsed_s`

These numbers go into the table below once `scripts/measure_baseline.py` runs.

### 3.2 The hybrid-claim probe — cross-segment boundary metrics

The hybrid claim's strongest empirical prediction lives at the *transition
between adjacent segments of different types*:

* If GenerationTool's neighbour-anchor conditioning works, then
  **R → G** and **G → R** transitions should *not* be markedly worse than
  **R → R** transitions on m3 (motion continuity) and m4 (framing).
* If they *are* markedly worse, the conditioning is decorative — generation
  ignores the retrieval anchor and the "hybrid" sells of §0 are just
  alternation, not seamless splicing.

For each run we compute the means of the next-segment m3 / m4 broken down by
4 transition types (R→R, R→G, G→R, G→G). Aggregated for the Hybrid regime,
these go in the second table below.

(Implementation note: because the per-step m3/m4 numbers used inside
RetrievalTool's beam search aren't preserved post-hoc, we use the
**next segment's** m2/m3/m4 — which by definition depend on the previous
segment's tail features — as the boundary signal. This is a proxy; it's the
strongest signal we can extract without re-running perception. The script
documents this clearly.)

### 3.3 The verdict logic (computed automatically by the script)

```
hybrid_val_delta_R = mean_validator(Hybrid) − mean_validator(R-only)
hybrid_val_delta_G = mean_validator(Hybrid) − mean_validator(G-only)
```

* both deltas > **+0.1** ⇒ ✅ Hybrid wins. Architectural bet pays off (at
  least in mock mode). Time to repeat with real LLM + real video-gen.
* both |deltas| ≤ 0.1 ⇒ ⚠️  Hybrid is statistical noise. Pipeline is too flat
  to expose the routing decision OR the routing rule is too coarse.
* either delta < **−0.1** ⇒ ❌ Hybrid is worse than a pure baseline. This is
  a falsification — look at the boundary table to find which transition type
  drags the average down.

## 4. How to run

```bash
# Default sweep: tiny_clip × 3 prompts × 3 seeds × 3 regimes = 27 runs.
# Takes a few minutes on CPU (mock backend everywhere).
python scripts/measure_baseline.py \
    --source tests/fixtures/tiny_clip.mp4 \
    --output-dir outputs/baseline \
    --prompts \
        "Make a 4-second high-energy montage" \
        "Make a calm cinematic establishing sequence" \
        "Make a fast action cut with motion continuity" \
    --seeds 0 1 2
```

Produces:

* `outputs/baseline/runs.jsonl` — one line per (regime, prompt, seed)
* `outputs/baseline/report.json` — aggregated means per regime
* `outputs/baseline/report.md` — markdown report w/ tables + verdict logic;
  paste relevant sections into §5 below

## 5. Findings (TO BE FILLED FROM `outputs/baseline/report.md`)

### 5.1 Per-regime summary

| Metric | R-only | G-only | Hybrid |
|---|---|---|---|
| # runs | _to fill_ | _to fill_ | _to fill_ |
| mean #segments | _to fill_ | _to fill_ | _to fill_ |
| %generation | _to fill_ | _to fill_ | _to fill_ |
| accepted rate | _to fill_ | _to fill_ | _to fill_ |
| m1 prompt-relevance | _to fill_ | _to fill_ | _to fill_ |
| m2 seg-consistency  | _to fill_ | _to fill_ | _to fill_ |
| m3 motion-continuity| _to fill_ | _to fill_ | _to fill_ |
| m4 framing          | _to fill_ | _to fill_ | _to fill_ |
| m5 beat-sync        | _to fill_ | _to fill_ | _to fill_ |
| m6 energy           | _to fill_ | _to fill_ | _to fill_ |
| **validator (1-10)**| _to fill_ | _to fill_ | _to fill_ |

### 5.2 Cross-segment boundary (Hybrid regime only)

| Transition | m3 | m4 |
|---|---|---|
| R → R | _to fill_ | (n/a — no R→R in our proxy) |
| R → G | _to fill_ | _to fill_ |
| G → R | _to fill_ | _to fill_ |

### 5.3 Verdict (auto-emitted)

_to fill — one of the three branches in §3.3 above, with the numeric deltas
inline. Paste from `outputs/baseline/report.md`._

## 6. What we will do next based on the verdict

| Verdict | Next move |
|---|---|
| ✅ Hybrid wins | Move to **plan A** — Editorial Critique Loop (ReviewerAgent + per-segment refinement, see §3 of the previous round notes). Critique focus = R→G / G→R boundaries (where Hybrid distinguishes itself). |
| ⚠️  Hybrid ≈ noise | Don't add a critic yet. Two sub-experiments first: (a) widen the SegmentGuidance.retrieval_feasibility distribution by tweaking how DirectorAgent estimates it, (b) repeat with a real LLM client so the routing prompt actually varies. |
| ❌ Hybrid loses | This is the most interesting case. Investigate which boundary type drags the average — almost certainly the generation-anchor signal is decorative in mock mode. Decide: (a) fix the mock GenerationTool's metric_scores to honestly degrade when neighbour-anchor is missing, (b) treat this as the v0.3 swap-in trigger for a real video-gen backend like HunyuanVideo. |

## 7. Constraints (deliberately not violated by this measurement)

* No new agents added.
* No new training stages added.
* No new reward models added.
* No changes to `src/longvideoagent/` whatsoever — only `scripts/` + `docs/`.
* `rl-integration` branch untouched; this file and the script live on `main`.

The whole point is: we already have plenty of infrastructure. What we don't
have is *numbers from running it*. This document, plus the runner script,
gives us the smallest possible probe to start filling that gap before we
add anything else.

---

## Appendix: hybrid vs alternation — why the boundary table is the key

A worst-case-aligned reading of "hybrid":
> "Hybrid" might just mean "sometimes pick retrieval, sometimes pick generation,
> and let the assembly tool ffmpeg-concatenate them blindly. Calling it
> *hybrid* is then PR — it's really alternation."

A stronger reading:
> Hybrid means GenerationTool is *constrained* by the surrounding retrieval
> material — same character ref, continuous flow at the boundary, matching
> first frame — so a viewer can't tell where the source video ended and the
> generated material began.

The framework's GenerationTool API and OmniWeaving backend choice
(`supports_conditions() = {text, first_frame, last_frame, reference_images,
flow_field}`) commit to the *stronger* reading. The R→G / G→R boundary
metrics are how we tell which reading actually obtains.
