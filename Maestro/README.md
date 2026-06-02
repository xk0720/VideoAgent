# Maestro

**Training-free, self-improving, physically-grounded agentic video generation.**

You give an instruction plus optional multimodal materials (video / image / music).
Maestro plans with multiple agents, generates each shot, reviews it with a board of
critics against a quantitative metric suite, and **locally repairs failing keyframes
in a monotonic self-improvement loop** — with **physics treated as a first-class
citizen**.

- **v0.1 (now):** a mock-first scaffold. The full control flow, self-improvement
  loop, physics module, planning validation, and logging run **end-to-end on CPU
  with no API keys**. The `.mp4` it writes is a *placeholder* (see "Why no pixels
  yet" below).
- **v0.2 (next):** swap the mock model wrappers for real models behind the same
  interfaces to get real pixels. This README tells you exactly how.

Design rationale & differentiation vs UniVA / VideoAgent / ViMax / VISTA / M3 /
event-graph: see `REPORT_AND_INSTRUCTIONS.md`. End-to-end data flow & config: see
`DATAFLOW.md`. Incremental modules & citations: see `IMPROVEMENTS.md`.

---

## Six core innovations

1. **Physics as a first-class citizen (C1)** — a *sketch layer* (lightweight
   simulation → trajectory/control signal that conditions the generator) plus a
   *critic layer* that localizes failures by mode (penetration / gravity /
   collision / fluid / object-permanence / deformation / conservation) to specific
   frames, each mapped to an executable fix.
2. **Keyframe-level local self-improvement (C2)** — M3's "checklist → local edit →
   monotonic Verifier → escape hatch", extended from images to video.
3. **Self-loop = multi-agent review × metric suite (C3)** — a `ReviewBoard` of
   critics drives a quantitative, non-black-box loop.
4. **Cross-task experience memory (C4)** — a `LessonLibrary` distills failures +
   fixes and injects them into future plans ("gets better the more you use it").
5. **Hierarchical Self-Improvement (C5, v0.2.1 NEW)** — adaptive *scope*
   escalation across three tiers. The loop tries the cheapest repair first and
   only widens scope when needed:
   - **Tier 0** local keyframe edit (M3-style)
   - **Tier 1** rebuild the physics sketch with stricter constraints
     (slower trajectories → easier target for the generator)
   - **Tier 2** Director rewrites the `ShotSpec` (cinematography + prompt) —
     bounded VISTA-style replan
   - **Tier 3** escape hatch (drop the worst remaining defect)

   Verifier's monotonic-improvement rule applies at every tier; we never accept
   a regression. After any acceptance the next revision restarts at Tier 0 —
   cost-amortized adaptive scope. **This fills the gap between VISTA (always
   whole-segment) and M3 (always local patch).**
6. **Closed-loop Sketch↔Video consistency (C6, v0.2.1 NEW)** — a dedicated
   `PhysicsConsistencyCritic` cross-checks that the rendered clip actually
   *followed* the sketch the generator was conditioned on. Divergences surface
   as a `CONSERVATION`-mode verdict and feed the same self-improvement loop.
   This makes the physics layer **bidirectional** — sim is no longer just an
   input but also a verification reference. Reported as a separate
   `p2_sketch_consistency` metric.

Plus: GEST-style event-graph IR, a plan-level Validate→Correct loop, VISTA
bidirectional tournament selection, and asset-retrieval grounding (see
`IMPROVEMENTS.md`).

---

## Architecture (data flow)

```
Inputs (prompt + optional video/image/music)
  └─ Stage 0  Understand   → AssetMemory (shots / identities / styles / music)
  └─ Stage 1  Plan          → Screenwriter → Director → PhysicsPlanner
                              + PlanValidator (Validate→Correct loop)  → ShotSpec[]
  └─ Stage 2  Generate + Hierarchical Self-Improve Loop (HSI, per shot)
                Generator → Tournament → ReviewBoard(5 critics) → Verifier
                if not accepted →  Tier 0 keyframe edit (Refiner)
                              →    Tier 1 physics-sketch replan (PhysicsPlanner)
                              →    Tier 2 spec rewrite (Director.refine_spec)
                              →    Tier 3 escape hatch  ↺
                LessonLibrary distills the actually-resolved failure mode
  └─ Stage 3  Assemble (ffmpeg, graceful fallback) → demo.mp4 + report + trajectory
```

### Repo layout

```
Maestro/
├── README.md  REPORT_AND_INSTRUCTIONS.md  DATAFLOW.md  IMPROVEMENTS.md
├── pyproject.toml  requirements.txt  .env.example
├── configs/
│   └── default.yaml                 # all knobs; override with --config
├── scripts/run_pipeline.py          # end-to-end CLI entry
├── src/maestro/
│   ├── types.py                     # all dataclasses / enums (incl. EventGraph)
│   ├── config.py logging_utils.py trajectory.py embeddings.py
│   ├── physics/                     # ← differentiation core
│   │   ├── failure_modes.py         #   taxonomy + localizable→actionable bridge
│   │   ├── sketch.py sim_wrapper.py #   physics sketch + simulator
│   │   └── control_render.py        #   sketch → generator control signal (C1)
│   ├── planning/event_graph.py      # GEST-style IR + validation
│   ├── memory/lesson_library.py     # C4 cross-task memory
│   ├── models/                      # wrappers (mock now; real = v0.2)
│   │   ├── llm.py mllm.py image_edit.py
│   │   ├── video_gen.py             #   factory + mock
│   │   └── video_gen_backends.py    #   OmniWeaving / Wan / Veo skeletons
│   ├── agents/                      # screenwriter director physics_planner
│   │                                #   plan_validator generator verifier refiner
│   ├── critics/                     # semantic physics physics_consistency (C6)
│   │                                #   consistency rhythm + board + tournament
│   ├── tools/                       # metric_tool assembly_tool retrieval_tool
│   ├── orchestration/state.py       # run state
│   ├── pipeline/                    # understand plan generate_loop assemble run
│   └── prompts/                     # agent prompt templates (not hardcoded)
└── tests/                           # unit + integration (pytest)
```

---

## Quickstart (v0.1 — CPU, no keys)

```bash
cd Maestro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # numpy, pyyaml, pytest

# smoke test
pytest -q

# run the pipeline end-to-end (mock models)
python scripts/run_pipeline.py \
  --prompt "a ball is thrown and bounces; a person runs through a city" \
  --music data/track.mp3 \
  --image data/hero.png \
  --output outputs/demo.mp4
```

CLI flags: `--prompt` (required), `--source` (0+ source videos), `--image` (0+
reference images), `--music`, `--output`, `--config`.

### Outputs
| File | Meaning |
|---|---|
| `outputs/demo.mp4` | the assembled video — **v0.1 = placeholder text file** |
| `outputs/demo.report.json` | per-shot revision count, convergence, score history, **HSI `tier_used` / `escalations`**, final metrics (incl. `p2_sketch_consistency`) |
| `outputs/demo.trajectory.jsonl` | every agent decision (state/action/observation) including `replan_sketch` / `refine_spec` when HSI escalates |
| `outputs/lessons.jsonl` | cross-task experience memory (C4) — keyed on the *actually resolved* failure mode |

### Why no pixels yet
v0.1 mocks every heavy model so the *orchestration* is testable on CPU. The loop,
metrics, physics verdicts, planning validation, report and logs are all real; only
the pixel-producing step is a stub. To get real video, do v0.2 below.

---

## Configuration reference (`configs/default.yaml`)

```yaml
models:                      # v0.1 all mock; change name/weights for v0.2
  llm:        {name: mock-llm}
  mllm:       {name: mock-mllm}
  video_gen:  {name: mock-video-gen}   # weights_path / device for local backends
  image_edit: {name: mock-image-edit}

plan:
  n_shots: 3            # default shots (follows music section count if music given)
  max_shots: 6
  shot_duration: 3.0
  max_plan_iters: 3     # plan-level Validate→Correct loop cap

compose:
  fps: 8
  n_candidates: 2       # tournament pool per shot (E3)
  max_revisions: 5      # self-improvement rounds cap (C2/C5)
  k_retries: 2          # per-revision retries WITHIN each HSI tier (C5)

metrics:                # 7 dims; weighted_total drives Verifier's monotonic check
  weights:
    m1_semantic: 0.22
    m2_temporal: 0.13
    p1_physics:  0.22   # native physics-failure modes (C1 critic layer)
    p2_sketch_consistency: 0.10   # closed-loop sketch verify (C6, v0.2.1)
    id1_identity: 0.13
    m5_rhythm:   0.10
    aesthetic:   0.10

physics:
  simulator: mock              # v0.2: mujoco / newton / particle-sim
  acceptance_severity: 0.30    # a failure mode below this is "resolved"
```

Tuning intuition: quality↑ → raise `compose.max_revisions` / `n_candidates`
(slower); native physics↑ → raise `metrics.weights.p1_physics`, lower
`physics.acceptance_severity`; sketch-tracking↑ → raise
`metrics.weights.p2_sketch_consistency`. Any value is overridable via
`--config your.yaml`.

---

## v0.2 — going real (real pixels)

### What is already wired vs what you implement

| Wrapper | Factory | v0.1 | Real backend in v0.2 |
|---|---|---|---|
| **video_gen** | `build_video_gen` | mock | **skeletons exist** in `video_gen_backends.py` (OmniWeaving / Wan / Veo) — just fill `generate()` |
| **mllm** (judge/critic) | `build_mllm` | mock | add a `BaseMLLMClient` subclass + extend the factory |
| **llm** (planning) | `build_llm` | mock | add a `BaseLLMClient` subclass + extend the factory |
| **image_edit** | `build_image_edit` | mock | add a `BaseImageEditClient` subclass + extend the factory |
| **physics sim** | `MockSimulator` | analytic | implement `BaseSimulator` (MuJoCo / Newton) |

The factories for `video_gen` already dispatch to real backends by `name`; the
other three factories currently always return the mock (extend them the same way).

### Prerequisites
- A GPU (for local video generation / local VLM). API-only backends can run CPU-side.
- `ffmpeg` on PATH (real assembly; without it, assembly falls back to a manifest).
- Extra Python deps for whatever backend you pick (torch, the model's package, etc.).

### Recommended order (highest payoff first)
1. **`video_gen`** — see the first real pixels.
2. **`mllm`** — let `PhysicsCritic`/`SemanticCritic` use a real VLM so physics
   verdicts are grounded (PhyGenEval-style), not mock.
3. **`llm`** — smarter Screenwriter/Director planning.
4. **`physics` simulator** — replace analytic priors with MuJoCo/Newton.

### Step 1 — install deps & set keys
```bash
pip install -e .                      # installs the package
pip install torch                     # + your backend's package, e.g. omniweaving / wan
cp .env.example .env                  # then fill in:
#   OMNIWEAVING_WEIGHTS=/data/ckpts/omniweaving      (or WAN_WEIGHTS / VEO_API_KEY)
#   QWEN_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY  (when wiring mllm/llm)
```

### Step 2 — point the config at a real backend
```yaml
# configs/local.yaml
models:
  video_gen:
    name: "omniweaving"               # dispatched by build_video_gen
    weights_path: "/data/ckpts/omniweaving"
    device: "cuda"
```

### Step 3 — implement the backend body
Open `src/maestro/models/video_gen_backends.py`, pick your class
(`OmniWeavingClient` recommended — native text + multi-image + video conditioning),
and fill the two TODOs:
- `_ensure_loaded()` — load the pipeline from `self.weights_path` onto `self.device`.
- `generate(...)` — the contract is **identical to the mock**, so nothing else in
  the data flow changes. Critically, map the conditioning inputs:
  - `control_signal` → run it through `physics.control_render.load_control_spec()`
    to get a model-agnostic `ControlSpec` (trajectory → motion/drag control). **This
    is the C1 bridge — it is what makes generation physics-grounded.**
  - `first_frame` → I2V anchor (continuity across keyframe-local repairs, C2).
  - `reference_images` → identity/style anchors from `RetrievalTool` (consistency, E1).

> The video model **must** accept conditioning (control / first-frame / reference).
> If it is text-only, the physics sketch degrades to a prompt hint and the C1
> grounding claim weakens — prefer OmniWeaving/Wan over a pure text API.

### Step 4 — run
```bash
python scripts/run_pipeline.py \
  --prompt "a glass of water is poured onto a table; it splashes" \
  --image data/hero.png \
  --output outputs/real.mp4 \
  --config configs/local.yaml
```
With `ffmpeg` present, `outputs/real.mp4` is now a real, assembled clip; inspect
`outputs/real.report.json` to see how many self-improvement rounds each shot took
and how the physics score climbed.

### Wiring the other wrappers (same pattern)
For `mllm` / `llm` / `image_edit`: add a subclass implementing the ABC in the
respective `models/*.py`, then extend its `build_*` factory to dispatch on `name`
(mirror how `build_video_gen` does it). For `mllm`, implement `assess_semantic`,
`assess_physics`, and optionally `compare` (for the tournament).

---

## Tests & CI

```bash
pytest -q
```

Root-repo GitHub Actions is currently disabled (it targeted the old project). To
run CI on Maestro only, set the workflow's `working-directory` to `Maestro/`,
install `Maestro/requirements.txt`, and run `Maestro/tests`.

---

## Status

- **v0.1** = scaffold, all heavy models mocked, CPU-only.
- **v0.2** swaps mocks for real models (OmniWeaving/Wan/Veo · Qwen-VL ·
  DeepSeek/GPT/Claude · MuJoCo/Newton) behind the same wrapper ABCs — at that
  point API keys / GPU are needed (see `.env.example`). The orchestration,
  self-improvement loop, physics grounding and evaluation harness do not change
  between v0.1 and v0.2 — only the model wrappers do.
- **v0.2.1** (current) deepens the agentic paradigm itself, *without* model
  changes:
  - **C5 HSI** — generation-time self-improvement is now hierarchical
    (Tier 0 keyframe edit → Tier 1 physics-sketch replan → Tier 2 spec rewrite
    → Tier 3 escape). Verifier's monotonic-improvement rule still applies at
    every tier.
  - **C6 Sketch↔Video consistency** — a new `PhysicsConsistencyCritic` flags
    rendered clips that diverge from the sketch's predicted trajectory; the
    physics layer is now bidirectional. Surfaces as `p2_sketch_consistency`.
  - **Logic hardening** — `ReviewBoard.recompute_metrics` refreshes scores
    after the escape hatch so the Verifier's next comparison is honest;
    `LessonLibrary` now distills from the *actually resolved* failure mode
    rather than `expected_modes[0]`.
  - **Observability** — the JSON report now includes per-shot `tier_used` and
    `escalations`, and the trajectory log adds `replan_sketch` / `refine_spec`
    actions when HSI escalates.

  Tests: `pytest -q` → **35 passed** (CPU, ~0.1s). See
  `tests/unit/test_hsi_and_consistency.py`.
