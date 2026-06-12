# Maestro

**Training-free, self-improving, physically-grounded agentic video generation.**

You give an instruction plus optional multimodal materials (video / image / music).
Maestro plans with multiple agents, generates each shot, reviews it with a board of
critics against a quantitative metric suite, and **locally repairs failing keyframes
in a monotonic self-improvement loop** — with **physics treated as a first-class
citizen**.

- **Mock-first (default):** the full control flow, self-improvement loop,
  physics verification, planning validation, and logging run **end-to-end on
  CPU with no API keys**. The `.mp4` it writes is a *placeholder* (see "Why no
  pixels yet" below).
- **Real pixels (v0.4):** flip `models.video_gen.name: wavespeed` with a
  `$WAVESPEED_API_KEY` (hosted REST API, fully implemented, no local GPU) — or
  wire a local backend behind the same interface. See "Going real" below.

Design rationale & differentiation vs UniVA / VideoAgent / ViMax / VISTA / M3 /
event-graph: see `REPORT_AND_INSTRUCTIONS.md`. **Side-by-side comparison vs
UniVA / CutClaw / VISTA / M3 / VideoAgent / ViMax / Event-Graph with measured
effect of each Maestro innovation: see `COMPARISON.md`.** **v0.3 research:
agentic memory + skill paradigms survey, the gap, and the C7+C8 task-specific
design: see `RESEARCH_MEMORY_SKILL.md`.** **v0.4 physics (reference-free
verification): literature survey `docs/research/survey_physics_2026_06.md` +
positioning `docs/research/INNOVATION_PLAN_2026_06.md` §3.2
(`PHYSICS_LITERATURE_REVIEW.md` is the earlier, partially superseded review).**
End-to-end data flow & config: see
`DATAFLOW.md`. Incremental modules & citations: see `IMPROVEMENTS.md`.

---

## Eight core innovations

1. **Physics as a first-class citizen (C1)** — a *critic layer* + a
   *measurement layer*, and **nothing is ever injected into the generator**:
   physics is *verified from the generated pixels*. The old sketch/simulator
   line is gone entirely (v0.4) — a synthetic sketch cannot control a frozen
   video model, and comparing against *one* simulated rollout presumes
   masses/friction/scale that are unknowable from a prompt (see
   `PHYSICS_LITERATURE_REVIEW.md` + `docs/research/survey_physics_2026_06.md`).
   The critic layer (VLM-judged, `p1_physics`) localizes failures by mode
   (penetration / gravity / collision / fluid / object-permanence /
   deformation / conservation) to specific frames, each mapped to an
   executable fix. The measurement layer (C6 below) recovers the **observed**
   motion from the clip and asks the reference-free, parameter-free question
   *"is there ANY physically consistent explanation for this track?"*. Physics
   improvement is then driven by **test-time search** — best-of-N tournament +
   monotonic Verifier (+ optional world-model reward, à la WMReward) — plus
   HSI targeted repair. All training-free.
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
   - **Tier 1** re-annotate physics with **tightened verification strictness**
     + regenerate with anti-violation hints distilled from the *observed*
     failures (harder bar for the verifier, clearer target for the generator)
   - **Tier 2** Director rewrites the `ShotSpec` (cinematography + prompt) —
     bounded VISTA-style replan
   - **Tier 3** escape hatch (drop the worst remaining defect)

   Verifier's monotonic-improvement rule applies at every tier; we never accept
   a regression. After any acceptance the next revision restarts at Tier 0 —
   cost-amortized adaptive scope. **This fills the gap between VISTA (always
   whole-segment) and M3 (always local patch).**
6. **Reference-free physics-from-pixels verification (C6, v0.2.1; rewritten
   v0.4)** — a point tracker (deterministic mock by default;
   `cotracker`/`tapir` lazy-load torch behind the same ABC) recovers the
   **observed** per-entity tracks from the generated clip; a **reliability
   gate** (`physics/reliability.py`) certifies tracks *before* trusting them —
   trackers are trained on real video and lie on generated content, and
   cross-tracker disagreement is itself an implausibility cue (nobody else
   quantifies tracker reliability on generated video); the **law layer**
   (`physics/laws.py`) fits the small family of passive motion laws (static /
   constant-velocity / constant-acceleration with a *free* gravity vector, so
   no scale calibration) and takes the best-fit residual as the violation,
   plus localized anomaly detectors (teleport→object-permanence, mid-air
   reversal→gravity, energy gain→conservation, jerk spike→collision); a
   **VerifiabilityRouter** (`physics/router.py`) assigns each annotated entity
   the strongest tier that can actually check it (measurement / world_model /
   vlm / none) and reports coverage explicitly — partial verification never
   reads as full verification. Verdicts are measured, interpretable,
   per-entity, frame-localized; reported as `p2_law_consistency`
   (source `law_verifier`), kept separate from the VLM-judged `p1_physics`.
   Honest degradation: on an unreadable (mock) clip the verifier stays silent;
   a misconfigured real backend fails loudly rather than emitting a
   fake-perfect p2.
7. **PhysicsTyped SkillLibrary (C7, v0.3 NEW)** — *compiled shot recipes*
   distilled when HSI converges at Tier 0 with non-trivial initial severity
   (≥ 0.5). Skills are keyed on `PhysFailureMode` signatures (not pure text)
   and carry pointers to coupled lessons that auto-inject on retrieval.
   Different from Voyager (env-reward distillation) and SkillWeaver
   (rehearsal-repeatability). Lifecycle borrowed from SkillOps. See
   `RESEARCH_MEMORY_SKILL.md` §4.1.
8. **Multi-Layer Memory (C8, v0.3 NEW)** — six-tier memory: working /
   episodic (with **replay**) / semantic (extended with A-MEM bidirectional
   links) / procedural (= C7) / entity (VideoMemory-style, **cross-run**) /
   preference (Me-Agent-style). `MultiLayerMemory` façade exposes an
   associative query lighting up multiple tiers at once (HippoRAG-inspired,
   lightweight in v0.3). See `RESEARCH_MEMORY_SKILL.md` §4.2.

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
                              →    Tier 1 physics replan (PhysicsPlanner:
                                          strictness↑ + anti-violation hints)
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
│   ├── physics/                     # ← differentiation core (v0.4 reference-free)
│   │   ├── annotate.py router.py    #   entity/motion-class annotation + verifiability tiers
│   │   ├── tracks.py reliability.py #   observed tracks + tracker-reliability gate
│   │   ├── laws.py verifier.py      #   best-law residual + the assembled C6 stack
│   │   ├── track_extractor_backends.py  # real CoTracker / TAPIR (torch)
│   │   └── failure_modes.py         #   taxonomy + localizable→actionable bridge
│   ├── planning/event_graph.py      # GEST-style IR + validation
│   ├── memory/lesson_library.py     # C4 cross-task memory
│   ├── models/                      # wrappers (mock now; real = v0.2)
│   │   ├── llm.py mllm.py image_edit.py
│   │   ├── video_gen.py             #   factory + mock
│   │   └── video_gen_backends.py    #   WaveSpeed (real, hosted) + OmniWeaving/Wan/Veo skeletons
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

## Quickstart (v0.4 — CPU, no keys, no GPU)

```bash
cd Maestro
python -m venv .venv && source .venv/bin/activate

# Minimal: mock pipeline + tests (numpy + pyyaml + pytest only)
pip install -e '.[dev]'

# Recommended on a server: also get the FastAPI server + image ops
pip install -e '.[all]'                    # core + server + image
# (or `pip install -r requirements.txt` for the same set)

# unit + integration tests — should print "125 passed"
pytest -q

# single-shot generation (mock models, CPU)
maestro run-once \
  --prompt "a ball is thrown and bounces; a person runs through a city" \
  --music data/track.mp3 --image data/hero.png \
  -o outputs/demo.mp4

# operator's healthcheck — verify the box is wired right
maestro smoke

# bring up the FastAPI server (UniVA-compatible /health)
maestro serve --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

The `maestro` command is registered by `pyproject.toml`'s `[project.scripts]`,
so `pip install -e .` is enough to get it on PATH — no `PYTHONPATH` gymnastics.

CLI flags (`run-once`): `--prompt` (required), `--source` (0+ source videos),
`--image` (0+ reference images), `--music`, `--output`, `--config`.

### Outputs
| File | Meaning |
|---|---|
| `outputs/demo.mp4` | the assembled video — **v0.1 = placeholder text file** |
| `outputs/demo.report.json` | per-shot revision count, convergence, score history, **HSI `tier_used` / `escalations`**, final metrics (incl. the measured `p2_law_consistency`) |
| `outputs/demo.trajectory.jsonl` | every agent decision (state/action/observation) including `annotate_physics` at planning time and `replan_physics` / `refine_spec` when HSI escalates |
| `outputs/lessons.jsonl` | cross-task experience memory (C4) — keyed on the *actually resolved* failure mode |

### Why no pixels yet
The default config mocks every heavy model so the *orchestration* is testable on
CPU. The loop, metrics, physics verdicts, planning validation, report and logs
are all real; only the pixel-producing step is a stub (and the mock track
extractor *synthesizes* tracks so the law checks have a signal path to
exercise). To get real video, the fastest route is the WaveSpeed backend —
see "Going real" below.

---

## Tool library (UniVA-inspired, v0.2.2)

Every tool self-describes via `BaseTool.spec` and registers with the in-process
`ToolRegistry` (`maestro.tools.default_registry()`). Borrowed *pattern* from
UniVA's MCP tool servers (arXiv:2511.08521) without the wire protocol.

| Category | Tools | Purpose |
|---|---|---|
| **analysis**   | `video_probe`, `frame_extract`, `caption` | inspect / extract / describe assets before generation |
| **generation** | `audio_gen` (+ neural T2V via `models/video_gen_backends.py`) | synthesize new media |
| **editing**    | `assemble`, `video_concat`, `image_ops` | ffmpeg/PIL deterministic transforms |
| **tracking**   | `detect_objects` | object/identity bboxes for grounding |
| **metric**     | `compute_metrics` | the C3 quantitative scorer |
| **physics**    | (annotation + verification stack; see `physics/`) | the C1/C6 differentiation core — measured verdicts from pixels, nothing injected |
| **retrieval**  | `retrieve_assets` (constructed per run) | E1 grounding lookup |

Every tool degrades gracefully when its optional system dep (ffmpeg, PIL) is
absent — sandbox-runs always produce valid `Path` outputs so the higher loop
keeps converging.

### Plan ↔ Act
Maestro's planning agents (Screenwriter / Director / PhysicsPlanner / Refiner)
are domain-specialized **plan** agents. `ActAgent` is the generic **executor**
mirroring UniVA's Act side: it takes a list of `ToolCall(name, args, kwargs)`,
routes each through the registry, and writes a `tool_call` event into the
trajectory. Plan→Act is now a uniform, observable handoff:

```python
from maestro.agents import ActAgent, ToolCall
plan = [
    ToolCall("video_probe", args=["src.mp4"]),
    ToolCall("caption", args=["src.mp4"]),
    ToolCall("detect_objects", kwargs={"media": "src.mp4", "query": "hero ball"}),
]
results = ActAgent().run(plan)   # each call logged with category + status
```

---

## Server (production-ready shim)

Same `/health` shape as UniVA's `univa_server.py` so existing orchestrators
(k8s liveness, docker-compose healthchecks) just work.

```bash
maestro serve --host 0.0.0.0 --port 8000
# or: uvicorn maestro.server:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Body / Purpose |
|---|---|---|
| `/health`   | GET  | `{status, service, version, n_tools}` — liveness probe |
| `/tools`    | GET  | machine-readable manifest of every registered tool's `spec` |
| `/generate` | POST | `{prompt, source_videos?, images?, music?}` → enqueues a job, returns `job_id` |
| `/jobs/{id}` | GET  | poll job state (`queued/running/done/error`), output path, report |

Jobs run in an in-process `ThreadPoolExecutor` (v0.2.2; v0.3 swaps for
Redis/Celery — the interface in `server.JobStore` is stable for the upgrade).

### Docker (CPU-friendly)

```bash
docker build -t maestro:0.2.2 .
docker run --rm -p 8000:8000 maestro:0.2.2
curl http://localhost:8000/health
```

The image is `python:3.11-slim + ffmpeg` (no GPU dep). `HEALTHCHECK` hits
`/health` so k8s & docker-compose probe out-of-the-box. v0.3 GPU image: swap
the base to `nvidia/cuda:12.4.0-runtime-ubuntu22.04` and add torch in a
separate stage — application contract above does NOT change.

---

## Configuration reference (`configs/default.yaml`)

```yaml
models:                      # all mock by default; flip a name to go real
  llm:        {name: mock-llm}
  mllm:       {name: mock-mllm}
  video_gen:  {name: mock-video-gen}   # "wavespeed" = real pixels via hosted API
                                       # ($WAVESPEED_API_KEY, no GPU); local:
                                       # "omniweaving" / "wan" (weights_path/device)
  image_edit: {name: mock-image-edit}
  # --- optional physics backends (omit / mock = identical CPU behavior) ---
  track_extractor: {name: mock-track}  # C6 observed tracks: "cotracker" / "tapir"
                                       # (need torch + real decoded frames)
  # world_reward:  {name: mock-world-reward}   # adds wm_reward (V-JEPA-2 / WMReward)

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

metrics:                # weighted_total drives Verifier's monotonic check
  weights:
    m1_semantic: 0.22
    m2_temporal: 0.13
    p1_physics:  0.22   # native physics-failure modes (C1 critic layer, VLM-judged)
    p2_law_consistency: 0.10   # measured law-residual verification (C6 v0.4)
    id1_identity: 0.13
    m5_rhythm:   0.10
    aesthetic:   0.10

physics:
  acceptance_severity: 0.30    # a failure mode below this is "resolved"

memory:                  # C8 Multi-Layer Memory (v0.3) — all tiers on by default
  user_id: default
  enable_skills: true          # C7 SkillLibrary (procedural memory)
  enable_entities: true        # cross-run identity persistence
  enable_preferences: true     # per-user cinematic / strictness biases
  enable_episodes: true        # episodic trace store
  skill_distill_severity_threshold: 0.5
```

Tuning intuition: quality↑ → raise `compose.max_revisions` / `n_candidates`
(slower); native physics↑ → raise `metrics.weights.p1_physics`, lower
`physics.acceptance_severity`; measured-law verification↑ → raise
`metrics.weights.p2_law_consistency`. Any value is overridable via
`--config your.yaml`. Memory persists to `<output_dir>/memory/` across runs;
delete that dir for a cold start.

---

## Going real (real pixels / real physics — GPU)

> The CPU mock pipeline above runs with no GPU and no keys. The **fastest route
> to real pixels needs no GPU either**: set `models.video_gen.name: wavespeed`
> with a `$WAVESPEED_API_KEY` (hosted REST API — the same service UniVA uses).
> For local backends, swap a mock wrapper for a real backend (each is a stable
> ABC) and install that backend's own heavy deps (torch etc.) to match your
> CUDA — they are intentionally not pinned in this repo.

### What is already wired vs what you implement

| Wrapper | Factory | Default | Real backend |
|---|---|---|---|
| **video_gen** | `build_video_gen` | mock | **`wavespeed` fully implemented** (hosted REST: submit → poll → download; `$WAVESPEED_API_KEY`; default `model_id: bytedance/seedance-v1-pro-t2v-480p`, auto-switches to the i2v variant when a `first_frame` is given). OmniWeaving / Wan / Veo are local-weights **skeletons** — fill `generate()` |
| **mllm** (judge/critic) | `build_mllm` | mock | add a `BaseMLLMClient` subclass + extend the factory |
| **llm** (planning) | `build_llm` | mock | add a `BaseLLMClient` subclass + extend the factory |
| **image_edit** | `build_image_edit` | mock | add a `BaseImageEditClient` subclass + extend the factory |
| **track_extractor** (C6) | `build_track_extractor` | mock (synthesizes tracks) | `cotracker` / `tapir` **wired** behind the same ABC — install torch + weights to track real frames |

The factories for `video_gen` and `track_extractor` already dispatch to real
backends by `name`; the other factories currently always return the mock
(extend them the same way).

### Prerequisites
- For `wavespeed`: just an API key — no GPU, no torch.
- For local backends: a GPU + extra Python deps (torch, the model's package, etc.).
- `ffmpeg` on PATH (real assembly; without it, assembly falls back to a manifest).

### Recommended order (highest payoff first)
1. **`video_gen` = `wavespeed`** — first real pixels in minutes, no GPU.
2. **`mllm`** — let `PhysicsCritic`/`SemanticCritic` use a real VLM so p1
   verdicts are grounded (PhyGenEval-style), not mock.
3. **`track_extractor` = `cotracker`/`tapir`** — measured `p2_law_consistency`
   from *real* tracked pixels (the mock synthesizes tracks to exercise the
   law checks; the real path needs torch + weights).
4. **`llm`** — smarter Screenwriter/Director planning.

### Step 1 — install deps & set keys
```bash
pip install -e '.[all]'               # wavespeed needs `requests` (in [all])
cp .env.example .env                  # then fill in:
#   WAVESPEED_API_KEY=...                            (fastest: hosted video gen)
#   OMNIWEAVING_WEIGHTS=/data/ckpts/omniweaving      (or WAN_WEIGHTS / VEO_API_KEY)
#   QWEN_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY  (when wiring mllm/llm)
```

### Step 2 — point the config at a real backend
```yaml
# configs/local.yaml
models:
  video_gen:
    name: "wavespeed"                 # fully implemented — real pixels, no GPU
    model_id: "bytedance/seedance-v1-pro-t2v-480p"   # any WaveSpeed t2v/i2v id
    # api_key: ...                    # or $WAVESPEED_API_KEY
  # local alternative (fill the skeleton first):
  # video_gen: {name: "omniweaving", weights_path: "/data/ckpts/omniweaving", device: "cuda"}
```

### Step 3 — (local backends only) implement the backend body
`wavespeed` works out of the box. For a local model, open
`src/maestro/models/video_gen_backends.py`, pick your class
(`OmniWeavingClient` recommended — native text + multi-image conditioning),
and fill the two TODOs:
- `_ensure_loaded()` — load the pipeline from `self.weights_path` onto `self.device`.
- `generate(...)` — the contract is **identical to the mock**, so nothing else in
  the data flow changes. The conditioning inputs:
  - `first_frame` → I2V anchor (continuity across keyframe-local repairs, C2).
  - `reference_images` → identity/style anchors from `RetrievalTool` (consistency, E1).

> There is **no physics control signal** (v0.4): physics is *verified from the
> generated pixels*, never injected into the generator. That is why a text-only
> API backend is now perfectly fine — the C6 verifier measures whatever pixels
> come back, regardless of how they were conditioned. `first_frame` /
> `reference_images` buy continuity (C2) and identity (E1), not physics.

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
- **v0.2** swaps mocks for real models (WaveSpeed/OmniWeaving/Wan/Veo ·
  Qwen-VL · DeepSeek/GPT/Claude) behind the same wrapper ABCs — at that
  point API keys / GPU are needed (see `.env.example`). The orchestration,
  self-improvement loop, physics verification and evaluation harness do not
  change between mock and real — only the model wrappers do.
- **v0.4 (current)** — *physics rewritten: reference-free
  "physics-from-pixels" verification*. The sketch/simulator line
  (v0.2.1–v0.3) is **gone entirely** — both halves failed scrutiny: a
  synthetic sketch cannot control a frozen video model, and comparing the
  clip against *one* simulated rollout presumes masses/friction/scale that
  are unknowable from a prompt. v0.4 keeps the question, removes the
  reference (rationale: `docs/research/INNOVATION_PLAN_2026_06.md` §3.2 +
  `docs/research/survey_physics_2026_06.md`):
  - **`physics/` rebuilt**: `annotate.py` (entities + motion class
    [ballistic/rigid/fluid/agentive/static] + interactions + expected modes
    + strictness — *verification seeds*, never trajectories or control),
    `router.py` (VerifiabilityRouter: measurement / world_model / vlm /
    none, with an explicit coverage report — partial verification never
    reads as full verification), `tracks.py` +
    `track_extractor_backends.py` (observed tracks from the generated clip;
    deterministic mock on CPU, real CoTracker/TAPIR wired but needing
    torch + weights), `reliability.py` (certify tracks before trusting
    them — trackers lie on generated video; cross-tracker disagreement is
    itself an implausibility cue, which nobody else quantifies), `laws.py`
    (best fit over static / constant-velocity / constant-acceleration with
    a *free* gravity vector → residual = violation; anomaly detectors:
    teleport→object_permanence, mid-air reversal→gravity_inertia, energy
    gain→conservation, jerk spike→collision), `verifier.py` (the assembled
    stack). **Deleted:** `sketch.py`, `sim_wrapper.py`, `control_render.py`,
    `oracle.py`.
  - **API renames**: `PhysicsSketch` → `PhysicsAnnotation`;
    `ShotSpec.physics_sketch` → `ShotSpec.physics_annotation`; metric
    `p2_sketch_consistency` → `p2_law_consistency` (measured law verdicts,
    `source="law_verifier"`; `p1_physics` stays VLM-judged); trajectory
    actions `build_sketch` / `replan_sketch` → `annotate_physics` /
    `replan_physics`. HSI Tier 1 is now "tighten verification strictness +
    regenerate with anti-violation hints from the *observed* failures".
  - **`models/video_gen.py`**: `generate()` has no `control_signal` param —
    conditioning is `first_frame` + `reference_images` only. New
    **fully-implemented `WaveSpeedClient`** (hosted REST: submit → poll →
    download; `$WAVESPEED_API_KEY`; default model
    `bytedance/seedance-v1-pro-t2v-480p`) — the fastest no-GPU route to
    real pixels, the same service UniVA uses. A text-only backend is now
    fine because physics is *verified*, not injected.
  - **Positioning vs literature** (see `COMPARISON.md`): PSIVG / PhyRPR /
    PhysCtrl inject simulation open-loop and never verify; WMReward closes
    the loop with an opaque scalar (can't localize or explain); PhyT2V
    closes it through a lossy VLM-text bottleneck; Morpheus / PISA measure
    but only for benchmarking. The intersection — measured + interpretable
    + localized + drives selection AND targeted regen + training-free — is
    unoccupied, and the new framing has no "your simulator is wrong"
    attack surface.
  - Tests: **125 passed** (CPU, no GPU, no keys) —
    `tests/unit/test_physics_laws.py` + `test_physics_verifier.py` replace
    the deleted `test_physics.py` / `test_physics_oracle.py`.
- **v0.3** — *memory + skill: two more innovations*. Adds two
  task-specific extensions on top of C1-C6 (see `RESEARCH_MEMORY_SKILL.md`
  for the survey + design). No model changes; mock pipeline still CPU-only.
  - **C7 PhysicsTyped SkillLibrary** — a *compiled shot recipe* is born when
    HSI converges at Tier 0 with non-trivial initial severity (≥ 0.5). Skills
    are keyed on `PhysFailureMode` signatures (not text) and carry pointers
    to their coupled lessons; retrieval auto-injects them into the next
    plan. Differentiates from Voyager (env-reward distillation) /
    SkillWeaver (rehearsal repeatability).
  - **C8 Multi-Layer Memory** — six tiers (working / episodic / semantic /
    procedural / entity / preference). `LessonLibrary` extended with A-MEM
    bidirectional links + confidence + stable `lesson_id`; new
    `SkillLibrary` / `EntityStore` (cross-run identity persistence, à la
    VideoMemory but cross-run) / `PreferenceStore` (Me-Agent style) /
    `EpisodicStore` (with replay). `MultiLayerMemory` façade exposes an
    associative query that lights up multiple tiers at once.
  - Wired into the pipeline: `plan_shots` calls `skill_library.retrieve` and
    attaches `spec.matched_skill`; `generate_shot` calls
    `skill_library.distill` post-acceptance; `understand` consults
    `EntityStore` for cross-run dedup; `run_maestro` writes one
    `EpisodicTrace` per task.
  - Report adds `skills_learned`, `entities_persisted`, per-shot
    `matched_skill_id` / `distilled_skill_id` / `distilled_lesson_id`.
  - Configurable under `memory:` in `configs/default.yaml`; persisted to
    `<output_dir>/memory/{lessons,skills,entities,preferences,episodes}.{jsonl,json}`.
  - *(historical — **superseded by v0.4**)* Physics was repositioned to
    "sketch-as-oracle" in this release: a rigid-body simulator computed an
    *expected* trajectory and a `TrajectoryOracle` scored observed-vs-expected
    Trajectory-L2. v0.4 deleted that line (`sketch.py` / `sim_wrapper.py` /
    `oracle.py`) because the comparison still presumed simulator parameters a
    prompt cannot supply — see the v0.4 entry above. What survives: the
    track-extractor factory (`mock-track` default, real `cotracker`/`tapir`)
    and the optional `world_reward` (`models/world_reward.py`, WMReward /
    V-JEPA-2) adding a `wm_reward` metric.
  - Tests at v0.3: 113 passed (now **125** after the v0.4 physics rewrite).

- **v0.2.2** — *UniVA borrowing for breadth*. Same six core innovations
  preserved; pulls in UniVA-style infrastructure to make the framework
  server-deployable:
  - **ToolRegistry** with 7-category taxonomy (analysis / generation / editing /
    tracking / physics / metric / retrieval). 9 default tools self-register;
    every tool exposes a `spec` for discovery.
  - **ActAgent** — UniVA's Act-side dual-agent executor. Takes
    `list[ToolCall]`, routes through the registry, logs every call.
  - **FastAPI server** (`maestro serve`) — UniVA-compatible `/health` +
    `/tools` manifest + `/generate` job submission + `/jobs/{id}` polling.
    Optional dep — graceful degradation when fastapi is absent.
  - **CLI** (`maestro {smoke,serve,run-once}`) registered via
    `pyproject.toml [project.scripts]`. `maestro smoke` is the one-command
    operator healthcheck.
  - **Dockerfile** (`python:3.11-slim + ffmpeg`, `HEALTHCHECK` on `/health`)
    for "upload to server and run" literally.
  - Tests: 54 passed (CPU, ~0.6s). 8 new tools tests + 4 server tests +
    ActAgent Plan→Act handoff test.

- **v0.2.1** deepens the agentic paradigm itself, *without* model
  changes (physics passages below are **historical — superseded by v0.4**,
  where the sketch line was removed entirely):
  - **C5 HSI** — generation-time self-improvement is now hierarchical
    (Tier 0 keyframe edit → Tier 1 physics replan → Tier 2 spec rewrite
    → Tier 3 escape). Verifier's monotonic-improvement rule still applies at
    every tier. *(Tier 1 was "rebuild the physics sketch" here; since v0.4 it
    is "tighten verification strictness + anti-violation hints".)*
  - **C6 Sketch↔Video consistency** *(superseded by v0.4's reference-free
    verifier)* — the original `PhysicsConsistencyCritic` flagged rendered
    clips that diverged from the sketch's predicted trajectory; it surfaced
    as `p2_sketch_consistency` (now `p2_law_consistency`).
  - **Logic hardening** — `ReviewBoard.recompute_metrics` refreshes scores
    after the escape hatch so the Verifier's next comparison is honest;
    `LessonLibrary` now distills from the *actually resolved* failure mode
    rather than `expected_modes[0]`.
  - **Observability** — the JSON report now includes per-shot `tier_used` and
    `escalations`, and the trajectory log adds replan / `refine_spec`
    actions when HSI escalates (the replan action is named `replan_physics`
    since v0.4).

  Tests: `pytest -q` → **35 passed** (CPU, ~0.1s) at v0.2.1; →
  **54 passed** (CPU, ~0.6s) at v0.2.2 with the new tool/server tests. See
  `tests/unit/test_hsi_and_consistency.py` (v0.2.1) and
  `tests/unit/test_tools_and_act.py` + `test_server.py` (v0.2.2).
