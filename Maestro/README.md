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
event-graph: see `REPORT_AND_INSTRUCTIONS.md`. **Side-by-side comparison vs
UniVA / CutClaw / VISTA / M3 / VideoAgent / ViMax / Event-Graph with measured
effect of each Maestro innovation: see `COMPARISON.md`.** **v0.3 research:
agentic memory + skill paradigms survey, the gap, and the C7+C8 task-specific
design: see `RESEARCH_MEMORY_SKILL.md`.** End-to-end data flow & config: see
`DATAFLOW.md`. Incremental modules & citations: see `IMPROVEMENTS.md`.

---

## Eight core innovations

1. **Physics as a first-class citizen (C1)** — an *oracle layer* + a *critic
   layer*. The oracle is a lightweight rigid-body **simulator** (ground/wall
   collision response, restitution, contact events — `physics/sim_wrapper.py`)
   that computes the **expected** motion of a scene. Crucially the sketch is
   **not a controller** — conditioning a frozen video model on an abstract sim
   trajectory is an unvalidated path (trajectories under-determine physics; see
   `PHYSICS_LITERATURE_REVIEW.md`). Instead the simulator is a **verification
   oracle**: a track extractor recovers the **observed** motion from the
   generated video and we score the deviation (PISA-style normalized
   Trajectory-L2, `physics/oracle.py`). Physics improvement is then driven by
   **test-time search** — best-of-N tournament + monotonic Verifier (+ optional
   world-model reward, à la WMReward). The *critic layer* localizes failures by
   mode (penetration / gravity / collision / fluid / object-permanence /
   deformation / conservation) to specific frames, each mapped to an executable
   fix.
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
6. **Sketch-as-oracle physics verification (C6, v0.2.1; repositioned v0.3)** —
   a dedicated `PhysicsConsistencyCritic` compares the **observed** motion of
   the generated clip (recovered by a track extractor — mock by default;
   `cotracker`/`tapir` lazy-load torch behind the same ABC) against the
   **simulator's expected** motion, via PISA-style normalized Trajectory-L2.
   Divergence beyond a threshold surfaces a localized `CONSERVATION`-mode
   verdict (worst entity + severity) that feeds the same HSI repair loop.
   Reported as `p2_sketch_consistency`, kept separate from native-physics
   `p1_physics`. Honest degradation: on a non-video (mock) clip the extractor
   returns None and the oracle stays silent; a misconfigured real backend
   fails loudly rather than emitting a fake-perfect p2.
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

## Quickstart (v0.3 — CPU, no keys, no GPU)

```bash
cd Maestro
python -m venv .venv && source .venv/bin/activate

# Minimal: mock pipeline + tests (numpy + pyyaml + pytest only)
pip install -e '.[dev]'

# Recommended on a server: also get the FastAPI server + image ops
pip install -e '.[all]'                    # core + server + image
# (or `pip install -r requirements.txt` for the same set)

# unit + integration tests — should print "113 passed"
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
| `outputs/demo.report.json` | per-shot revision count, convergence, score history, **HSI `tier_used` / `escalations`**, final metrics (incl. `p2_sketch_consistency`) |
| `outputs/demo.trajectory.jsonl` | every agent decision (state/action/observation) including `replan_sketch` / `refine_spec` when HSI escalates |
| `outputs/lessons.jsonl` | cross-task experience memory (C4) — keyed on the *actually resolved* failure mode |

### Why no pixels yet
v0.1 mocks every heavy model so the *orchestration* is testable on CPU. The loop,
metrics, physics verdicts, planning validation, report and logs are all real; only
the pixel-producing step is a stub. To get real video, do v0.2 below.

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
| **physics**    | (sketch + sim wrappers; see `physics/`) | the C1 differentiation core |
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
  video_gen:  {name: mock-video-gen}   # weights_path / device for local backends
  image_edit: {name: mock-image-edit}
  # --- optional v0.3 physics backends (omit / mock = identical CPU behavior) ---
  track_extractor: {name: mock-track}  # C6 oracle: "cotracker" / "tapir" (needs torch)
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
    p1_physics:  0.22   # native physics-failure modes (C1 critic layer)
    p2_sketch_consistency: 0.10   # oracle observed-vs-expected trajectory (C6)
    id1_identity: 0.13
    m5_rhythm:   0.10
    aesthetic:   0.10

physics:
  simulator: mock              # v0.4: mujoco / newton / particle-sim
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
`physics.acceptance_severity`; sketch-tracking↑ → raise
`metrics.weights.p2_sketch_consistency`. Any value is overridable via
`--config your.yaml`. Memory persists to `<output_dir>/memory/` across runs;
delete that dir for a cold start.

---

## Going real (real pixels / real physics — GPU)

> The CPU mock pipeline above runs with no GPU and no keys. To get real outputs,
> swap a mock wrapper for a real backend (each is a stable ABC) and install that
> backend's own heavy deps (torch etc.) to match your CUDA — they are
> intentionally not pinned in this repo. Flip the matching `name` in your config.

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
- **v0.3** (current) — *memory + skill: two more innovations*. Adds two
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
  - **Physics repositioned to sketch-as-oracle** (same release line): the
    simulator became a real rigid-body integrator (ground/wall collision,
    restitution, contact events — `physics/sim_wrapper.py`); the consistency
    check became a `TrajectoryOracle` (PISA-style observed-vs-expected
    Trajectory-L2 — `physics/oracle.py`); a track-extractor factory wires
    `mock-track` (default) or real `cotracker`/`tapir`
    (`physics/track_extractor_backends.py`); an optional `world_reward`
    (`models/world_reward.py`, WMReward / V-JEPA-2) adds a `wm_reward` metric.
    See `PHYSICS_LITERATURE_REVIEW.md` for the rationale.
  - Tests: **113 passed** (CPU, ~1.8 s) — incl. skills × 6, memory tiers × 9,
    v0.3 E2E × 3, physics simulator × 6, track extractor × 9.

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

  Tests: `pytest -q` → **35 passed** (CPU, ~0.1s) at v0.2.1; →
  **54 passed** (CPU, ~0.6s) at v0.2.2 with the new tool/server tests. See
  `tests/unit/test_hsi_and_consistency.py` (v0.2.1) and
  `tests/unit/test_tools_and_act.py` + `test_server.py` (v0.2.2).
