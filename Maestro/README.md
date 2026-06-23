# Maestro

> **Training-free, multi-agent, self-improving, physically-grounded video generation.**
> Input = a natural-language instruction (+ optional source videos / reference images / a music track).
> Output = a multi-shot `.mp4`, produced by a multi-agent loop that **plans → generates → reviews → repairs → remembers**, with physics verified from the generated pixels.

This README is the **production deployment guide**: it documents the *real* backends and the *real* config to run Maestro on a server. There is no training step anywhere — every model is called at inference time only.

If you only want a key-free local healthcheck, jump to [Smoke test](#0-smoke-test-no-keys-no-gpu). For the design rationale and per-innovation source/feasibility analysis, read [`docs/INNOVATIONS_v0.4.md`](./docs/INNOVATIONS_v0.4.md).

---

## What is real vs what is a skeleton

Maestro is built mock-first: every model sits behind an ABC so a mock and a real backend are interchangeable. As of **v0.4** the following are **fully wired real backends** — flip a name in the config and supply a key, no code changes:

| Role | Real backend(s) | Module | Needs |
|---|---|---|---|
| **Planning LLM** | OpenAI · DeepSeek · Qwen · Anthropic · vLLM / any OpenAI-compatible host | `models/llm_backends.py` | an API key (or a self-hosted endpoint) |
| **VLM judge & critic** | GPT-4o · Qwen-VL (any OpenAI-compatible multimodal) | `models/mllm_backends.py` | an API key |
| **Video generation** | **WaveSpeed** hosted API (Seedance / Wan / Hailuo / Runway / … any t2v·i2v id) | `models/video_gen_backends.py` | `WAVESPEED_API_KEY` — **no local GPU** |
| **Physics track extractor (C6)** | **CoTracker** | `physics/track_extractor_backends.py` | `torch` + a CUDA GPU |
| **Assembly** | **ffmpeg** (concat demuxer + optional music mux) | `tools/assembly_tool.py` | `ffmpeg` on `$PATH` |

**Skeletons (raise a clear error until you implement the body):** OmniWeaving / Wan *local-weight* video backends, TAPIR tracker, V-JEPA-2 world-model reward, Qwen-Image-Edit keyframe editor, and the audio tool. None of these block a real run — the table above is a complete, working pipeline. Where a skeleton is selected the system **fails loudly**; it never silently falls back to a mock.

> **Honesty contract.** Critics read the *artifact* (decoded pixels / the clip record), never a revision counter. A clip a backend cannot decode yields **no** verdict rather than a fabricated one. This is enforced by a regression test (`test_loop_signal_is_content_derived`): a generator that ignores repair instructions never converges.

---

## 0. Smoke test (no keys, no GPU)

Confirms the box is wired correctly before you spend any API budget. Uses mock backends end-to-end (real `ffmpeg` assembly if present).

```bash
git clone <this-repo> && cd Maestro
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'                 # CPU-only deps: numpy, pyyaml, fastapi, opencv, imageio, pillow
maestro smoke                           # → "[smoke] OK: n_shots=3 ..."
python -m pytest -q                     # → 211 passed
```

If `maestro smoke` prints a per-shot report and `pytest` is green, the install is sound. Now wire real backends.

---

## 1. Install (server, real backends)

```bash
python -m venv .venv && source .venv/bin/activate

# Core + server + image/video decoding (CPU, pip-only):
pip install -e '.[all]'

# Real physics verification (C6) — match your CUDA. CoTracker pulls weights via
# torch.hub on first use, or point configs at a local checkpoint.
pip install torch                       # the CUDA build for your driver
pip install git+https://github.com/facebookresearch/co-tracker.git

# System dependency for final assembly:
#   Debian/Ubuntu:  apt-get install -y ffmpeg
#   macOS:          brew install ffmpeg
ffmpeg -version                         # must be on $PATH
```

GPU video deps (`torch`, `cotracker`) are intentionally **not** pinned in `pyproject.toml` so they don't fight your CUDA. Everything else is in the `all` extra.

> No GPU on the box? You can still run a fully real *generation* pipeline with **WaveSpeed video + an LLM + a VLM judge**, and set `track_extractor.name: mock-track` — physics verification then degrades to neutral (it reports "not measured", never a fake pass). The rest of the loop is unaffected.

---

## 2. Keys

```bash
cp .env.example .env        # then edit, OR just export the vars in your shell
```

Set only the keys for the backends you select in the config. The minimal real setup (single provider) needs **two** keys:

```bash
export QWEN_API_KEY=...        # planning LLM + VLM judge (one provider, two roles)
export WAVESPEED_API_KEY=...   # real video pixels
```

All recognized variables (full list in [`.env.example`](./.env.example)):

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `llm`/`mllm` name `openai` / `gpt-4o` |
| `DEEPSEEK_API_KEY` | `llm` name `deepseek` |
| `QWEN_API_KEY` | `llm` name `qwen`, `mllm` name `qwen-vl` |
| `ANTHROPIC_API_KEY` | `llm` name `anthropic` / `claude*` |
| `LLM_API_KEY` / `LLM_BASE_URL` | `llm` name `vllm` / `openai-compat` (self-hosted) |
| `WAVESPEED_API_KEY` | `video_gen` name `wavespeed` |
| `COTRACKER_CKPT` | optional CoTracker checkpoint path (else torch.hub) |
| `MAESTRO_HOST` / `MAESTRO_PORT` / `MAESTRO_LOG_LEVEL` | the server |

A real backend selected **without** its key raises a `RuntimeError` at call time — it will not quietly run on mocks.

---

## 3. Config

The shipped [`configs/default.yaml`](./configs/default.yaml) is **all-mock** (so the smoke test and CI need no keys). For a real run use [`configs/server.yaml`](./configs/server.yaml), which wires every backend above. Its default is single-provider Qwen + WaveSpeed + CoTracker; mix-and-match alternatives are in its comments.

The model block (the part you change most):

```yaml
models:
  llm:                              # planning brain
    name: "qwen"                    # or deepseek / openai / anthropic / vllm
    model: "qwen-plus"
  mllm:                             # VLM judge & critic
    name: "qwen-vl"                 # or gpt-4o
    model: "qwen-vl-max"
    n_frames: 4                     # frames sampled from each clip to judge
  video_gen:
    name: "wavespeed"               # REAL pixels, no local GPU
    model_id: "bytedance/seedance-v1-pro-t2v-480p"
  track_extractor:
    name: "cotracker"               # REAL physics verification (needs torch+GPU)
    device: "cuda"
```

**Backend → key → endpoint dispatch:**

| `models.llm.name` | key | default endpoint · model |
|---|---|---|
| `openai`, `gpt*` | `OPENAI_API_KEY` | `api.openai.com/v1` · `gpt-4o` |
| `deepseek` | `DEEPSEEK_API_KEY` | `api.deepseek.com/v1` · `deepseek-chat` |
| `qwen` | `QWEN_API_KEY` | `dashscope.aliyuncs.com/compatible-mode/v1` · `qwen-plus` |
| `anthropic`, `claude*` | `ANTHROPIC_API_KEY` | `api.anthropic.com` · `claude-sonnet-4-6` |
| `vllm` / `openai-compat` | `LLM_API_KEY` (opt) | `LLM_BASE_URL` (e.g. `localhost:8000/v1`) · *your served id* |

| `models.mllm.name` | key | default endpoint · model |
|---|---|---|
| `gpt-4o`, `openai-vlm` | `OPENAI_API_KEY` | `api.openai.com/v1` · `gpt-4o` |
| `qwen-vl` | `QWEN_API_KEY` | `dashscope.aliyuncs.com/compatible-mode/v1` · `qwen-vl-max` |

Other tunables in `server.yaml`: `compose.{fps,n_candidates,max_revisions,k_retries}` (the best-of-N + self-improve budget), `metrics.weights` (the 7-dim review suite — **an explicit block overrides the code defaults wholesale, so list every dimension**), `physics.{violation_threshold,post_accept_strictness}`, and the `memory.*` tier switches.

---

## 4. Run

### One-shot (CLI / cron / batch)

```bash
maestro run-once \
  --prompt "A red kite breaks free and tumbles down onto wet pavement, slow motion" \
  --config configs/server.yaml \
  -o outputs/kite.mp4
```

With source material (retrieval-grounded generation; identity anchors keep a character consistent across shots):

```bash
maestro run-once \
  --prompt "Stitch the best action beats into a 3-shot montage on the beat" \
  --source footage/clip1.mp4 footage/clip2.mp4 \
  --image refs/hero.png \
  --music track.mp3 \
  --config configs/server.yaml \
  -o outputs/montage.mp4
```

`--source` (1+ videos), `--image` (reference/identity images), `--music` (a track) are all optional.

### Outputs

| File | Contents |
|---|---|
| `outputs/kite.mp4` | the finished multi-shot video (real ffmpeg concat of generated clips) |
| `outputs/kite.report.json` | per-shot revisions, HSI `tier_used` / `escalations`, final metric scores, `entities` / `transitions_committed` / `skills_learned` / `lessons_learned` |
| `outputs/kite.trajectory.jsonl` | every agent decision (state · action · observation) — `annotate_physics`, `generate`, `review`, `verify`, `replan_physics`, `validate_plan` … |
| `outputs/memory/` | persistent library: distilled **skills**, **entity** identity registers + state-transition log, **lessons**, preferences — compounds across runs |

Point a stable `memory_dir` (the directory above) at all of one user's jobs and the skill library + character banks accumulate run over run.

---

## 5. Server

```bash
maestro serve --port 8000          # or: uvicorn maestro.server:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | UniVA-compatible liveness (`{status, service, version, n_tools}`) |
| `GET /tools` | the registered tool inventory |
| `POST /generate` | submit a job → `{job_id, state}` (async; the run happens in a worker thread) |
| `GET /jobs/{job_id}` | poll job state + result paths |

```bash
curl -s localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"a paper boat drifts down a rain gutter","images":["refs/boat.png"]}'
# → {"job_id":"...","state":"running"}
curl -s localhost:8000/jobs/<job_id>      # poll until state == "done"
```

The server reads the same config/env. Set `MAESTRO_SANDBOX=1` to make the tool-executing agent refuse side-effecting tools (useful for untrusted prompts).

### Docker (CPU host; WaveSpeed for pixels, no GPU)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install -e '.[all]'
EXPOSE 8000
CMD ["maestro", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t maestro:0.4 .
docker run --rm -p 8000:8000 \
  -e QWEN_API_KEY -e WAVESPEED_API_KEY \
  -v "$PWD/outputs:/app/outputs" maestro:0.4
```

For real physics verification add a CUDA base image + `pip install torch cotracker` and run with `--gpus all`; otherwise keep `track_extractor.name: mock-track`.

---

## 6. How a run flows

```
instruction (+ source / images / music)
        │
  Stage 0  Understanding ── perceive source assets → AssetMemory (identity / style / music)
        │
  Stage 1  Plan ───────── Screenwriter → Director → PlanValidator (Critique·Correct·Verify)
        │                  → ShotSpecs, each annotated with physics (entities + motion class)
        │
  Stage 2  Generate + Self-Improve (per shot, the core loop)
        │     best-of-N candidates  →  Review Board (semantic · physics VLM · law verifier
        │     · consistency · rhythm) + metric suite  →  Verifier (monotonic)
        │       repair, cheapest scope first:
        │         Tier 0  keyframe-local edit
        │         Tier 1  regenerate with verdict-derived anti-violation hints (unchanged bar)
        │         Tier 2  director rewrites the shot spec
        │         Tier 3  escape hatch (logged, accounted)
        │       → distill a Lesson + (if verified) a Skill; commit verified entity-state writes
        │
  Stage 3  Assemble ───── ffmpeg concat of accepted clips (+ music) → final .mp4
```

The five **innovations** (full sourcing & feasibility in [`docs/INNOVATIONS_v0.4.md`](./docs/INNOVATIONS_v0.4.md)):

- **Unified skill lifecycle** — creation / review / memory skills under one training-free *distill → admission ("skill CI") → retrieve → EMA → evolve/evict* loop; the agent's only learnable substrate.
- **Reference-free physics-from-pixels (C6)** — no sketch, no simulator. A tracker recovers observed motion; a reliability gate certifies it (trackers lie on generated video); the law layer asks *"is there any physically consistent explanation?"* (best passive-law residual + localized anomaly detection), driving best-of-N selection and targeted repair.
- **Dual-register entity memory** — immutable canonical identity ⊕ mutable state via a typed transition log, with **verification-gated writes**: a state change commits only when confirmed in the rendered clip.
- **Hierarchical Self-Improvement (HSI)** — adaptive-scope repair (Tier 0→3) with a monotonic verifier and cross-task lesson reuse.
- **Real tool/asset spine** — UniVA-style tool registry, retrieval-grounded generation, hosted-API video so a real run needs no local GPU.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: ...needs an API key` | a real backend is selected but its env var is unset — export it or switch the name to `mock-*`. |
| Final `.mp4` exists but isn't playable | `ffmpeg` not on `$PATH`, or the video backend returned non-video bytes — the assembler logged `mock assembly`. Install ffmpeg; confirm `video_gen.name: wavespeed` + a valid `WAVESPEED_API_KEY`. |
| `p2_law_consistency` is always `1.0` | physics not measured: `track_extractor.name` is `mock-track`, or torch/CoTracker isn't installed, or the clip didn't decode. This is "not verified", not "verified perfect" — wire CoTracker on a GPU to get a real signal. |
| VLM judge returns no verdicts | the clip couldn't be decoded into frames (install `opencv-python-headless` / `imageio[ffmpeg]`, already in `.[all]`), or the model returned unparseable JSON (logged as a warning). |
| WaveSpeed `TimeoutError` | raise `video_gen.timeout`; check the model id exists at wavespeed.ai. |
| Server returns 422 on `/generate` | send the body as JSON (`-H 'content-type: application/json'`), not query params. |
| `pip install -e '.[all]'` then `ModuleNotFoundError: maestro` | transient pip/network failure mid-install — re-run the install; `pythonpath=["src"]` is already set for pytest. |

---

## 8. Tests & layout

```bash
python -m pytest -q          # 211 passed — CPU-only, no keys, no network
```

```
src/maestro/
├── cli.py  config.py  server.py          # entry points
├── agents/        screenwriter · director · physics_planner · generator · refiner · verifier · act
├── critics/       semantic · physics (VLM) · physics_consistency (law verifier) · consistency · rhythm · board · tournament
├── physics/       annotate · router · tracks · reliability · laws · verifier · track_extractor_backends   (C6, reference-free)
├── memory/        skill_library · skill_admission · entity_store · write_gate · multi_layer · lesson_library
├── models/        llm · llm_backends · mllm · mllm_backends · video_gen(+backends) · world_reward · image_edit · mock_signals
├── tools/         registry + retrieval · assembly · metric · captioning · detection · audio_gen · …
└── pipeline/      understand · plan · generate_loop · assemble · run
configs/   default.yaml (mock) · server.yaml (real)
docs/      INNOVATIONS_v0.4.md · DATAFLOW.md · COMPARISON.md · research/ (surveys + plan + UniVA map)
```

---

## License

See [LICENSE](./LICENSE).
