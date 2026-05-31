# Server Run Guide

End-to-end recipe for running **LongVideoEditAgent** on a GPU server (or any
Linux box) using a CutClaw-style data layout + a single manifest.

> This doc is the operational counterpart to `README.md` (which targets a
> laptop smoke test). For *what the system does* see `docs/SYSTEM_GUIDE.md`;
> for *why CSA is differentiated* see `docs/CSA_FRAMEWORK.md`.

The data layout is intentionally aligned with
[CutClaw](https://github.com/GVCLab/CutClaw) so that anyone who has run CutClaw
can pick this up in five minutes. The differences are:

1. We do **hybrid retrieve+generation**, so each sample carries a
   `generation.*` block (CutClaw is retrieval-only).
2. Each sample may carry an **Arc context** (`intended_arc`, `energy_curve`,
   `expected_characters`) — these feed the CSA framework's whole-script judge
   and end up in the per-sample report (see `docs/CSA_FRAMEWORK.md`).
3. There is no UI dependency. Everything runs from one CLI:
   `scripts/server_run.py`.

---

## 1. Install on the server

```bash
git clone <this-repo> && cd VideoAgent
python -m venv .venv && source .venv/bin/activate

# Full v0.2 stack (perception + LLM + orchestration + music + video_gen):
pip install -e '.[all]'

# Real backends not strictly required for a smoke test — see §3.

# External system dep:
sudo apt install -y ffmpeg      # or `brew install ffmpeg` on macOS
```

GPU notes (only if you flip the real backends on):

* CUDA-capable GPU recommended for CLIP / RAFT / Qwen-VL / OmniWeaving.
  Tested on A100/H100. CPU-only works for the mocks.
* For faster video decoding, build [Decord](https://github.com/dmlc/decord)
  with NVDEC support (matches CutClaw's recommendation).
* Set `HF_HOME=/path/to/scratch/hf_cache` if your home dir is small.

API keys (only if you use real LLM / video-gen backends):

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, VLLM_BASE_URL, …
```

---

## 2. Data layout

The runner reads its inputs from anywhere you point it at — there is no hard
requirement that footage live under `data/`. The convention below mirrors
CutClaw's `resource/{video,audio,subtitle}` so you can drop CutClaw data in
unchanged:

```
data/                       # gitignored — your own footage lives here
├── videos/                 # .mp4 / .mkv source clips (any duration)
├── audio/                  # .mp3 / .wav music tracks (optional per sample)
├── subtitle/               # optional .srt to skip ASR (currently informational only)
└── character_refs/         # optional reference frames for generation conditioning
    └── joker_close_up.png
.cache/                     # gitignored — auto-populated perception cache (per sample)
outputs/server_run/         # gitignored — per-sample mp4 + trajectory + report
```

Reference samples like CutClaw's gallery (dark_knight, kyoto, interstellar,
lalaland, etc.) are not shipped with this repo — drop them into `data/videos/`
and `data/audio/` yourself and reference them from the manifest.

---

## 3. Manifest format

A *manifest* is a YAML file with two top-level keys: `defaults` (optional) and
`samples` (required). Each sample is one (video, audio, instruction) triple
plus optional Arc context, optional retrieve/generation knobs, and optional
config overrides.

The full annotated template lives at
[`configs/instructions.example.yaml`](../configs/instructions.example.yaml).
Minimal form:

```yaml
defaults:
  output_root: outputs/server_run
  cache_root: .cache

samples:
  - name: dark_knight
    sources:
      - data/videos/The_Dark_Knight.mkv
    audio: data/audio/Way_Down_We_Go.mp3
    instruction: "Joker's crazy that wants to change the world."
    arc:                                  # CSA Arc context (optional)
      intended_arc: [setup, rising, climax, falling, resolution]
      expected_characters: [joker, batman]
      energy_curve: [[0.0, 0.3], [0.5, 0.85], [1.0, 0.55]]
    generation:
      enabled: true
      fallback_threshold: 0.4
    config_overrides:                     # dotted, see §5
      compose.editor_model: "claude-sonnet-4-5"
      preprocess.parallel.num_workers: 8
    output:
      mp4: outputs/server_run/dark_knight.mp4
      # trajectory + report default to mp4 with the corresponding suffix
```

Field reference (per sample):

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Subdir-safe identifier used for cache + default outputs. |
| `sources` | yes | List of one or more video paths. Relative paths resolve against the repo root. |
| `audio` | no | Music track. If null, MusicProfile falls back to a mock 120 BPM profile. |
| `subtitle` | no | Currently informational; ASR backend selection is done via `config_overrides`. |
| `instruction` | yes | The editing prompt. Single string. |
| `arc.intended_arc` | no | List from `{setup, rising, climax, falling, resolution}` (or your own labels). |
| `arc.expected_characters` | no | Character IDs / substring matches required to appear. |
| `arc.energy_curve` | no | List of `[relative_time_0to1, target_energy_0to1]` pairs. |
| `generation.enabled` | no | Boolean. False → R-only. True → hybrid (default `compose.generation.fallback_threshold`). |
| `generation.fallback_threshold` | no | Float; see `configs/default.yaml`. |
| `config_overrides` | no | Dict of dotted paths → values; deep-merged into the pydantic Config. |
| `output.mp4` | no | Defaults to `{output_root}/{name}.mp4`. |
| `output.trajectory` | no | Defaults to `{mp4_path}.trajectory.jsonl`. |
| `output.report` | no | Defaults to `{mp4_path}.report.json`. |
| `log_level` | no | Per-sample log level. |
| `self_consistency_k` | no | Screenwriter K-of-K sampling (default 1). |
| `run_critic` | no | Run the post-hoc CriticAgent (default `true`). |

---

## 4. Run

### Batch (the common case)

```bash
python scripts/server_run.py \
    --manifest configs/instructions.example.yaml
```

Run a subset by name:

```bash
python scripts/server_run.py \
    --manifest configs/instructions.example.yaml \
    --only dark_knight_demo tiny_clip_smoke
```

### Single-shot (CutClaw `local_run.py` analogue)

```bash
python scripts/server_run.py --single \
    --video data/videos/The_Dark_Knight.mkv \
    --audio data/audio/Way_Down_We_Go.mp3 \
    --instruction "Joker's crazy that wants to change the world." \
    --output outputs/dark_knight.mp4 \
    --config.compose.generation.fallback_threshold=0.4 \
    --config.preprocess.parallel.num_workers=8
```

`--video` is repeatable for multi-source edits.

### Smoke test (no real footage required)

```bash
python scripts/server_run.py \
    --manifest configs/instructions.example.yaml \
    --only tiny_clip_smoke
```

This exercises the entire pipeline on the auto-generated tiny clip in
`tests/fixtures/tiny_clip.mp4` and produces a playable mp4 plus a JSON report.
Use it to confirm a new server install is healthy before pointing the runner
at real data.

---

## 5. Config overrides (CutClaw `--config.PARAM VALUE` analogue)

Anything in [`configs/default.yaml`](../configs/default.yaml) can be overridden
in three ways. They compose (later wins, deep-merge):

1. **Manifest globals** — `defaults.config_overrides:` block.
2. **CLI flags** — either form works:

   ```bash
   --config.compose.generation.fallback_threshold=0.3
   --config-override compose.generation.fallback_threshold=0.3
   ```

3. **Per-sample** — sample's `config_overrides:` block.

Common knobs:

| Override | Default | Effect |
|---|---|---|
| `compose.generation.enabled` | `true` | Globally enable / disable the generation pathway. |
| `compose.generation.fallback_threshold` | `0.4` | Below this feasibility, generation candidate replaces retrieval. |
| `compose.generation.backend` | `omniweaving` | One of `omniweaving | wan_local | api_veo`. |
| `compose.editor_model` | `claude-sonnet-4-5` | LLM driving the Editor ReAct loop. |
| `compose.validator_threshold` | `6.0` | MLLM judge threshold for accepting a segment. |
| `plan.models.screenwriter` | `deepseek-v3` | Screenwriter LLM. |
| `plan.max_iterations` | `5` | Plan ↔ orchestrator iteration cap. |
| `preprocess.parallel.num_workers` | `4` | Perception worker processes. |
| `mocks.perception` / `mocks.llm` / `mocks.video_gen` | `true` | Flip individual backends off the mock. |

---

## 6. Outputs

For each sample the runner writes three artifacts plus a top-level summary:

```
outputs/server_run/
├── dark_knight_demo.mp4                  # the rendered edit
├── dark_knight_demo.trajectory.jsonl     # one agent decision per line
├── dark_knight_demo.report.json          # per-sample metrics + Arc scores
├── tiny_clip_smoke.mp4
├── tiny_clip_smoke.trajectory.jsonl
├── tiny_clip_smoke.report.json
└── manifest_summary.json                 # aggregate across samples
```

The per-sample `report.json` schema:

```jsonc
{
  "name": "dark_knight_demo",
  "instruction": "Joker's crazy that wants to change the world.",
  "sources": [".../The_Dark_Knight.mkv"],
  "audio": ".../Way_Down_We_Go.mp3",
  "output_mp4": ".../dark_knight_demo.mp4",
  "trajectory": ".../dark_knight_demo.trajectory.jsonl",
  "elapsed_s": 41.2,
  "error": null,

  "n_segments": 14,
  "n_retrieval": 11,
  "n_generation": 3,
  "accepted_rate": 0.93,
  "total_duration_s": 58.4,

  "mean_m1": 0.71, "mean_m2": 0.66, "mean_m3": 0.74, "mean_m4": 0.62,
  "mean_m5": 0.68, "mean_m6": 0.59,
  "mean_validator": 7.4,

  "arc_context_provided": true,           // false → arc_* scores still computed
                                          //   from validator-trajectory shape.
  "arc": {
    "arc_progression":     0.81,
    "arc_energy_match":    0.74,
    "arc_character_cover": 1.0,
    "arc_continuity":      0.66,
    "arc_overall":         0.78
  }
}
```

The Arc-scale block is the CSA framework's whole-script judge — see
`docs/CSA_FRAMEWORK.md` for what each sub-score actually measures.

Inspect a trajectory log interactively:

```bash
python scripts/visualize_trajectory.py --log outputs/server_run/dark_knight_demo.trajectory.jsonl
```

---

## 7. The baseline sweep (optional, the v0.2 hybrid claim probe)

Once you have a sample running, you can probe whether **Hybrid** actually
beats **R-only** and **G-only** on your data by running the sweep:

```bash
python scripts/measure_baseline.py \
    --source data/videos/The_Dark_Knight.mkv \
    --output-dir outputs/baseline/dark_knight \
    --prompts "Make a 4-second high-energy montage" \
              "Make a slow emotional resolution" \
    --seeds 0 1 2
```

This sweeps `compose.generation.fallback_threshold` to coerce each of the
three regimes and writes:

```
outputs/baseline/dark_knight/
├── runs.jsonl       # 3 regimes × 2 prompts × 3 seeds = 18 lines
├── report.json      # per-regime aggregated means / stds
└── report.md        # human-readable verdict (✅ / ⚠️ / ❌)
```

See `docs/BASELINE_v0_2.md` for what the verdicts mean and when to trust them.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `source video not found: ...` | Relative path resolved against the wrong root. | Manifest paths resolve against the repo root. Use an absolute path if in doubt. |
| `ffmpeg: command not found` | System dep missing. | `apt install ffmpeg` / `brew install ffmpeg`. Required for assembly. |
| Hangs during "extracting features..." | Heavy backend without GPU. | Either flip `mocks.perception=true` in `config_overrides` or run on a CUDA box. |
| Hangs during planning. | LLM backend timing out / rate-limited. | Drop `plan.max_iterations` to 2 and watch the trajectory log to confirm which agent is stuck. |
| All segments come back retrieval. | Generation disabled or feasibility never drops below threshold. | Verify `compose.generation.enabled=true` and try `compose.generation.fallback_threshold=0.7`. |
| Arc scores are all 0.5 / look generic. | No `arc:` block in the sample. | Add at least `intended_arc:` and `expected_characters:` — see §3 field reference. |
| `manifest_summary.json: n_errors > 0` | Per-sample crashes; details in the per-sample `error` field. | Re-run with `--only <name>` + `--log-level DEBUG` for the failing sample. |
| Video codec compatibility (assembly hangs / artifact). | Same issue CutClaw flags. | Re-encode source with `ffmpeg -i in.mp4 -c:v libx264 out.mp4`. |

---

## 9. What this runner is and is not

It **is**: a thin batch wrapper around `pipeline.run.run_pipeline` that mirrors
CutClaw's UX so you can drop CutClaw-style data folders on a server, point the
manifest at them, and get a per-sample mp4 + report.json with CSA Arc scores.

It **is not**: a new pipeline, a new agent, or a new judge. Everything it
calls already exists. If a metric looks wrong, the bug is in the underlying
pipeline / Arc judge, not here. See `docs/CRITICAL_REVIEW.md` for the standing
honesty audit.
