# Maestro

**Training-free, self-improving, physically-grounded agentic video generation.**

Users give an instruction plus optional multimodal materials (video / image / music);
Maestro plans with multiple agents, generates each shot, reviews it with a board of
critics against a quantitative metric suite, and **locally repairs failing keyframes
in a monotonic self-improvement loop** — with **physics treated as a first-class
citizen**. v0.1 is a mock-first scaffold that runs end-to-end on CPU.

See `REPORT_AND_INSTRUCTIONS.md` for the full design rationale and how Maestro
differs from UniVA / VideoAgent / ViMax / VISTA / M3 / event-graph approaches.

## Four core innovations

1. **Physics as first-class (C1)** — a *sketch layer* (lightweight simulation →
   trajectory/control signal that conditions the generator) plus a *critic layer*
   that localizes failures by mode (penetration / gravity / collision / fluid /
   object-permanence / deformation / conservation) to specific frames.
2. **Keyframe-level local self-improvement (C2)** — M3's "checklist → local edit →
   monotonic Verifier → escape hatch", extended from images to video. No
   VISTA-style whole-segment regeneration.
3. **Self-loop = multi-agent review × metric suite (C3)** — a ReviewBoard of
   critics drives a quantitative, non-black-box loop.
4. **Cross-task experience memory (C4)** — a `LessonLibrary` distills failures +
   fixes and injects them into future plans ("gets better the more you use it").

## Install

```bash
cd Maestro
pip install -r requirements.txt   # numpy, pyyaml, pytest
```

## Quickstart

```bash
python scripts/run_pipeline.py \
  --prompt "a ball is thrown and bounces; a person runs through a city" \
  --music data/track.mp3 \
  --image data/hero.png \
  --output outputs/demo.mp4
```

Outputs: `outputs/demo.mp4` (mock), `outputs/demo.report.json` (per-shot revision
counts, score history, final metrics), `outputs/demo.trajectory.jsonl` (every agent
decision), `outputs/lessons.jsonl` (cross-task memory).

## Tests

```bash
pytest -q
```

## Module map

```
Inputs ─► Stage 0 Understand (AssetMemory) ─► Stage 1 Plan (Screenwriter→Director→PhysicsPlanner)
       ─► Stage 2 Generate + Self-Improve Loop (Generator ↔ ReviewBoard ↔ Verifier/Refiner, +LessonLibrary)
       ─► Stage 3 Assemble ─► video + report + trajectory
```

```
src/maestro/
  types.py            # all dataclasses / enums
  config.py logging_utils.py trajectory.py embeddings.py
  physics/            # failure_modes, sketch, sim_wrapper   ← differentiation core
  memory/             # lesson_library (C4)
  models/             # llm / mllm / video_gen / image_edit  (mock wrappers)
  agents/             # screenwriter director physics_planner generator verifier refiner
  critics/            # semantic physics consistency rhythm + board
  tools/              # metric_tool (real) assembly_tool (ffmpeg w/ fallback)
  orchestration/      # run state
  pipeline/           # understand plan generate_loop assemble run
  prompts/            # agent prompt templates (not hardcoded)
```

## Status

v0.1 = scaffold, all heavy models mocked, CPU-only, `pytest` green. v0.2 swaps mocks
for real models (DeepSeek/Qwen-VL/OmniWeaving/MuJoCo) behind the same wrapper ABCs —
**at that point API keys are needed** (see `.env.example`).
