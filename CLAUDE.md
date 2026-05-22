# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LongVideoEditAgent — an instruction-driven long-video editing agent. Takes source videos + a natural-language editing prompt (+ optional music) and produces a finished edit via a multi-agent system that retrieves shots from sources and generates missing shots with video diffusion models. Currently at v0.1 (mock pipeline); real model backends are optional extras.

## Common Commands

```bash
# Install (v0.1 mock pipeline, CPU-only)
pip install -e .

# Install with all backends + dev tools
pip install -e '.[all]'

# Run full test suite (no GPU, no network required)
make test                   # or: PYTHONPATH=src python -m pytest tests/ -q

# Run only unit / integration tests
make test-unit
make test-integration

# Run a single test file or test
PYTHONPATH=src python -m pytest tests/unit/test_memory.py -q
PYTHONPATH=src python -m pytest tests/unit/test_memory.py::test_function_name -q

# Lint
make lint                   # runs pyflakes on src/ tests/ benchmark/ scripts/

# End-to-end demo (generates tiny_clip.mp4 fixture, runs pipeline, outputs .mp4)
make demo

# Generate test fixture manually
make fixture
```

**External dependency:** `ffmpeg` must be on `$PATH`.

## Architecture

Three-stage pipeline: **Understanding → Planning → Compose**

### Stage 1: Understanding (offline preprocessing)
- `src/longvideoagent/perception/` — shot detection, CLIP features, optical flow, saliency, captions, cinematography tags, music analysis, character ID
- `src/longvideoagent/memory/` — four-layer narrative memory (shot → event → story → character) backed by SQLite + FAISS

### Stage 2: Planning (multi-agent)
- `src/longvideoagent/agents/` — five agents: Screenwriter, Director, Orchestrator, Editor, Validator (+ Critic for post-hoc reflection)
- `src/longvideoagent/orchestration/` — LangGraph state-machine (with Python fallback when langgraph not installed)
- Planning loop: Screenwriter drafts → Director/Orchestrator iterate (up to `max_iterations`) until validated

### Stage 3: Compose (Editor ReAct loop)
- `src/longvideoagent/tools/` — Retrieval, Generation, Assembly, Metric tools
- `src/longvideoagent/models/video_gen/` — OmniWeaving (preferred), Wan2.6 local, Veo API
- `src/longvideoagent/models/llm/` — OpenAI, Anthropic, DeepSeek, vLLM clients (all behind `BaseLLMClient` ABC)

### Cross-cutting
- `src/longvideoagent/config.py` — dataclass-based config loader; mirrors `configs/default.yaml`; supports env var overrides (`LVA_CACHE_ROOT`, `LVA_OUTPUT_ROOT`)
- `src/longvideoagent/prompts/*.txt` — all agent prompts as plain text files (never hardcoded)
- `src/longvideoagent/utils/trajectory.py` — JSONL trajectory logger for agent decisions
- `src/longvideoagent/memory/lessons.py` — cross-run LessonBook for agentic self-improvement

### Entry points
- CLI commands: `lva-preprocess`, `lva-build-memory`, `lva-run`, `lva-eval`, `lva-viz` (defined in `src/longvideoagent/cli.py`)
- Scripts: `scripts/preprocess_video.py`, `scripts/run_pipeline.py`, `scripts/build_memory.py`, `scripts/eval_benchmark.py`, `scripts/visualize_trajectory.py`

## Key Design Constraints

- **Mock-first:** All perception/LLM/video-gen backends are mocked in v0.1 (`configs/default.yaml` → `mocks:` section). No CPU-only test should import torch, faiss, transformers, or GPU-dependent libraries.
- **Prompt templates on disk:** Agent prompts must live in `src/longvideoagent/prompts/*.txt`, not inline in code.
- **Config via dataclasses:** `Config` tree in `config.py` mirrors `configs/default.yaml`. Add new fields to both.
- **LLM clients are injected:** Agents receive a `BaseLLMClient` — tests pass `MockLLMClient`.
- **ffmpeg assembly is real** even in v0.1 (not mocked).

## Testing

- Tests run with `PYTHONPATH=src` (configured in `pyproject.toml`).
- `tests/fixtures/tiny_clip.mp4` is auto-generated on first test run (no binary in git).
- Markers: `@pytest.mark.slow` (real models/APIs), `@pytest.mark.integration`.
- CI runs on Python 3.10, 3.11, 3.12 with base install only.

## Code Style

- Ruff: line-length 110, target Python 3.10, select `["E", "F", "I", "B", "UP"]`, E501 ignored.
- Type annotations used throughout; `mypy` configured with `ignore_missing_imports = true`.
