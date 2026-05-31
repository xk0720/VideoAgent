# LongVideoEditAgent (VideoAgent)

> Instruction-driven long-video editing agent.
> Implements the v0.1 scaffold described in [`LongVideoEditAgent_DESIGN.md`](./LongVideoEditAgent_DESIGN.md).

A research-grade Python framework that takes **(1+ source videos, a natural-language editing prompt, an optional music track)** and returns a finished edit. Editing is performed by a small multi-agent system that jointly **retrieves** shots from the sources and **generates** missing shots with a video diffusion model, sharing a four-layer narrative memory (shot ⇢ event ⇢ story ⇢ character).

The repository is organised around the design doc; this README only tells you how to install and run the v0.1 mock pipeline. For everything else, see the index below.

## Where to read next

| Want to ... | Open |
|---|---|
| just run it (laptop smoke test) | this README, §Quickstart |
| **run it on a server with real footage (CutClaw-style data layout + batch manifest)** | [`docs/SERVER_RUN.md`](./docs/SERVER_RUN.md) ⭐ |
| understand the data flow & the agentic-evolution / self-loop-evaluation design | [`docs/SYSTEM_GUIDE.md`](./docs/SYSTEM_GUIDE.md) ⭐ |
| see how every section of the design doc maps to source files | [`docs/architecture_tour.md`](./docs/architecture_tour.md) |
| know which open-source library backs which module | [`docs/dependencies.md`](./docs/dependencies.md) |
| see the design decisions we made (and *didn't* make) | [`docs/decisions.md`](./docs/decisions.md) |
| read the original blueprint | [`LongVideoEditAgent_DESIGN.md`](./LongVideoEditAgent_DESIGN.md) |
| **see an honest accounting of what 14 rounds actually delivered** | [`docs/CRITICAL_REVIEW.md`](./docs/CRITICAL_REVIEW.md) ⭐ |
| **the differentiated framework: Cut · Score · Arc** | [`docs/CSA_FRAMEWORK.md`](./docs/CSA_FRAMEWORK.md) ⭐ |
| see the v0.2 baseline measurement plan (hybrid claim probe) | [`docs/BASELINE_v0_2.md`](./docs/BASELINE_v0_2.md) |

---

## Architecture at a glance

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Stage 1      │   │ Stage 2      │   │ Stage 3      │
│ Understanding│ → │ Planning     │ → │ Compose      │
│ (offline)    │   │ (multi-agent)│   │ (retrieve+gen│
│ → memory     │   │ → script     │   │ → mp4)       │
└──────────────┘   └──────────────┘   └──────────────┘
   perception/         agents/             tools/
   memory/             orchestration/      models/
```

| Layer | Real open-source backbone (v0.1 mocked, v0.2 plug in) |
|---|---|
| Shot segmentation | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (`scenedetect`) |
| Visual features | [open_clip](https://github.com/mlfoundations/open_clip) / HuggingFace `transformers` CLIPModel |
| Optical flow | `torchvision.models.optical_flow.raft_large` (PyTorch RAFT) |
| Saliency | [U²-Net](https://github.com/xuebinqin/U-2-Net) |
| Shot captioning | Qwen-VL / GPT-4o via HuggingFace `transformers` or OpenAI SDK |
| Cinematography tags | [ShotVL / ShotBench](https://huggingface.co/Vchitect) |
| Music structure | [All-In-One](https://github.com/mir-aidj/all-in-one) (`allin1`) |
| Character ID | [InsightFace](https://github.com/deepinsight/insightface) |
| Memory index | SQLite (stdlib) + [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| Multi-agent graph | [LangGraph](https://langchain-ai.github.io/langgraph) (`langgraph` + `langchain-core`) |
| LLM clients | [OpenAI SDK](https://github.com/openai/openai-python), [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python), DeepSeek (OpenAI-compatible), vLLM (OpenAI-compatible) |
| Video generation | [OmniWeaving](https://github.com/Tencent-Hunyuan/OmniWeaving) (preferred), Wan2.6, Veo API |
| Assembly | [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) (real, not mocked) |
| Logging | [loguru](https://github.com/Delgan/loguru) |
| Config | [pydantic v2](https://docs.pydantic.dev) + [PyYAML](https://pyyaml.org) |

A complete map (per-file ↔ dependency ↔ mock status) lives in [`docs/dependencies.md`](./docs/dependencies.md).

---

## Install

```bash
git clone <this-repo> && cd VideoAgent
python -m venv .venv && source .venv/bin/activate

# v0.1 mock pipeline only (CPU, ~10 packages, ~2 min):
pip install -e .

# Add real backends as you flip them on:
pip install -e '.[perception,llm,orchestration,music,video_gen]'
# or everything at once:
pip install -e '.[all]'

cp .env.example .env  # fill in API keys if you'll exercise real LLMs
```

External system dependency: `ffmpeg` must be on `$PATH`. (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu.)

---

## Quickstart (v0.1 mock pipeline, CPU-only)

```bash
# 1. Offline preprocessing of a source video → narrative memory (mock perception)
python scripts/preprocess_video.py \
    --source tests/fixtures/tiny_clip.mp4 \
    --cache-dir ./.cache/demo \
    --config configs/default.yaml

# 2. End-to-end pipeline (mocks Stage 2 LLM calls, mocks Stage 3 generation,
#    runs real ffmpeg assembly → ./outputs/demo.mp4 is a genuine playable file)
python scripts/run_pipeline.py \
    --cache-dir ./.cache/demo \
    --user-prompt "Make a 10-second highlight reel with high energy" \
    --output ./outputs/demo.mp4 \
    --trajectory-log ./outputs/demo_trajectory.jsonl

# 3. Inspect agent decisions
python scripts/visualize_trajectory.py --log ./outputs/demo_trajectory.jsonl
```

Run the test suite (all mocked, no GPU, no network):

```bash
pytest tests/ -q
```

---

## Repository layout

See [`LongVideoEditAgent_DESIGN.md` §3](./LongVideoEditAgent_DESIGN.md#3-仓库结构) for the complete tree.
The TL;DR:

```
src/longvideoagent/
├── types.py              # all cross-module dataclasses (Shot, Event, Story, ...)
├── config.py             # pydantic-based config loader
├── memory/               # SQLite + FAISS narrative memory
├── perception/           # Stage-1 wrappers (mock-first)
├── agents/               # Screenwriter / Director / Orchestrator / Editor / Validator
├── tools/                # Retrieval / Generation / Assembly / Metric tools
├── models/               # LLM, video-gen, reward-model wrappers
├── orchestration/        # LangGraph-style state machine
├── pipeline/             # preprocess / plan / compose / run
├── prompts/              # all agent prompts as plain text
└── utils/                # video/audio I/O + JSONL trajectory logger
```

---

## What works today (v0.1 acceptance, design doc §11)

- [x] `pytest tests/unit/` green
- [x] `pytest tests/integration/test_end_to_end.py` green (uses an auto-generated `tiny_clip.mp4`)
- [x] `scripts/run_pipeline.py` produces a real `.mp4` file plus structured JSONL trajectory log
- [x] All agent prompts live in `prompts/` as plain text — none hardcoded
- [x] Config is loaded from `configs/default.yaml` via pydantic; runtime overrides reflected
- [x] No CPU-only test or script imports torch, faiss, transformers, or any GPU-dependent library

## What is intentionally mocked (will be swapped in v0.2)

Perception backends (CLIP, RAFT, U²-Net, Qwen-VL, ShotVL, InsightFace, All-In-One), all LLM calls (default agent backend returns canned outputs), and the video-generation client. The wrappers expose the exact same public API you'll need when you swap a real implementation in — see each module's top-of-file docstring for the upstream library it stubs.

---

## License

See [LICENSE](./LICENSE).
