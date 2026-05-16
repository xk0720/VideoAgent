# Architecture Tour

A point-by-point map of [`LongVideoEditAgent_DESIGN.md`](../LongVideoEditAgent_DESIGN.md)
to the actual source files. Read this side-by-side with the design doc when
you're about to extend a stage.

---

## §0–§1 — TL;DR & Engineering principles

| Design doc concept | Where it lives |
|---|---|
| Three-stage pipeline (Understanding / Planning / Compose) | `src/longvideoagent/pipeline/{preprocess,plan,compose,run}.py` |
| YAML-driven config | `configs/default.yaml` + `src/longvideoagent/config.py` |
| Offline/online separation | `pipeline/preprocess.py` (offline), `pipeline/{plan,compose}.py` (online) |
| Module-as-replaceable wrappers | `models/llm/` & `models/video_gen/` factories, `mocks_enabled` switch in `Config.mocks` |
| Typed interfaces, no `dict` fallbacks | `src/longvideoagent/types.py` (dataclasses, single source of truth) |
| Structured trajectory log | `utils/trajectory.py` |

## §2 — System overview

The ASCII diagram in §2 is implemented as:

| ASCII box | Code |
|---|---|
| `Inputs` block | CLI args parsed by `scripts_impl.run_pipeline_main` → `pipeline.run.run_pipeline(...)` |
| `Stage 1 · Understanding` | `pipeline.preprocess.preprocess()` (calls every `perception/*.py`) → `memory.store.MemoryStore` |
| `Stage 2 · Planning` | `pipeline.plan.plan()` → builds `orchestration.graph` over the three planning agents |
| `Stage 3 · Compose` | `pipeline.compose.compose()` → `agents.editor.EditorAgent` ReAct loop → `tools.assembly_tool.AssemblyTool` |
| `Trajectory Log` arrow | `utils.trajectory.TrajectoryLogger` (JSONL, fed by `BaseAgent.log_step`) |

## §3 — Repo layout

Top-level mapping (every file in the design doc exists at the same path):

```
configs/                                 ⇄  configs/{default,agents/*,models/*,heuristics/*}.yaml
src/longvideoagent/types.py             ⇄  global dataclasses
src/longvideoagent/config.py            ⇄  loader (dataclass-based, see D-012)
src/longvideoagent/logging.py           ⇄  loguru + stdlib fallback
src/longvideoagent/memory/              ⇄  4-file SQLite/FAISS narrative memory
src/longvideoagent/perception/          ⇄  9 mock-first wrappers
src/longvideoagent/agents/              ⇄  6 BaseAgent subclasses (5 online + CriticAgent post-hoc)
src/longvideoagent/tools/               ⇄  Retrieval/Generation/Assembly/Metric
src/longvideoagent/models/              ⇄  LLM, video-gen, reward wrappers (+ EnsembleRewardModel)
src/longvideoagent/memory/              ⇄  4-file narrative memory + LessonBook (v0.2)
src/longvideoagent/orchestration/       ⇄  graph + state + messages
src/longvideoagent/pipeline/            ⇄  preprocess/plan/compose/run drivers
src/longvideoagent/prompts/             ⇄  7 plain-text prompt files
src/longvideoagent/utils/               ⇄  trajectory + preferences (DPO/IPO/KTO/SimPO/GRPO-ready) + video/audio I/O
benchmark/                              ⇄  Mashup/CineBench adapters (v0.2)
scripts/                                ⇄  5 CLI entry points
tests/                                  ⇄  unit + integration (87 tests)
```

## §4 — Core data structures

Every dataclass in §4 is defined verbatim in `src/longvideoagent/types.py`:

| §4 dataclass | File line  (search `class <name>`) |
|---|---|
| `CinematographyTags` | `types.py` |
| `ShotFeatures` | `types.py` |
| `Shot` / `Event` / `Story` / `Character` | `types.py` |
| `NarrativeMemory` (+ `.summarize()`) | `types.py` |
| `MusicSection` / `MusicProfile` | `types.py` |
| `GlobalStructuralPlan` / `SectionPlan` | `types.py` |
| `SegmentGuidance` | `types.py` |
| `EditingSegment` / `EditingScript` | `types.py` |
| `AgentStep` | `types.py` |

## §5 — Stage interfaces

| §5 spec | Implementation |
|---|---|
| `preprocess(source_videos, music, cache_dir, config) -> NarrativeMemory` | `pipeline/preprocess.py::preprocess` |
| `plan(memory, user_prompt, config, trajectory_logger) -> list[SegmentGuidance]` | `pipeline/plan.py::plan` |
| `compose(memory, guidances, config, trajectory_logger) -> EditingScript` | `pipeline/compose.py::compose` |
| Rolling 10-shot caption buffer | `perception/captioner.py::ShotCaptioner.buffer` (deque) |
| LangGraph state schema `PlanState` | `orchestration/state.py::PlanState` |
| Conditional re-plan edge | `orchestration/graph.py::_fallback_run` |

## §6 — Agent specifications

| §6.x agent | File | Reference (older + 2024-2025) |
|---|---|---|
| `BaseAgent` ABC | `agents/base.py` | — |
| `ScreenwriterAgent` (DIRECT §4.1) | `agents/screenwriter.py` + prompt `prompts/screenwriter.txt` | DIRECT (2024); v0.2 adds **Self-Consistency** (Wang et al., ICLR 2023) + **rStar** (Microsoft, 2024) |
| `DirectorAgent` (DIRECT §4.2, 3 CoT steps + feasibility) | `agents/director.py` + 3 prompts | DIRECT (2024) |
| `OrchestratorAgent` (CineAgents iterative validation) | `agents/orchestrator.py` + `prompts/orchestrator_validate.txt` | CineAgents (2024) |
| `EditorAgent` (multi-step ReAct) | `agents/editor.py` + `prompts/editor_summary.txt` | **ReAct** (Yao et al., ICLR 2023); **GLANCE** (2024) bi-loop |
| `ValidatorAgent` (MLLM judge) | `agents/validator.py` + `prompts/reward_judge.txt` | **G-Eval** (2023); v0.3 → **Tülu-3-RM** (Nov 2024), **Skywork-Reward** (Oct 2024), **JudgeLM** (2024) |
| **`CriticAgent` (v0.2)** — post-hoc trajectory reviewer | `agents/critic.py` | **Reflexion** (Shinn et al., 2023); 2024 successors: **Trace** (Microsoft), **rStar** (Microsoft), **AFlow** (Zhang et al.) |

## §7 — Tool specifications

| §7.x tool | File | Real OSS library |
|---|---|---|
| RetrievalTool (beam search + sliding window) | `tools/retrieval_tool.py` | numpy + scipy (with numpy fallback) |
| GenerationTool | `tools/generation_tool.py` | wraps `models.video_gen.BaseVideoGenClient` |
| AssemblyTool | `tools/assembly_tool.py` | ffmpeg-python + system ffmpeg |
| MetricTool (m1..m6) | `tools/metric_tool.py` | scipy.stats with numpy fallback |

The six metrics m1..m6 in §7.1 are public functions in `tools/metric_tool.py`.

## §8 — Model wrappers

| §8.x wrapper | File | PyPI / Reference |
|---|---|---|
| LLM ABC | `models/llm/base.py` | `openai`, `anthropic` (lazy) |
| OpenAI / DeepSeek / vLLM | `models/llm/openai_client.py` + `deepseek_client.py` + `vllm_local.py` | `openai` |
| Anthropic | `models/llm/anthropic_client.py` | `anthropic` |
| Video gen ABC + 4 backends | `models/video_gen/{base,omniweaving,wan_local,api_client}.py` | **HunyuanVideo** (Dec 2024) · **CogVideoX-5B** (Aug 2024) · **Mochi-1** (Oct 2024) · **LTX-Video** (Dec 2024) · OmniWeaving · Wan2.x · **Veo 2** (Dec 2024) |
| Reward ABC + MLLM judge | `models/reward/{base,mllm_judge}.py` | uses the LLM wrappers; v0.3 → Tülu-3-RM, Skywork-Reward |
| **EnsembleRewardModel (v0.2)** | `models/reward/ensemble.py` | **Multi-Agent Debate** (Du et al. 2023), **DyLAN** (Liu et al. 2024), **MJ-Bench** (2024), **JudgeLM** (2024) |

## §9 — Config

`configs/default.yaml` is the canonical config. Sub-configs live in
`configs/agents/`, `configs/models/`, `configs/heuristics/`.

| §9 YAML key | Pydantic-style dataclass |
|---|---|
| `preprocess.shot_detector` etc. | `config.PreprocessCfg` |
| `plan.*` | `config.PlanCfg` |
| `compose.metric_weights` | `config.MetricWeightsCfg` |
| `compose.retrieval` / `.generation` | `config.RetrievalCfg` / `GenerationCfg` |
| `assembly.*` | `config.AssemblyCfg` |
| `mocks.*` | `config.MocksCfg` |

## §10 — CLI

| §10 command | Console script | Implementation |
|---|---|---|
| `preprocess_video.py` | `lva-preprocess` | `scripts_impl.preprocess_main` |
| `build_memory.py` | `lva-build-memory` | `scripts_impl.build_memory_main` |
| `run_pipeline.py` | `lva-run` | `scripts_impl.run_pipeline_main` |
| `eval_benchmark.py` | `lva-eval` | `scripts_impl.eval_main` |
| `visualize_trajectory.py` | `lva-viz` | `scripts_impl.viz_trajectory_main` |

## §11 — v0.1 acceptance

| Acceptance criterion | Verified by |
|---|---|
| `pytest tests/unit/` green | All unit tests under `tests/unit/` (47+ tests) |
| `pytest tests/integration/test_end_to_end.py` green | `tests/integration/test_end_to_end.py` |
| Real .mp4 + trajectory.jsonl from `scripts/run_pipeline.py` | `outputs/demo.mp4` + `outputs/demo_trajectory.jsonl` |
| Prompts in `prompts/` plain text | `src/longvideoagent/prompts/*.txt` |
| Config from `configs/default.yaml` reflected at runtime | `tests/unit/test_config.py::test_overrides_merge` |
| README with quickstart + module diagram | `README.md` |

## §12 — v0.2 roadmap

Each path is wired but mocked today:

| §12 roadmap item | Current scaffolding |
|---|---|
| Real OmniWeaving / Wan / Veo | `models/video_gen/{omniweaving,wan_local,api_client}.py` (raise `NotImplementedError`) |
| Mashup-Bench / CineBench | `benchmark/{mashup_bench,cine_bench}.py` (raise `NotImplementedError`) |
| Trajectory → RL training data | `utils/trajectory.py` (JSONL format compatible with RAGEN) |
| Fine-tuned RM | `models/reward/mllm_judge.py` (interface stable, swap inner `mllm` client) |

## §13 — Execution priority

Implementation order followed §13's checklist:

1. `types.py` ✓
2. `config.py` ✓ (dataclass-based per D-012)
3. `logging.py` + `utils/trajectory.py` ✓
4. `memory/{schema,store,builder,retriever}.py` ✓
5. `perception/*` (mock) ✓
6. `pipeline/preprocess.py` ✓
7. `agents/base.py` + `tools/base.py` ✓
8. `prompts/*.txt` ✓
9. `orchestration/{state,graph}.py` ✓
10. `agents/{screenwriter,director,orchestrator}.py` ✓
11. `tools/retrieval_tool.py` (beam + sliding window) ✓
12. `tools/generation_tool.py` (mock, but dynamic metric_scores) ✓
13. `tools/assembly_tool.py` (real ffmpeg) ✓
14. `agents/{editor,validator}.py` ✓
15. `pipeline/run.py` ✓
16. `scripts/*` + `tests/*` ✓

## §14 — Open-source references

See [`docs/dependencies.md`](./dependencies.md) for the canonical mapping of
each upstream project to the file that wraps it.

## §15 — Anti-patterns avoided

| Anti-pattern | How we avoided it |
|---|---|
| Single agent doing many jobs | DirectorAgent split into 3 LLM calls (`director_{query,heuristic,pacing}.txt`) |
| LLM in hot loop | `compose.max_editor_steps = 10` cap; auto-validate avoids redundant LLM-controlled validate |
| Whole NarrativeMemory in agent state | Agents receive a `MemoryStore` handle; only summaries are inlined |
| Hardcoded prompts | Every prompt is a `.txt` file in `src/longvideoagent/prompts/` |
| Video-gen orchestration in agent code | All scheduling logic lives inside `BaseVideoGenClient` subclasses |
| GPU-dependent v0.1 | `mocks.*` defaults to true everywhere; tests run on CPU only |
