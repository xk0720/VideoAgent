# LongVideoEditAgent — 长视频剪辑 Agent 框架设计文档

> 给 Claude Code 看的项目搭建说明书。
> 目标：搭出一个可运行的 Python 框架骨架，后续基于此做实验和迭代。
> 工期定位：v0.1（脚手架 + 跑通端到端 dummy pipeline）= 1 周；v0.2（接入真实模型）= 2-3 周。

---

## 0. TL;DR — 一段话讲清楚要做什么

构建一个 **指令驱动的长视频剪辑 Agent 系统**，输入是「一/多段长视频源 + 用户自然语言 prompt + 可选音乐」，输出是「一段剪辑好的视频」。

剪辑过程通过 **多 agent 协作 + agent 思维链中联合调度 retrieval（从源视频找镜头）和 generation（用视频生成模型补素材）** 完成，整个流程共享一份 hierarchical narrative memory。

整套系统由三大子系统组成：

1. **Understanding** — 离线预处理源视频，构建多层次结构化记忆
2. **Planning** — 多 agent 协作把用户 prompt 转成具体剪辑脚本（编辑指令序列）
3. **Compose** — 联合调用 retrieval + video generation tools 落地脚本，产出最终视频

后续可选扩展：(a) 用 agentic RL 优化 agent planning 轨迹；(b) 微调专用 reward model 给 closed-loop validation 打分。

---

## 1. 设计原则（请 Claude Code 始终遵循）

### 1.1 工程原则
- **配置驱动**：所有 agent 角色、模型选择、tool 调用参数都通过 YAML 配置，不要 hardcode 在代码里
- **离线 / 在线分离**：所有重计算（CLIP 特征提取、光流、shot 分割、长视频 caption）放离线 preprocess 阶段，跑一次后缓存；agent 推理阶段只读 cache，避免重复推理
- **模块独立可替换**：每个 agent / tool / model wrapper 都是独立模块，能单独 unit test
- **接口先行**：所有跨模块通信先定义 dataclass / TypedDict（不用 dict 兜底）；这样 Claude Code 在补全细节时类型安全
- **第三方依赖宽容**：第三方模型（CLIP、RAFT、video gen API）都包一层 wrapper，外部接口稳定；这样换模型时只改 wrapper
- **日志结构化**：所有 agent 决策、tool 调用、validation 结果都写到结构化 trajectory log（JSONL），后续做 RL/Reward 训练直接用

### 1.2 核心抽象
- **Shot**：源视频被切分后的最小单元（典型 1-10 秒），是 retrieval 的基本对象
- **Event / Story / Character**：narrative memory 的更高层次抽象
- **Segment**：剪辑产出的时间线上的一段（可能由 1+ shots 组成，可以是 retrieval 来的也可以是 generation 来的）
- **EditingScript**：最终输出，一个 Segment 列表，每个 Segment 含 source（retrieve/generate）、起止时间、对应 shot ID 或 generation prompt
- **NarrativeMemory**：四层记忆（shot / event / story / character），全部以可查询的对象存在

---

## 2. 系统总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Inputs                                       │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐                  │
│  │ Source Video │  │ User Prompt    │  │ Music (opt.) │                  │
│  │ (1..N files) │  │ (free text)    │  │              │                  │
│  └──────────────┘  └────────────────┘  └──────────────┘                  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 1 · Understanding  (offline, cacheable per source video)          │
│  ───────────────────────────────────────────────────────────────────     │
│  • Shot Segmentation         (PySceneDetect / AutoShot)                  │
│  • Feature Extraction        (CLIP / RAFT / U²-Net)                      │
│  • Music Analysis            (All-In-One)                                │
│  • Shot-Level Captioning     (Qwen-VL / GPT-4o, 10-shot rolling buffer)  │
│  • Event Grouping            (BaSSL / heuristic)                         │
│  • Story Abstraction         (LLM call on event list)                    │
│  • Character ID + Voiceprint (InsightFace + WeSpeaker)                   │
│  • Cinematography Tags       (ShotVL)                                    │
│   ─────────────────────────────────────────────────────                  │
│   → NarrativeMemory(shots, events, stories, characters)                  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 2 · Planning  (multi-agent, online)                               │
│  ───────────────────────────────────────────────────────────────────     │
│   ScreenwriterAgent  ──→ produces GlobalStructuralPlan                   │
│   DirectorAgent      ──→ expands each section into SegmentGuidance       │
│        (semantic_query, heuristic_weights, rhythmic_pacing,              │
│         cinematography_constraints, retrieval_feasibility_hint)          │
│   OrchestratorAgent  ──→ validates plan against narrative memory         │
│        ↻ iterative loop (proposer ↔ validator)                           │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 3 · Compose  (hybrid retrieval + generation, online)              │
│  ───────────────────────────────────────────────────────────────────     │
│  For each SegmentGuidance:                                               │
│    EditorAgent:                                                          │
│      1. RetrievalTool      → top-K candidate shot sequences              │
│      2. (if low score) GenerationTool → synthesized shot                 │
│         · conditioned on neighbor end-frame + flow + char identity       │
│      3. ValidatorAgent (reward model) → score + accept/reject            │
│      4. on reject → relax constraint, loop or fallback                   │
│                                                                          │
│  Final stitching:                                                        │
│      AssemblyTool → ffmpeg compose → output.mp4                          │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                            Output Video + Trajectory Log
```

---

## 3. 仓库结构

```
LongVideoEditAgent/
├── README.md
├── pyproject.toml              # uv / poetry
├── requirements.txt
├── .env.example                # API keys, paths
│
├── configs/                    # YAML configs（全部不在代码里 hardcode）
│   ├── default.yaml            # 默认全局配置
│   ├── agents/
│   │   ├── screenwriter.yaml
│   │   ├── director.yaml
│   │   ├── orchestrator.yaml
│   │   ├── editor.yaml
│   │   └── validator.yaml
│   ├── models/
│   │   ├── llm.yaml            # backbone LLM 配置（多种后端）
│   │   ├── mllm.yaml           # 多模态模型（Qwen-VL / GPT-4o）
│   │   ├── video_gen.yaml      # 视频生成模型（Wan / OmniWeaving / API）
│   │   └── perception.yaml     # CLIP / RAFT / U²-Net / InsightFace
│   └── heuristics/             # 预设 editing heuristics 模板
│       └── presets.yaml
│
├── src/longvideoagent/
│   ├── __init__.py
│   ├── types.py                # 全局 dataclass / TypedDict
│   ├── config.py               # 配置加载
│   ├── logging.py              # 结构化 logging
│   │
│   ├── memory/                 # ⭐ Hierarchical Narrative Memory
│   │   ├── __init__.py
│   │   ├── schema.py           # Shot / Event / Story / Character 数据结构
│   │   ├── store.py            # 存储后端（SQLite + FAISS）
│   │   ├── builder.py          # 从 raw video 构建 memory
│   │   └── retriever.py        # 多粒度检索接口
│   │
│   ├── perception/             # ⭐ Stage 1 重计算的封装
│   │   ├── __init__.py
│   │   ├── shot_detector.py    # PySceneDetect wrapper
│   │   ├── feature_extractor.py# CLIP embedding
│   │   ├── flow_extractor.py   # RAFT 光流
│   │   ├── saliency.py         # U²-Net 显著性图
│   │   ├── captioner.py        # MLLM 给 shot 写 caption
│   │   ├── character_id.py     # InsightFace + SOLIDER + tracing
│   │   ├── dialogue_matcher.py # OCR字幕 + WeSpeaker
│   │   ├── cinematography.py   # 镜头属性（ShotVL）
│   │   └── music_analyzer.py   # All-In-One wrapper
│   │
│   ├── agents/                 # ⭐ Stage 2 多 agent 系统
│   │   ├── __init__.py
│   │   ├── base.py             # BaseAgent ABC
│   │   ├── screenwriter.py
│   │   ├── director.py
│   │   ├── orchestrator.py
│   │   ├── editor.py
│   │   └── validator.py
│   │
│   ├── tools/                  # ⭐ Stage 3 可被 agent 调用的工具
│   │   ├── __init__.py
│   │   ├── base.py             # BaseTool ABC（统一 function-call 接口）
│   │   ├── retrieval_tool.py   # 镜头检索 + beam search
│   │   ├── generation_tool.py  # 视频生成模型 wrapper
│   │   ├── assembly_tool.py    # ffmpeg 拼接
│   │   └── metric_tool.py      # 6+ 个量化 metric（视觉/听觉/语义）
│   │
│   ├── models/                 # ⭐ 模型 wrappers（外部接口）
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── base.py
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   ├── deepseek_client.py
│   │   │   └── vllm_local.py   # 本地 Qwen / Llama
│   │   ├── video_gen/
│   │   │   ├── base.py
│   │   │   ├── omniweaving.py  # Tencent-Hunyuan/OmniWeaving
│   │   │   ├── wan_local.py
│   │   │   └── api_client.py   # Veo / Sora 兜底
│   │   └── reward/
│   │       ├── base.py
│   │       └── mllm_judge.py   # zero-shot MLLM-as-judge（后续替换为 fine-tuned RM）
│   │
│   ├── orchestration/          # ⭐ Multi-agent 编排（LangGraph 风格）
│   │   ├── __init__.py
│   │   ├── state.py            # 全局 StateGraph 的 state schema
│   │   ├── graph.py            # 构建并运行 agent graph
│   │   └── messages.py         # agent 间消息协议
│   │
│   ├── pipeline/               # ⭐ 主流程入口
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Stage 1 主入口
│   │   ├── plan.py             # Stage 2 主入口
│   │   ├── compose.py          # Stage 3 主入口
│   │   └── run.py              # 端到端 main()
│   │
│   ├── prompts/                # ⭐ 所有 agent 的 prompt 模板（独立成文件）
│   │   ├── screenwriter.txt
│   │   ├── director_query.txt
│   │   ├── director_heuristic.txt
│   │   ├── director_pacing.txt
│   │   ├── orchestrator_validate.txt
│   │   ├── editor_summary.txt
│   │   └── reward_judge.txt
│   │
│   └── utils/
│       ├── video_io.py         # ffmpeg 包装
│       ├── audio_io.py
│       └── trajectory.py       # 轨迹 logger（为后续 RL 用）
│
├── benchmark/                  # 评估
│   ├── __init__.py
│   ├── metrics.py              # 6+ low-level metric + 高层 human eval
│   ├── mashup_bench.py         # 对接 DIRECT/Mashup-Bench
│   └── cine_bench.py           # 对接 CineBench
│
├── scripts/                    # 命令行入口
│   ├── preprocess_video.py     # 单视频离线预处理
│   ├── build_memory.py         # 构建 narrative memory
│   ├── run_pipeline.py         # 端到端跑一个 case
│   ├── eval_benchmark.py       # 跑评估
│   └── visualize_trajectory.py # 可视化 agent 决策轨迹
│
├── tests/
│   ├── unit/
│   │   ├── test_memory.py
│   │   ├── test_perception.py
│   │   ├── test_agents.py
│   │   └── test_tools.py
│   ├── integration/
│   │   └── test_end_to_end.py
│   └── fixtures/
│       └── tiny_clip.mp4       # 5秒小视频作 fixture
│
└── notebooks/                  # 调试用 Jupyter
    └── 01_explore_memory.ipynb
```

---

## 4. 核心数据结构（最先写，其他模块都依赖）

文件：`src/longvideoagent/types.py`

```python
"""
Global type definitions. 所有跨模块通信使用这里的类型，不用 dict。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import numpy as np


# ─────────────────────────────────────────────
# Memory primitives
# ─────────────────────────────────────────────

@dataclass
class CinematographyTags:
    """Shot-level cinematography metadata. 来自 ShotVL 等模型。"""
    shot_scale: Literal["extreme_close_up", "close_up", "medium", "long", "extreme_long"]
    shot_movement: Literal["static", "pan", "tilt", "zoom", "tracking", "handheld"]
    shot_angle: Literal["eye_level", "low", "high", "dutch", "overhead"]
    framing: Literal["single", "two_shot", "group", "ows", "pov"]


@dataclass
class ShotFeatures:
    """离线 pre-compute 的 frame-level/clip-level 特征。numpy ndarray on disk."""
    clip_embedding: np.ndarray          # (D,) average over keyframes
    start_flow: Optional[np.ndarray]    # (H, W, 2) end-of-prev-frame flow
    end_flow: Optional[np.ndarray]
    start_saliency: Optional[np.ndarray]
    end_saliency: Optional[np.ndarray]
    avg_flow_magnitude: float           # 动态强度，给 m6 用


@dataclass
class Shot:
    """The atomic unit. 1-10 秒。"""
    shot_id: str                        # e.g. "movie_a__s00042"
    source_video: str                   # path to source mp4
    start_time: float                   # seconds
    end_time: float
    caption: str                        # MLLM-generated, context-aware
    cinematography: CinematographyTags
    features: ShotFeatures              # heavy, lazy-load
    character_ids: list[str] = field(default_factory=list)
    dialogue: Optional[str] = None


@dataclass
class Event:
    """A few semantically related shots."""
    event_id: str
    shot_ids: list[str]
    summary: str
    start_time: float                   # in source
    end_time: float


@dataclass
class Story:
    """Top-level narrative arc (1+ events)."""
    story_id: str
    title: str
    event_ids: list[str]
    summary: str
    arc_role: Literal["setup", "rising", "climax", "falling", "resolution"]


@dataclass
class Character:
    """Identity entity across all sources."""
    character_id: str
    name: Optional[str]                 # auto-name from dialogue / metadata
    face_embedding: np.ndarray
    voice_embedding: Optional[np.ndarray]
    appearance_shot_ids: list[str]
    profile_summary: str                # LLM-generated biography


@dataclass
class NarrativeMemory:
    """The whole knowledge base for one or multiple source videos."""
    shots: dict[str, Shot]
    events: dict[str, Event]
    stories: dict[str, Story]
    characters: dict[str, Character]
    music_profile: Optional["MusicProfile"] = None


# ─────────────────────────────────────────────
# Music
# ─────────────────────────────────────────────

@dataclass
class MusicSection:
    name: Literal["intro", "verse", "chorus", "bridge", "outro", "instrumental"]
    start_time: float
    end_time: float
    energy_db: float
    num_beats: int


@dataclass
class MusicProfile:
    audio_path: Path
    duration: float
    bpm: float
    beats: list[float]                  # all beat onset times
    downbeats: list[float]
    sections: list[MusicSection]


# ─────────────────────────────────────────────
# Plan & Script (agent output)
# ─────────────────────────────────────────────

@dataclass
class GlobalStructuralPlan:
    """Output of ScreenwriterAgent. Section-wise mapping."""
    section_plans: list["SectionPlan"]


@dataclass
class SectionPlan:
    music_section_idx: int              # index into MusicProfile.sections
    energy_level: Literal["low", "medium", "high", "extreme"]
    visual_tags: list[str]              # e.g. ["urban_pursuit", "fast_paced"]
    rationale: str                      # for trace / debug


@dataclass
class SegmentGuidance:
    """Output of DirectorAgent. One per editing segment."""
    segment_idx: int
    parent_section_idx: int
    semantic_query: str                 # CLIP-friendly retrieval query
    editing_heuristic: str              # name in heuristics/presets.yaml
    rhythmic_pacing: list[int]          # shot durations in beats e.g. [2,2,2,2]
    cinematography_hints: list[str]
    retrieval_feasibility: float        # 0-1, 给 Editor 做 routing 用


@dataclass
class EditingSegment:
    """A produced segment ready for assembly."""
    segment_idx: int
    source: Literal["retrieval", "generation"]
    duration: float
    # Retrieval branch
    shot_ids: list[str] = field(default_factory=list)
    shot_trims: list[tuple[float, float]] = field(default_factory=list)  # (start, end) within each shot
    # Generation branch
    gen_prompt: Optional[str] = None
    gen_video_path: Optional[Path] = None
    gen_conditions: Optional[dict] = None  # neighbor anchor frames / flow
    # Quality
    metric_scores: dict[str, float] = field(default_factory=dict)
    accepted_by_validator: bool = False


@dataclass
class EditingScript:
    """The whole compiled timeline, ready for ffmpeg assembly."""
    segments: list[EditingSegment]
    total_duration: float
    output_path: Optional[Path] = None


# ─────────────────────────────────────────────
# Agent trajectory（给 future RL 用）
# ─────────────────────────────────────────────

@dataclass
class AgentStep:
    """A single agent decision, logged to JSONL."""
    timestamp: float
    agent_name: str
    state_snapshot: dict
    action: str                         # tool name or "respond"
    action_input: dict
    observation: dict
    reward: Optional[float] = None      # filled by validator / RM
```

---

## 5. 各 stage 接口规范

### 5.1 Stage 1 · Preprocess (Understanding)

**文件**：`src/longvideoagent/pipeline/preprocess.py`

**功能**：把一个或多个源视频 + 一个音乐文件，转成可序列化的 `NarrativeMemory`。

**接口**：
```python
def preprocess(
    source_videos: list[Path],
    music: Optional[Path],
    cache_dir: Path,
    config: PreprocessConfig,
) -> NarrativeMemory:
    ...
```

**实现要点**：
1. **断点续传**：每一步（shot detection → feature → caption → ...）都写自己的 cache 文件（按 source video hash 命名）。重跑时跳过已存在的。
2. **并行**：feature 提取按 shot 切分并行（用 `concurrent.futures.ProcessPoolExecutor`）。
3. **MLLM caption 的 rolling buffer**：调 captioner 时把前 10 个 shot 的 caption 也作为 context（CineAgents 的策略）。
4. **MusicProfile** 用 `mir-aidj/all-in-one` 包（`pip install allin1`）。
5. **Character ID**：先用 InsightFace 提 face embedding，然后聚类 + 用 trajectory（相邻 shot 同一 track）传播 ID。这一步精度对后续 narrative 很关键，但 v0.1 可以先用占位（每个 shot 标 "unknown character N"），v0.2 再 swap 真实实现。

### 5.2 Stage 2 · Plan

**文件**：`src/longvideoagent/pipeline/plan.py`

**功能**：用 multi-agent 把 `(NarrativeMemory, user_prompt)` 转成 `list[SegmentGuidance]`。

**接口**：
```python
def plan(
    memory: NarrativeMemory,
    user_prompt: str,
    config: PlanConfig,
    trajectory_logger: TrajectoryLogger,
) -> list[SegmentGuidance]:
    ...
```

**实现要点**：
1. **用 LangGraph 编排**（推荐，因为：state 持久化、checkpoint、易扩展加 RL）。或者自己写一个 mini state machine。
2. **state schema**（见 `orchestration/state.py`）：
```python
class PlanState(TypedDict):
    user_prompt: str
    memory_ref: str                # narrative memory 的标识（不直接放进 state，太大）
    global_plan: Optional[GlobalStructuralPlan]
    segment_guidances: list[SegmentGuidance]
    validation_messages: list[str]
    iteration: int
    max_iterations: int            # default 5
```
3. **节点**：`screenwriter_node` → `director_node` → `orchestrator_node` → 条件边（验证通过 END，否则回 director）
4. **三个 agent 各自的实现**：每个 agent 是 `BaseAgent` 子类，实现 `run(state) -> partial_state`；内部调 LLM 用 prompt 模板（从 `prompts/` 读）。
5. **每一步落 trajectory log**。

### 5.3 Stage 3 · Compose

**文件**：`src/longvideoagent/pipeline/compose.py`

**功能**：对每个 `SegmentGuidance`，由 `EditorAgent` 在 thinking 中决定调 retrieval / generation tool，最终产出 `EditingScript`。

**接口**：
```python
def compose(
    memory: NarrativeMemory,
    guidances: list[SegmentGuidance],
    config: ComposeConfig,
    trajectory_logger: TrajectoryLogger,
) -> EditingScript:
    ...
```

**EditorAgent 的 ReAct 循环**（这是核心）：
```
For each guidance G:
    state = {"guidance": G, "candidates": [], "neighbor_context": last_segment_end_frame}
    
    while not done and step < MAX_STEPS:
        thought = LLM.think(state)
        
        # decide which tool to call
        if thought.action == "retrieve":
            candidates = RetrievalTool(G.semantic_query, G.heuristic_weights, ...)
            state["candidates"] += candidates
        
        elif thought.action == "generate":
            # 关键：condition on neighbor for style continuity
            gen_video = GenerationTool(
                prompt = G.semantic_query + cinematography_hint,
                first_frame_cond = state["neighbor_context"]["end_frame"],
                flow_cond = state["neighbor_context"]["end_flow"],
                character_ref = state["character_anchors"],
            )
            state["candidates"].append({"source": "generation", "video": gen_video})
        
        elif thought.action == "validate":
            scores = ValidatorAgent(state["candidates"])
            best = pick_best_above_threshold(scores)
            if best:
                done = True
                segment = build_segment(best)
            else:
                # relax constraints, try again
                G = relax(G, feedback=scores.feedback)
        
        elif thought.action == "fallback":
            # accept best-of-bad, log warning
            done = True
    
    script.append(segment)
    state["neighbor_context"] = segment_end_features(segment)
```

**关键设计点**：
- **neighbor_context 跨 segment 传递**：保证视觉连续性。这是 hybrid 路线相对纯 retrieval / 纯 generation 的核心优势。
- **character_anchors**：跨 segment 保持角色一致；生成时作为 reference image 喂给 video gen model（OmniWeaving 接受 multi-image input，正合适）。
- **MAX_STEPS=10**：防止 agent 进入死循环。
- **Routing 决策**：v0.1 可以用简单规则（`retrieval_feasibility < 0.3` 才 generate），v0.2 替换成 learned policy。

---

## 6. Agent 详细规格

每个 agent 都是 `BaseAgent` 子类：

```python
# src/longvideoagent/agents/base.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, llm_client, prompt_template_path: Path, config: dict):
        self.llm = llm_client
        self.prompt_template = load_prompt(prompt_template_path)
        self.config = config
    
    @abstractmethod
    def run(self, state) -> dict:
        """Take a state snapshot, return partial state update."""
        ...
    
    def log_step(self, logger, state, action, action_input, observation):
        logger.append(AgentStep(
            timestamp=time.time(),
            agent_name=self.__class__.__name__,
            state_snapshot=summarize(state),
            action=action,
            action_input=action_input,
            observation=observation,
        ))
```

### 6.1 ScreenwriterAgent
- **输入**：`(NarrativeMemory.summary(), user_prompt, MusicProfile)`
- **输出**：`GlobalStructuralPlan`
- **prompt** 在 `prompts/screenwriter.txt`，对应 DIRECT paper Sec 4.1 的 "Music-Driven Structure Anchoring"
- **特殊事项**：先调用 `memory.summarize()`（基于 cluster + captioning）得到一个 footage summary 字符串，避免 context overflow

### 6.2 DirectorAgent
三阶段 CoT，每个阶段一次 LLM call（DIRECT 的做法）：
1. `generate_semantic_query(section_plan, history)` → `q_sem`
2. `select_heuristic(q_sem, energy)` → 从 `heuristics/presets.yaml` 选一个
3. `compute_rhythmic_pacing(q_sem, music_section)` → list of beats
4. （新增）`estimate_retrieval_feasibility(q_sem, memory)` → 0-1，简单方法：用 q_sem CLIP-embed 后查 memory，看 top-K similarity 的最大值

### 6.3 OrchestratorAgent
- **输入**：`(GlobalStructuralPlan, [SegmentGuidance], NarrativeMemory)`
- **输出**：`(passed: bool, feedback: list[str])`
- 检查每个 guidance 的 semantic_query 在 memory 里 grounding 是否充分（这是 CineAgents 的 iterative narrative planning 核心）
- 不通过时返回具体 feedback（"Segment 3 的 query 'underwater fight' 在 memory 里只有 1 个相关 shot，建议改成 'character struggling'"）

### 6.4 EditorAgent
- 唯一会做 multi-step ReAct 的 agent
- prompt 里要明确：「你有 retrieve / generate / validate / fallback 四个 action」
- 推荐用 function-calling 接口（OpenAI tool calling / Anthropic tools）而不是纯 text parsing，更稳

### 6.5 ValidatorAgent (Reward Model)
- v0.1：zero-shot MLLM judge，输入 candidate video + guidance，输出 1-10 分 + 拒绝理由
- v0.2：替换成 fine-tuned reward model（接 Stage 3 后期实验）
- 接口稳定：`score(candidate_segment, guidance, context) -> (float, str)`

---

## 7. Tool 详细规格

### 7.1 RetrievalTool
**文件**：`src/longvideoagent/tools/retrieval_tool.py`

实现 DIRECT 的 beam search + dynamic sliding-window trimming（论文 Sec 4.3）：

```python
class RetrievalTool(BaseTool):
    name = "retrieve_shots"
    
    def __init__(self, memory: NarrativeMemory, config):
        self.memory = memory
        self.beam_width = config.beam_width  # default 3
        self.stride = config.sliding_stride
    
    def run(self, 
            semantic_query: str,
            heuristic_weights: dict[str, float],
            rhythmic_pacing: list[int],
            prev_shot_features: Optional[ShotFeatures] = None
    ) -> list[CandidateSequence]:
        # 1. Filter shot pool by semantic query (CLIP similarity)
        pool = self.memory.retrieve_by_query(semantic_query, top_k=200)
        
        # 2. Beam search
        beams = [PartialPath()]
        for k, beats in enumerate(rhythmic_pacing):
            new_beams = []
            for beam in beams:
                for shot in pool:
                    # Dynamic sliding-window trimming to maximize continuity
                    best_window = self._slide_trim(shot, beats, beam.tail_features)
                    score = self._score(beam, best_window, heuristic_weights)
                    new_beams.append(beam.extend(best_window, score))
            beams = topk(new_beams, self.beam_width)
        
        return beams  # top-B complete sequences
    
    def _slide_trim(self, shot, beats, tail):
        # enumerate windows within shot that satisfy duration,
        # pick the one maximizing motion_continuity + framing_consistency
        ...
    
    def _score(self, beam, candidate, weights):
        # 6 metrics: m1 prompt rel, m2 seg consistency,
        #            m3 motion cont, m4 framing,
        #            m5 beat sync, m6 energy
        ...
```

**Metric 实现**：复用 DIRECT supplementary 给出的公式（完整定义见我的对话历史，下面只列出文件位置）。
- `m1` prompt relevance: cosine(CLIP(shot), CLIP(prompt))
- `m2` segment consistency: cosine(CLIP(prev), CLIP(curr))
- `m3` motion continuity: magnitude + direction weighted similarity of optical flows
- `m4` framing: 1 - Wasserstein(saliency_prev_end, saliency_curr_start)
- `m5` beat-cut sync: exp(-||cut_time - nearest_beat||²/2σ²)
- `m6` energy correspondence: Spearman corr between flow magnitude and RMS

### 7.2 GenerationTool
**文件**：`src/longvideoagent/tools/generation_tool.py`

```python
class GenerationTool(BaseTool):
    name = "generate_shot"
    
    def __init__(self, model_client, config):
        self.client = model_client  # OmniWeaving / Wan / Veo API
        self.config = config
    
    def run(self, 
            prompt: str,
            duration: float,
            first_frame_cond: Optional[np.ndarray] = None,
            flow_cond: Optional[np.ndarray] = None,
            character_refs: list[np.ndarray] = None,
            cinematography_hint: Optional[str] = None,
    ) -> Path:
        """Returns path to generated mp4."""
        # Compose prompt with cinematography
        full_prompt = self._compose_prompt(prompt, cinematography_hint)
        # Call model
        return self.client.generate(
            prompt=full_prompt,
            duration=duration,
            first_frame=first_frame_cond,
            reference_images=character_refs,
            # OmniWeaving supports multi-image + text + video conditions
        )
```

推荐首选 **OmniWeaving (Tencent-Hunyuan)** 因为它原生支持 free-form composition（text + multi-image + video 条件混合），最适合「邻居 frame 当 anchor + character ref」这种用法。备用 Wan2.6 / API-only Veo3。

### 7.3 AssemblyTool
**文件**：`src/longvideoagent/tools/assembly_tool.py`

用 ffmpeg-python 把 `EditingScript` 拼成最终 mp4，包括：
- 按 shot_trims 切片
- 拼接
- 叠音乐
- 可选：交叉淡入淡出 transition

### 7.4 MetricTool
**文件**：`src/longvideoagent/tools/metric_tool.py`

独立成 tool 是为了 EditorAgent 可以主动查询 candidate 的具体 metric 值（agent 推理时能看到具体分数而不是黑盒打分）。

---

## 8. 模型 wrappers（外部接口）

### 8.1 LLM wrapper
统一接口（参考 LangChain BaseChatModel 但更轻量）：

```python
# src/longvideoagent/models/llm/base.py
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: Optional[list] = None, **kwargs) -> Response:
        ...
    
    @abstractmethod
    def supports_function_calling(self) -> bool: ...

# 实现 OpenAIClient / AnthropicClient / DeepSeekClient / VLLMLocalClient
```

### 8.2 Video Generation wrapper
关键：必须支持 conditioning（first frame / flow / ref image），否则 hybrid 没意义。

```python
class BaseVideoGenClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: float,
        first_frame: Optional[np.ndarray] = None,
        last_frame: Optional[np.ndarray] = None,
        reference_images: Optional[list[np.ndarray]] = None,
        flow_field: Optional[np.ndarray] = None,
    ) -> Path: ...
    
    @abstractmethod
    def supported_conditions(self) -> set[str]: ...
```

具体实现（按优先级）：
1. `omniweaving.py`：本地部署或 API 调用 https://github.com/Tencent-Hunyuan/OmniWeaving
2. `wan_local.py`：本地部署 Wan2.6
3. `api_client.py`：Veo3 / Sora2（兜底）

---

## 9. 配置示例（confgs/default.yaml）

```yaml
project_name: "longvideoagent"
cache_root: "./.cache"
output_root: "./outputs"

preprocess:
  shot_detector: "pyscenedetect"
  shot_detector_threshold: 27.0
  feature_extractor:
    name: "clip-vit-base-patch32"
    stride: 4              # sample every 4 frames
  flow_extractor:
    name: "raft-large"
    spatial_pool: 8
  saliency:
    name: "u2net"
  captioner:
    name: "qwen3-vl-8b-instruct"
    buffer_size: 10        # rolling context
  cinematography:
    name: "shotvl-3b"
  music_analyzer:
    name: "allin1"
  character_id:
    detector: "insightface"
    fallback: "solider"
    cluster_threshold: 0.6

plan:
  max_iterations: 5
  models:
    screenwriter: "deepseek-v3"
    director: "deepseek-v3"
    orchestrator: "claude-sonnet-4-5"   # better at validation

compose:
  editor_model: "claude-sonnet-4-5"
  validator_model: "qwen3-vl-8b-instruct"
  max_editor_steps: 10
  retrieval:
    beam_width: 3
    sliding_stride: 4
  generation:
    enabled: true
    backend: "omniweaving"
    fallback_threshold: 0.4   # if retrieval score < this, try generation
  metric_weights:                # default; can be overridden per heuristic
    m1_prompt: 0.20
    m2_seg_consistency: 0.15
    m3_motion_continuity: 0.20
    m4_framing: 0.15
    m5_beat_sync: 0.15
    m6_energy: 0.15

heuristic_presets:
  semantic_priority:        {m1: 0.5, m2: 0.3, m3: 0.05, m4: 0.05, m5: 0.05, m6: 0.05}
  motion_continuity:        {m1: 0.1, m2: 0.1, m3: 0.4, m4: 0.1, m5: 0.15, m6: 0.15}
  framing_consistency:      {m1: 0.1, m2: 0.1, m3: 0.15, m4: 0.4, m5: 0.15, m6: 0.10}
  hybrid_visual_coherent:   {m1: 0.1, m2: 0.1, m3: 0.25, m4: 0.25, m5: 0.15, m6: 0.15}
  default:                  {m1: 0.20, m2: 0.15, m3: 0.20, m4: 0.15, m5: 0.15, m6: 0.15}
```

---

## 10. 命令行入口

```bash
# 1. 预处理（单视频或多视频）
python scripts/preprocess_video.py \
    --source /data/movie_a.mp4 /data/movie_b.mp4 \
    --music /data/track.mp3 \
    --cache-dir ./.cache/run1 \
    --config configs/default.yaml

# 2. 构建 narrative memory（也可在 preprocess 里自动完成）
python scripts/build_memory.py --cache-dir ./.cache/run1

# 3. 跑端到端 pipeline
python scripts/run_pipeline.py \
    --cache-dir ./.cache/run1 \
    --user-prompt "Make a 3-minute action montage with high energy chase scenes" \
    --output ./outputs/run1.mp4 \
    --trajectory-log ./outputs/run1_trajectory.jsonl

# 4. 评估
python scripts/eval_benchmark.py \
    --benchmark mashup-bench \
    --output-dir ./outputs/eval_run1
```

---

## 11. v0.1 验收标准（搭脚手架阶段）

Claude Code 完成 v0.1 后，应能 demonstrate：

- [ ] **`pytest tests/unit/` 全绿**（即使每个模块只有占位实现）
- [ ] **`pytest tests/integration/test_end_to_end.py` 能跑通**（用 `tests/fixtures/tiny_clip.mp4` 作输入，所有 MLLM/video-gen call 用 mock 返回固定值）
- [ ] **跑 `scripts/run_pipeline.py` on `tiny_clip.mp4`** 能产出一个 `.mp4` 文件（即使内容是 dummy），并写出结构化 trajectory log
- [ ] **所有 agent 的 prompt 都在 `prompts/` 下作为单独文件**，不在代码里 hardcode
- [ ] **配置加载正常**：改 `configs/default.yaml` 里某个值，运行时能反映出来
- [ ] **README.md** 包含：项目介绍、安装步骤、quickstart 命令、模块图

---

## 12. v0.2 路线图（先不要做，但 Claude Code 写代码时要为这些预留接口）

1. **接入真实模型**：把所有 mock 换成 OmniWeaving / Qwen-VL / All-In-One / RAFT 真实调用
2. **跑 Mashup-Bench / CineBench** 看初始指标，建立 baseline
3. **Trajectory log → 训练数据**：把 trajectory log 整理成 (state, action, reward) tuples，准备做 RL
4. **接入 Agentic RL 训练**（可选 path A）：用 RAGEN/VerlTool 的 StarPO 训 EditorAgent；reward = ValidatorAgent 打分 + benchmark metric 综合
5. **微调 Reward Model**（可选 path B）：收集 (segment, guidance, human_label) 三元组，从通用 MLLM 微调出专门的 EditingQualityRM、ProcessRM、NarrativeCoherenceRM
6. **personalize**：给 trajectory log 加 user_id，按用户做 LoRA fine-tune（更后期）

---

## 13. 给 Claude Code 的具体执行指令（写代码时优先级）

按下面顺序写，每一步都先写好接口和单元测试再实现：

1. **types.py** — 第一个写，所有人都要 import
2. **config.py** — 配置加载，pydantic 校验
3. **logging.py + trajectory.py** — 早写，后面所有模块都要用
4. **memory/schema.py + store.py** — 用 SQLite 存 metadata，用 FAISS 存 CLIP embeddings；shot features 用 npz 文件单独存
5. **perception/* (mock first)** — 每个 perception 模块先写 mock 实现（返回 dummy data 但保持接口），后面再 swap 真实模型
6. **pipeline/preprocess.py** — 串起 perception 模块，能跑通 fixture
7. **agents/base.py + tools/base.py** — ABC 接口
8. **prompts/*.txt** — 占位 prompt，能让 LLM mock 跑通
9. **orchestration/state.py + graph.py** — LangGraph 编排
10. **agents/screenwriter.py → director.py → orchestrator.py** — 逐个实现，每个都 mock LLM 跑通
11. **tools/retrieval_tool.py** — beam search 完整实现（不需要真实模型，用 mock features 就行）
12. **tools/generation_tool.py** — 接口完整 + mock 实现
13. **tools/assembly_tool.py** — ffmpeg 拼接（这步要真实，否则没法产出 mp4）
14. **agents/editor.py + validator.py** — ReAct 循环
15. **pipeline/run.py** — 端到端串起来
16. **scripts/* + tests/*** — 命令行 + 测试

---

## 14. 关键参考代码（Claude Code 可学习的开源工程）

| 项目 | 用途 | URL |
|---|---|---|
| **DIRECT** | 整体 pipeline 风格参照（hierarchical agent + beam search） | https://github.com/AK-DREAM/DIRECT |
| **FilmAgent** | Multi-agent 通信 & Critique-Correct-Verify 模式 | https://github.com/HITsz-TMG/FilmAgent |
| **MovieAgent** | 视频生成端 agent 编排（character bank 用法） | https://github.com/showlab/MovieAgent |
| **GLANCE** | bi-loop architecture / observe-think-act-verify | https://github.com/ZihaoLinQZ/GLANCE-Video-Editing-Agent (代码 announced) |
| **OmniWeaving** | 视频生成模型 wrapper 参考 | https://github.com/Tencent-Hunyuan/OmniWeaving |
| **PySceneDetect** | shot detection | https://github.com/Breakthrough/PySceneDetect |
| **All-In-One** | 音乐结构分析 | https://github.com/mir-aidj/all-in-one |
| **LangGraph** | Multi-agent 编排框架 | https://langchain-ai.github.io/langgraph/ |
| **RAGEN** | (v0.2 用) agentic RL framework | https://github.com/mll-lab-nu/RAGEN |
| **verl** | (v0.2 用) RL training | https://verl.readthedocs.io |

---

## 15. 注意事项 / 反模式（Claude Code 请避免）

- ❌ **不要把多个职责塞进一个 agent**（比如让 Director 同时做语义查询和镜头选择）
- ❌ **不要在 hot loop 里调 LLM**（Editor 的每个 segment 最多 10 步，超出说明 prompt 有问题）
- ❌ **不要 import 整个 narrative memory 进 state**（太大，用引用 + lazy load）
- ❌ **不要 hardcode prompt 到 .py 文件**（必须 `prompts/*.txt`）
- ❌ **不要把 video gen 跨多 GPU 调度逻辑写在 agent 里**（agent 只调 tool，调度逻辑在 wrapper 内）
- ❌ **不要让 v0.1 依赖任何需要 GPU 的真实模型**（所有 perception/MLLM/video-gen 都先 mock；这样在 CPU 也能跑通脚手架）

---

## END

如有任何模糊处，**优先做出选择并写注释 `# DESIGN_DECISION: ...`，不要停下来问**。
所有 design decision 都记到 `docs/decisions.md`，方便后续回顾。
