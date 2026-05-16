# LongVideoEditAgent — 系统说明书

> 这是给"想理解这个 codebase 在做什么、为什么这样做、与领域内其他工作什么关系"的人的主文档。
>
> 阅读顺序建议：先看 §0 TL;DR → §1 你为什么应该关心 → §2 数据流（细致到字段） → §3 self-loop 评估机制 → §4 agentic evolution 机制 → §5 相关工作调研。

---

## 0. TL;DR — 一段话讲清楚

LongVideoEditAgent 是一个 **指令驱动的长视频剪辑 agent 系统**。系统由三个 stage 串联：

1. **Understanding (offline)** — 把原始长视频切 shot、抽 CLIP/RAFT/U²-Net 特征、跑 MLLM 写 caption、聚事件 / 故事 / 角色，**沉淀成一份四层 narrative memory**。
2. **Planning (online, multi-agent)** — 用 Screenwriter → Director → Orchestrator 三 agent 协作，把 `(memory, user_prompt)` 翻译成一个**带语义查询、剪辑节奏、镜头偏好**的 `list[SegmentGuidance]`。
3. **Compose (online, hybrid retrieval+generation)** — Editor agent 跑 ReAct 循环，对每段 guidance 在 `(retrieval, generation)` 之间路由，validator 给候选打分，最终用 ffmpeg 拼成 mp4。

**之所以值得做**，是因为它不只是一个剪辑工具——而是一个 **agentic 学习场**：每次 run 都会写出一份结构化 trajectory（JSONL），未来可以用它（a）训练 EditorAgent 的 policy（RL/StarPO），（b）训练专门的 EditingQualityRM（reward model），（c）做 self-loop evaluation 闭环。本文档主要解释这两个目标——self-loop evaluation 与 agentic evolution——为什么这个架构是为它们生的、它们如何被领域内的相关工作（DIRECT、FilmAgent、CineAgents、Reflexion、Self-Refine、STaR、ReST、Voyager、RLAIF/Constitutional AI、RAGEN 等）启发，以及我们具体采纳了哪些机制、为什么。

---

## 1. 你为什么应该关心这套架构

剪长视频本身是一个**信息高度密集、目标长程、决策维度多元**的任务：

* **信息密集**：一段两小时的电影 ≈ 数千 shot × 每个 shot 多帧 × 视觉/听觉/字幕/角色等多模态。
* **目标长程**：用户的 prompt（"做一个 3 分钟的高能动作蒙太奇"）要被翻译成几十次微观决策（"这一刀切在第 47 秒还是 49 秒"）。
* **决策维度多元**：每个 cut 同时关心语义匹配、运动连续、镜头一致、节奏对齐、能量曲线 ...

任何能在这个任务上做得 **可学习、可评估、可自我改进** 的系统，几乎自动满足做一个通用 **agentic 学习场** 的所有条件：

| 通用 agent 系统需要 | 长视频剪辑天然提供 |
|---|---|
| 长程多步决策 | 每个 segment 都是若干 retrieve/generate 决策 |
| 子目标分解 | 三层 agent（structure → segment → shot）就是分解 |
| 工具调用 | RetrievalTool / GenerationTool / AssemblyTool / MetricTool |
| 可计算的反馈信号 | DIRECT m1..m6 + MLLM-as-judge + 最终用户评分 |
| 可观察的 trajectory | JSONL 日志（state, action, observation, reward） |
| 跨 run 的可比较性 | Mashup-Bench / CineBench 数据集 |

所以把这个系统造好的副产物是一个**完整的 agentic evolution 实验台**：trajectory 沉淀 → reward model 训练 → policy 更新 → 新 trajectory ...

---

## 2. 数据流（字段级）

### 2.1 全景图

```
                  ┌─────────────────────────────────────────────────┐
INPUTS            │ source_videos: list[Path]                       │
                  │ user_prompt: str                                │
                  │ music: Optional[Path]                           │
                  └────────────────────┬────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ Stage 1 · Understanding  (offline, cacheable)                             │
   │ pipeline/preprocess.py::preprocess(...)                                    │
   │                                                                            │
   │  PerceptionPipeline:                                                       │
   │    ShotDetector(PySceneDetect)         → list[(start_s, end_s)]            │
   │    FeatureExtractor(CLIP)              → ShotFeatures.clip_embedding        │
   │    FlowExtractor(RAFT)                 → ShotFeatures.{start,end}_flow      │
   │    SaliencyExtractor(U²-Net)           → ShotFeatures.{start,end}_saliency  │
   │    ShotCaptioner(Qwen-VL/GPT-4o)       → Shot.caption                       │
   │    CinematographyTagger(ShotVL)        → Shot.cinematography                 │
   │    CharacterIdentifier(InsightFace)    → Shot.character_ids + Character bank │
   │    DialogueMatcher(EasyOCR+WeSpeaker)  → Shot.dialogue                       │
   │    MusicAnalyzer(All-In-One)           → MusicProfile                        │
   │                                                                            │
   │  MemoryBuilder:                                                            │
   │    Shots ──greedy 30s grouping──> Events                                   │
   │    Events ──LLM summarise──> Stories                                       │
   │                                                                            │
   │  Persistence (memory/store.py::MemoryStore):                              │
   │    SQLite:        shots / events / stories / characters / music tables    │
   │    FAISS:         CLIP embedding ANN (numpy fallback if no FAISS)         │
   │    .npz per shot: heavy flow/saliency arrays                              │
   └──────────────────────────────────────┬───────────────────────────────────┘
                                          │ NarrativeMemory
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ Stage 2 · Planning  (online, multi-agent)                                  │
   │ pipeline/plan.py::plan(...)                                                │
   │                                                                            │
   │   ScreenwriterAgent(prompts/screenwriter.txt)                              │
   │      input :  memory.summarize(), user_prompt, music_profile.sections     │
   │      output:  GlobalStructuralPlan(section_plans=[SectionPlan, ...])      │
   │                                                                            │
   │              ↓                                                             │
   │                                                                            │
   │   DirectorAgent  ── runs 3+1 LLM calls per SectionPlan                    │
   │      step 1 (director_query.txt)     : semantic_query (CLIP-friendly)     │
   │      step 2 (director_heuristic.txt) : pick preset from heuristics.yaml   │
   │      step 3 (director_pacing.txt)    : rhythmic_pacing list[int]          │
   │      step 4 (in-code)                : retrieval_feasibility (0..1)       │
   │      → SegmentGuidance × one per SectionPlan                              │
   │                                                                            │
   │              ↓                                                             │
   │                                                                            │
   │   OrchestratorAgent(prompts/orchestrator_validate.txt)                     │
   │      heuristic checks:                                                     │
   │        • no_duplicate_queries                                              │
   │        • grounding_in_memory (feasibility > ε)                            │
   │        • plan-vs-guidance coverage                                         │
   │        • music-vs-plan coverage                                            │
   │      + LLM-level coherence judgement                                       │
   │      → {"passed": bool, "feedback": list[str]}                            │
   │                                                                            │
   │   Loop: if not passed and iteration < max_iterations,                     │
   │         feed feedback BACK into Director → re-emit guidances.             │
   │   (orchestration/graph.py — Python state machine; LangGraph optional)     │
   └──────────────────────────────────────┬───────────────────────────────────┘
                                          │ list[SegmentGuidance]
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ Stage 3 · Compose  (online, retrieve+generate)                             │
   │ pipeline/compose.py::compose(...)                                          │
   │                                                                            │
   │   EditorAgent ReAct loop per SegmentGuidance G:                            │
   │      neighbor_context = { end_frame, end_flow, character_anchors,         │
   │                           beats, expected_start_time_s }                  │
   │      first_action = "generate" if G.retrieval_feasibility < θ             │
   │                     else "retrieve"                                        │
   │      while step < max_editor_steps and not chosen:                        │
   │         action = first_action if step==1 else LLM.pick(...)               │
   │         if action == "retrieve":                                           │
   │            cand = RetrievalTool.run(G, memory, neighbor_context)          │
   │            # beam search × top-K shot pool × sliding window                │
   │            # × 6 metrics × heuristic-specific weights                      │
   │         elif action == "generate":                                         │
   │            cand = GenerationTool.run(G, neighbor_context, output_dir)     │
   │            # backed by BaseVideoGenClient (mock / OmniWeaving / Wan / Veo)│
   │         elif action == "fallback":                                         │
   │            chosen = best-of(candidates)                                    │
   │         auto-validate(cand)  → metric_scores["validator"]                 │
   │         if accepted: chosen = cand                                        │
   │      → EditingSegment(source, duration, shot_ids/gen_video_path,          │
   │                       metric_scores, accepted_by_validator, ...)          │
   │                                                                            │
   │   AssemblyTool (always real ffmpeg, never mocked):                        │
   │      for seg in script.segments:                                          │
   │         cut shot windows OR transcode gen mp4 → per-seg intermediate      │
   │      concat-demuxer ffmpeg → silent timeline                              │
   │      [optional] mux music                                                 │
   │      → outputs/run.mp4                                                    │
   └──────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
OUTPUTS           ┌──────────────────────────────────────────────────────────┐
                  │ outputs/run.mp4                                          │
                  │ outputs/run_trajectory.jsonl                             │
                  │   (per-step: agent_name, action, observation, reward)    │
                  │   (per-segment: segment_finalized summary w/ metric vec) │
                  └──────────────────────────────────────────────────────────┘
```

### 2.2 关键数据结构与字段（来自 `src/longvideoagent/types.py`）

下表只列**跨 stage 流动的字段**，全 dataclass 在 `types.py` 里：

| Dataclass | 关键字段 | 写入方 | 读取方 |
|---|---|---|---|
| `Shot` | `shot_id`, `source_video`, `start_time`, `end_time`, `caption`, `cinematography`, `features`, `character_ids`, `dialogue` | `pipeline.preprocess` | `RetrievalTool`、`AssemblyTool`、`MetricTool` |
| `ShotFeatures` | `clip_embedding`, `start_flow`, `end_flow`, `start_saliency`, `end_saliency`, `avg_flow_magnitude` | `perception/*` | `RetrievalTool._score_step`、`MetricTool` |
| `Event` | `event_id`, `shot_ids`, `summary`, `start_time`, `end_time` | `memory.builder._group_events` | `NarrativeMemory.summarize()` |
| `Story` | `story_id`, `event_ids`, `summary`, `arc_role` | `memory.builder._build_stories` | `Screenwriter` 经 summary |
| `Character` | `character_id`, `face_embedding`, `voice_embedding`, `appearance_shot_ids`, `profile_summary` | `perception.character_id` | `GenerationTool` reference_images、`EditorAgent` 维持身份 |
| `MusicProfile` | `bpm`, `beats`, `downbeats`, `sections` | `perception.music_analyzer` | `Screenwriter` 段落锚定、`Director` 节奏、`MetricTool.m5_beat_sync` |
| `SectionPlan` | `music_section_idx`, `energy_level`, `visual_tags`, `rationale` | `ScreenwriterAgent` | `DirectorAgent` |
| `SegmentGuidance` | `segment_idx`, `parent_section_idx`, `semantic_query`, `editing_heuristic`, `rhythmic_pacing`, `cinematography_hints`, `retrieval_feasibility` | `DirectorAgent` | `OrchestratorAgent` 验、`EditorAgent` 消费 |
| `EditingSegment` | `source∈{retrieval,generation}`, `duration`, `shot_ids`+`shot_trims`+`source_videos` 或 `gen_prompt`+`gen_video_path`, `metric_scores`, `accepted_by_validator`, `validator_reasons` | `RetrievalTool` / `GenerationTool` | `AssemblyTool`、trajectory log |
| `EditingScript` | `segments`, `total_duration`, `music_path`, `output_path` | `EditorAgent.run` | `AssemblyTool` |
| `AgentStep` | `timestamp`, `agent_name`, `state_snapshot`, `action`, `action_input`, `observation`, `reward` | 所有 agent.log_step() | 未来 RL 训练（RAGEN / verl） |

### 2.3 一个真实样本走完整个流水线

跑：
```bash
python scripts/run_pipeline.py \
    --source tests/fixtures/tiny_clip.mp4 \
    --cache-dir .cache/demo \
    --user-prompt "Make a 4-second highlight reel with high energy" \
    --output outputs/demo.mp4 \
    --trajectory-log outputs/demo_trajectory.jsonl
```

得到的 trajectory.jsonl（15 行）按事件展开：

```
[screenwriter.emit_global_plan]     → 4 SectionPlan（覆盖 intro/verse/chorus/outro 4 个 music sections）
[director.emit_segment_guidances]    → 4 SegmentGuidance，4 个不同的 semantic_query
[orchestrator.validate]              → passed=True，feedback=[]  （iteration=1 就过了）
[editor.retrieve   ×4]               → RetrievalTool 在每段都产出 EditingSegment
[validator.judge   ×4]               → MockReward 给每段 7.4~7.5 分（>6 阈值 → accept）
[editor.segment_finalized ×4]        → 每段写一份摘要：metric_scores、accepted、duration
```

最后 `outputs/demo.mp4` 是真实 ffmpeg 拼出的 16 秒 mp4。

> **数据流的关键不变量**：除了 `AssemblyTool` 一处写真实 mp4 之外，所有阶段**只产生不修改**之前阶段的对象。这是为了让 trajectory log 永远是单向因果链，可以被 RL 直接消费。

---

## 3. 核心机制 #1 — Self-Loop Evaluation

### 3.1 是什么

**Self-loop evaluation** 指系统在一次 run 内、**不需要人工标注**就能：

1. 产生一个候选（retrieval beam 的 best beam，或一个 gen 出的 clip）；
2. 用一个"自评判"模块（reward model / MLLM-as-judge / metric tool）给候选打分；
3. 根据分数决定 accept / reject / retry / relax constraints；
4. 留下结构化日志，下一次 run / 下一次训练用得上。

### 3.2 我们的实现

| 层 | self-eval 触发点 | 评分器 | 反馈走向 |
|---|---|---|---|
| **Stage 2 Plan 层** | OrchestratorAgent | (a) heuristic checks（duplicate query, grounding, coverage） + (b) LLM coherence call | feedback 被 DirectorAgent 在下一轮 re-plan 时读取，作为 query 重写的上下文 |
| **Stage 3 Segment 层** | EditorAgent 自动 validate | `BaseRewardModel.score(candidate, guidance)` — v0.1 `MockRewardModel`（用 6 metric 加权），v0.2 `MLLMJudge`，v0.3 fine-tuned `EditingQualityRM` | accept ⇒ chosen=cand；reject ⇒ EditorAgent 切下一个 action（retrieve/generate/fallback） |
| **Stage 3 Shot 层** | RetrievalTool beam 评分 | 6 个 metric（DIRECT m1..m6） + heuristic-specific 权重 + anti-repeat penalty | 高分 beam 留在前 K，低分丢弃 |

注意三层 self-eval **判据不同**：

* Plan 层判 "narrative coherence + grounding"——是否符合用户 prompt、是否能在 memory 里找到对应素材。
* Segment 层判 "candidate quality"——这段 retrieval/generation 到底好不好。
* Shot 层判 "step-wise compatibility"——这一刀的局部连续性如何。

这种分层是**故意的**：粗粒度 judge 用低开销启发式 + LLM 调用一次；细粒度 judge 在 beam search 里被调用上万次，所以必须是闭式公式（m1..m6）而不是 LLM。这避免了"LLM 在 hot loop 里"反模式（设计文档 §15 第二条）。

### 3.3 与领域内 Self-Loop / Self-Improve 工作的关系

| 相关工作 | 核心思想 | 我们采纳了什么 | 我们做了什么不同 |
|---|---|---|---|
| **Self-Refine** (Madaan et al., NeurIPS 2023) — 同一个 LLM 先生成 → 自评 → 自改，迭代 N 轮 | LLM 自我反馈循环 | OrchestratorAgent 的 LLM coherence call → DirectorAgent re-plan 这一对就是 Self-Refine 范式 | 我们**不**让 ScreenwriterAgent 自改自己的输出（避免一个 agent 既当裁判又当选手）；改由独立 OrchestratorAgent 评分 |
| **Reflexion** (Shinn et al., NeurIPS 2023) — 失败后用文本反馈写"verbal episodic memory"，下次决策时检索 | 文本化失败反思 + 跨 episode 记忆 | OrchestratorAgent 的 `feedback: list[str]` 是 Reflexion-style 文本反馈；写进 `state.validation.feedback`，下一轮 DirectorAgent 在 `_call_query` 中拼接进 prompt | 我们暂时**没做**跨 run 的 reflection——单 run 内即用即丢；v0.2 可以把 trajectory 沉淀成跨 run 的 "lesson book" |
| **Constitutional AI / RLAIF** (Bai et al. 2022, Lee et al. 2023) — 用 AI 自己生成的偏好数据替代 RLHF 的人工标注 | "AI 监督 AI" 是可行的 | `MLLMJudge` 就是 Constitutional 思路：用更强的 MLLM 给候选打分，作为后续 fine-tune RM 的训练信号 | 我们**叠加** DIRECT m1..m6 这种基于物理的 metric（不靠 LLM），降低对单一 judge 的偏置依赖 |
| **G-Eval** (Liu et al. 2023) — 用 CoT prompt 引导 LLM 给文本/对话打分，且校准过 | LLM-as-judge 的标准做法 | `prompts/reward_judge.txt` 就是 G-Eval 风格的 1-10 评分 prompt，要求输出 reasons + accepted | 我们要求 judge 同时输出 reject reasons，以便 EditorAgent 做 constraint relaxation |
| **Multi-Agent Debate** (Du et al. 2023) — 让多个 agent 对结果争论几轮再投票 | 多视角 self-eval 比单 agent 鲁棒 | OrchestratorAgent + ValidatorAgent 是两套独立判据（plan-level 一致性 vs segment-level 质量），形式上像两 agent 投票 | 我们**没**做 N-agent debate；剪辑任务上"投票"成本高于收益 |
| **CritiqueLLM / Skywork-Reward** — 训练专用的 critique 模型 | 通用 MLLM 不够好，要 fine-tune 一个 RM | Roadmap：v0.2 `EditingQualityRM` 就是这条路 | 我们的 trajectory log 格式（state, action, observation, reward）天然兼容 RM 训练 dataset |
| **DIRECT / Mashup-Bench** (https://github.com/AK-DREAM/DIRECT) — 提出 m1..m6 metric 与 beam search retrieval | 6-metric 加权评分公式 | 直接采纳：`tools/metric_tool.py` 是 m1..m6 的实现，权重由 `configs/heuristics/presets.yaml` 配置 | DIRECT 是纯 retrieval；我们扩展为 retrieval+generation 混合，并把 m1..m6 作为"step-wise self-eval signal" |

### 3.4 为什么这个组合是合理的

我们的 self-loop evaluation 不是单一信号，而是**三层异构判据互相校验**：

1. **物理/几何信号** (m1..m6) — 来自 DIRECT，闭式公式、确定性、能在 beam search 内调用 1e4 次；
2. **MLLM 判据** (ValidatorAgent) — 来自 G-Eval / Constitutional AI，捕捉物理信号抓不到的"主观质量"；
3. **结构判据** (OrchestratorAgent) — 来自 CineAgents / Reflexion，捕捉跨段一致性 / narrative arc。

任何一层失效，其他两层兜底；任何一层过强（比如 reward hacking），其他两层会以矛盾形式暴露出来。这是 **assurance via redundancy**——RLHF 时代的 reward hacking 教训直接套用到剪辑上。

---

## 4. 核心机制 #2 — Agentic Evolution

### 4.1 是什么

**Agentic evolution** 指 agent 跨 run 持续变好：通过收集自身行为数据、用 reward signal 评价、反向更新 policy（prompt / weights / search heuristic / sub-skills），下一次 run 比上一次更聪明。

### 4.2 我们的实现路径（v0.1 已就位 + v0.2 / v0.3 计划）

```
                ┌─────────────────────────┐
   v0.1 ✓     │ Trajectory JSONL log     │  ← BaseAgent.log_step + segment_finalized
                │ (state, action, obs, r) │
                └────────────┬────────────┘
                             │
                ┌────────────┴────────────────────────────────────┐
                │                                                  │
                ▼                                                  ▼
   ┌────────────────────────┐                  ┌────────────────────────────┐
v0.2 │ Reward model fine-tune │              v0.2 │ Offline RL / SFT on traj  │
   │  EditingQualityRM      │                  │   StarPO (RAGEN)           │
   │  ProcessRM             │                  │   GRPO / PPO (verl)        │
   │  NarrativeCoherenceRM  │                  │   reject-sample SFT (ReST) │
   └──────────┬─────────────┘                  └──────────┬─────────────────┘
              │                                            │
              └────────────────────┬───────────────────────┘
                                   ▼
                  ┌─────────────────────────────────┐
v0.3            │ New EditorAgent policy           │
                  │ (LoRA on Qwen-VL backbone, or   │
                  │  prompt/heuristic optimisation) │
                  └──────────────┬──────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────┐
                  │ Next run → better trajectory     │
                  └─────────────────────────────────┘
                                 │
                                 └─ (loop back to top)
```

### 4.3 为什么这个 codebase 已经为 agentic evolution 准备好了

要让 agentic evolution 跑起来，**必须**满足下面五条；我们逐条对照：

| 必要条件 | 我们怎么满足 | 文件 |
|---|---|---|
| **结构化 trajectory**（state, action, observation, reward） | `AgentStep` dataclass + `TrajectoryLogger`，JSONL 格式与 HF datasets / RAGEN trajectory loader 兼容 | `types.py::AgentStep`, `utils/trajectory.py` |
| **可计算的 reward**（不依赖人工标注） | 三层 self-eval（§3）；reward 字段写进每一步 trajectory | `models/reward/`、`tools/metric_tool.py` |
| **可替换的 policy**（不绑死单一 LLM） | `BaseLLMClient` ABC + factory；改 `configs/models/llm.yaml` 一键换 backbone | `models/llm/base.py::build_llm_from_alias` |
| **可学习的 sub-skill**（agent 不止生成文本，还选 tool） | `BaseTool` ABC + 函数调用 schema；EditorAgent 已经在 tool 间路由 | `tools/base.py`, `agents/editor.py` |
| **跨 run 的可比较 benchmark** | Mashup-Bench / CineBench 适配器接口已建（v0.2 接数据） | `benchmark/` |

### 4.4 与领域内 Agentic Evolution 工作的关系

| 相关工作 | 核心思想 | 我们采纳了什么 | 我们做了什么不同 |
|---|---|---|---|
| **STaR** (Zelikman et al., NeurIPS 2022) — 用模型自己生成的"对的"reasoning chain 做 SFT 训练，迭代 | self-generated 成功 trajectory → SFT 训练数据 | trajectory log 里每个 `segment_finalized` 都带 `accepted` 标志 + `reward`；天然能筛出"成功"样本做 STaR 训练 | STaR 用于纯文本推理；我们用于 multi-step tool use（跨段视频剪辑） |
| **ReST / ReST-EM** (Gulcehre et al. 2023; Singh et al. 2023) — Expectation-Maximisation：交替"生成大量轨迹 → 用 reward 筛选 → SFT 更新 policy" | EM 风格的 self-training pipeline | v0.2 的 EditorAgent 训练就是 ReST：让当前 EditorAgent 跑 N 次（不同 user_prompt × 不同视频源） → 用 ValidatorAgent 打分 → top-K 做 SFT | 我们的 reward 比 ReST 更稠密（每段都有 m1..m6 + judge），不是单一 final reward |
| **Voyager** (Wang et al. 2023, Minecraft) — 自动 curriculum + skill library + iterative prompting | "skill 是可被 agent 调用的 named tool" + iterative refinement | 我们的 `tools/` 目录是 Voyager 风格 skill library；新增一个工具 = 新增一个 `BaseTool` 子类，agent 立刻可调用 | Voyager 用环境奖励（task success）；我们用三层 self-eval（§3） |
| **Reflexion** (Shinn et al., NeurIPS 2023) — verbal episodic memory across episodes | "失败的 verbal lesson" 提升 next episode | OrchestratorAgent.feedback 已是单 run 内的 Reflexion；v0.2 把 feedback 沉淀成跨 run lesson book 是直接扩展 | 我们的 lesson 不是给单 agent 看，而是给 multi-agent 系统看（Director 看 Orchestrator 的 lesson） |
| **RAGEN / StarPO** (https://github.com/mll-lab-nu/RAGEN) — agentic RL training framework，特化于"LLM-as-agent" 场景 | trajectory-level RL 训练（不是 token-level） | v0.2 直接用 RAGEN 的 StarPO 训 EditorAgent；trajectory 格式已对齐 | RAGEN 关注"LLM 怎么调 tool"；我们的 reward 一部分来自 tool 内部（m1..m6 不是 LLM 输出） |
| **verl** (https://verl.readthedocs.io) — general-purpose RL training library，支持 PPO / GRPO | RL backbone | v0.3 当 EditingQualityRM 训好后，用 verl 做 GRPO；ValidatorAgent 的 reward 直接喂进去 | verl 是 framework，我们是 task / dataset / reward 提供方 |
| **AutoGen** (Microsoft) — 通用 multi-agent 对话框架 | "agent 间用消息通信" 的抽象 | `orchestration/messages.py::Message` 与 AutoGen 接口相似 | 我们没用 AutoGen runtime（绑死 conversational pattern）；而是 LangGraph-style state graph，更适合非对话型 stage-based pipeline |
| **LangGraph** (langchain-ai) | StateGraph + conditional edges + checkpoint | `orchestration/graph.py` 完全对齐 LangGraph 接口；v0.2 一行切换；v0.1 自带 Python state-machine fallback 避免重依赖 | LangGraph 的强项是 checkpoint / resume，我们在 v0.2 用上后能做"中途断电恢复" |
| **FilmAgent** (HITsz-TMG) — 多 agent 拍电影，Critique-Correct-Verify 三段循环 | CCV pattern；多个角色分工（producer, director, ...） | OrchestratorAgent 是 Critique；DirectorAgent re-plan 是 Correct；下一轮 OrchestratorAgent 是 Verify | FilmAgent 主要做"电影从无到有生成"；我们做"长视频源 → 剪辑"，是 retrieval-heavy 而非 generation-heavy |
| **MovieAgent** (showlab) — 视频生成 agent，重在角色一致性 ("character bank") | 跨 shot 维持人物身份 | `Character` dataclass + `GenerationTool` 的 `reference_images` 参数；调用 OmniWeaving 时传 character face_embedding | MovieAgent 默认每段都生成；我们优先 retrieval，generation 是兜底 |
| **CineAgents** — iterative narrative planning + per-shot caption rolling buffer | 滚动 buffer 的 captioning + 多 agent narrative validation | `ShotCaptioner.buffer`（10-shot deque） + OrchestratorAgent 的 narrative-coherence check 都直接来自 CineAgents | CineAgents 主要解决"长视频如何持续注入 narrative 上下文"；我们 reuse 这部分到 Stage 1 captioner 与 Stage 2 orchestrator |
| **GLANCE** (ZihaoLinQZ) — bi-loop architecture: observe-think-act-verify | bi-loop = 内层 ReAct + 外层 verify | EditorAgent 是内层 ReAct；ValidatorAgent + OrchestratorAgent 共同构成外层 verify | GLANCE 把 verify 也做成 LLM 调用；我们的外层 verify 一部分是闭式 metric（m1..m6） |

### 4.5 一句话总结：我们在做什么"独特"的事

**领域内大多数视频剪辑 agent 系统是 "一次性 inference 系统"**——做完一个 case 就结束，下一个 case 重新开始。

**我们想做的是一个 "agentic 学习场"**：每个 case 的 trajectory 都被结构化沉淀，下次 run 时（a）EditorAgent 的 policy 已被 trajectory 更新，（b）Reward Model 已被 trajectory 更新，（c）Director 的 heuristic 选择已被 trajectory 更新——三个 loop 嵌套着自我演化。

这套架构是**让"long video editing"既是任务又是测试床**——剪出来的 mp4 是产品，跑出来的 trajectory 才是真正的资产。

---

## 5. 详细 Related Work 调研（已经在 §3、§4 表格里出现，本节做更深的展开）

### 5.1 视频剪辑 / 视频生成 agent 类工作

#### 5.1.1 DIRECT (Mashup-Bench)
- **链接**: https://github.com/AK-DREAM/DIRECT
- **解决什么**: 给定一段音乐 + 一堆 footage，retrieve-only 地剪出 music video。
- **核心贡献**: 把剪辑评估拆成 6 个 metric (m1 prompt rel / m2 seg consistency / m3 motion / m4 framing / m5 beat / m6 energy)；用 beam search 做长序列 retrieval；引入 dynamic sliding-window trimming。
- **我们采纳**: 100% 采纳 m1..m6 公式（`tools/metric_tool.py`）；beam search 实现（`tools/retrieval_tool.py`）；heuristic presets 模式（`configs/heuristics/presets.yaml`）。
- **我们改造**: 加入 generation 路由 + character bank + 多源视频 anti-repeat penalty + Stage 2 多 agent planning 层（DIRECT 是单一 LLM call）。
- **为什么**: DIRECT 在 retrieval-only 场景已被验证；我们补 generation 是为了"源视频缺素材"时不止于 fail。

#### 5.1.2 FilmAgent
- **链接**: https://github.com/HITsz-TMG/FilmAgent
- **解决什么**: Producer/Director/Actor 多 agent 协作拍电影（视频生成驱动）。
- **核心贡献**: Critique-Correct-Verify (CCV) 三段 loop；角色化分工 prompt。
- **我们采纳**: CCV 模式套在 Plan 层（Orchestrator 是 Critique，Director re-plan 是 Correct，下一轮 Orchestrator 是 Verify）；角色化分工 prompt（一个 agent 一个 `prompts/*.txt`）。
- **我们改造**: FilmAgent 的 agent 在同一对话里轮流发言，我们改成 state-graph based（节点固定，data flow 明确）。
- **为什么**: state graph 比 conversational 更适合视频剪辑（数据量太大塞不进对话历史）。

#### 5.1.3 MovieAgent
- **链接**: https://github.com/showlab/MovieAgent
- **解决什么**: Story → 多段视频生成，重点解决跨段角色一致性。
- **核心贡献**: Character bank — 每个角色一个 face embedding + 风格 reference set；视频生成时作为 multi-image condition。
- **我们采纳**: `Character` dataclass + `GenerationTool` 的 `character_refs` 参数；OmniWeaving 调用时把 face_embedding 作为 reference image。
- **我们改造**: 我们的 character bank 来自源视频（perception 阶段提取），而 MovieAgent 来自用户配置。
- **为什么**: 我们的输入是**源视频**，所以 face embedding 直接从源里抽就行；不用让用户额外提供。

#### 5.1.4 GLANCE
- **链接**: https://github.com/ZihaoLinQZ/GLANCE-Video-Editing-Agent (代码 announced)
- **解决什么**: 长视频编辑的 agent。
- **核心贡献**: bi-loop = observe-think-act-verify 外层 + ReAct 内层。
- **我们采纳**: EditorAgent 是内层 ReAct（observe candidates → think → act = retrieve/generate）；外层 verify 由 ValidatorAgent + OrchestratorAgent 联合担任。
- **我们改造**: 外层 verify 一部分用闭式 metric（不是 LLM），降低 cost。
- **为什么**: 视频 segment 多，外层 verify 不能每段都是 LLM 调用。

#### 5.1.5 CineAgents
- **解决什么**: 长视频理解/编辑的 multi-agent narrative planning。
- **核心贡献**: 滚动 caption buffer（保 10 shot 的上下文）、iterative narrative validation。
- **我们采纳**: `ShotCaptioner.buffer = deque(maxlen=10)` 直接对齐；OrchestratorAgent 的 narrative-coherence loop。
- **我们改造**: 把 "iterative validation" 落地为 Python state-machine 的 conditional edge，确保 max_iterations 强制收敛。
- **为什么**: 避免 narrative coherence loop 无限循环。

### 5.2 Agent 框架 / 编排类工作

#### 5.2.1 ReAct (Yao et al., ICLR 2023)
- **解决什么**: 让 LLM 在 reason 和 act 之间交替推理（Thought → Action → Observation → Thought ...）。
- **核心贡献**: ReAct prompt pattern。
- **我们采纳**: EditorAgent 是标准 ReAct 实现；`prompts/editor_summary.txt` 是 ReAct prompt。
- **我们改造**: 把 "validate" 这一 action 自动化（避免冗余 LLM 调用），ReAct 简化为 {retrieve, generate, fallback}。
- **为什么**: validate 是闭式打分，不需要 LLM 自己决定。

#### 5.2.2 LangGraph
- **链接**: https://langchain-ai.github.io/langgraph/
- **解决什么**: Multi-agent 系统的 state graph 编排（StateGraph、conditional edges、checkpoint）。
- **核心贡献**: 显式 state schema + conditional routing + 持久化 checkpoint。
- **我们采纳**: `orchestration/state.py::PlanState` 是 LangGraph StateGraph 标准；`orchestration/graph.py` 在 LangGraph 可用时直接用，否则降级为 Python state-machine（同接口）。
- **我们改造**: v0.1 默认走 Python fallback 是为了避免重依赖；v0.2 一行切换到 LangGraph 拿 checkpoint。
- **为什么**: 让 v0.1 pip install 轻；让 v0.2 拿到 resume + persist 能力。

#### 5.2.3 AutoGen (Microsoft)
- **链接**: https://github.com/microsoft/autogen
- **解决什么**: 多 agent 对话框架，主要是 chat-style。
- **核心贡献**: GroupChat / ConversableAgent / 自定义角色。
- **我们没采纳**: 视频剪辑里 agent 间的"消息"是结构化 dataclass（SegmentGuidance、EditingScript），不是自然语言对话；AutoGen 的 conversational primitives 不合适。
- **我们借鉴**: `orchestration/messages.py::Message` 字段（sender/receiver/role/content/meta）参考 AutoGen 消息封装。

### 5.3 Self-Improvement / Agentic RL 类工作

#### 5.3.1 STaR (Zelikman et al., NeurIPS 2022)
- **解决什么**: "Self-Taught Reasoner"——让模型生成 reasoning chain，正确的留下做 SFT，错的用 rationalisation 改对再做 SFT。
- **我们采纳**: trajectory log 的 reward 字段是 STaR-style 筛选信号；v0.2 可以挑 reward > θ 的 segment 做 SFT。
- **我们改造**: STaR 是单 turn（一个 prompt 一个 answer）；我们是 multi-step（一个 prompt N 段编辑），筛选粒度更细（segment-level 而非 episode-level）。

#### 5.3.2 ReST / ReST-EM (Gulcehre et al. 2023, Singh et al. 2023)
- **解决什么**: Expectation-Maximisation 自训练：用当前 model 生成大量样本 → 用 reward filter 高质量 → SFT 更新 → 循环。
- **我们采纳**: v0.2 EditorAgent 训练完全对齐 ReST EM；ValidatorAgent 提供 reward。
- **我们改造**: ReST 一般用单一 reward；我们用多层 reward（m1..m6 + judge），训练时可以做 multi-objective SFT 或 reward 加权采样。

#### 5.3.3 Reflexion (Shinn et al., NeurIPS 2023)
- **解决什么**: Agent 失败后写一段 verbal lesson，存入 episodic memory；下次 attempt 时检索这些 lesson。
- **我们采纳**: OrchestratorAgent → DirectorAgent feedback 链就是单 run 内的 Reflexion；feedback 文本直接拼进 director 的 prompt。
- **v0.2 计划**: 把 feedback 跨 run 沉淀成 "lesson book"，下次类似 user_prompt 时检索。

#### 5.3.4 Self-Refine (Madaan et al., NeurIPS 2023)
- **解决什么**: 同一个 LLM 先生成 → 自评 → 自改，迭代 N 轮。
- **我们采纳**: OrchestratorAgent 评分 + DirectorAgent re-plan 在结构上是 Self-Refine；但我们把"评"和"改"拆给了不同 agent（避免同一 agent 既当裁判又当选手）。
- **为什么不让同一 agent 自评自改**: 经验上 self-grading 容易自我合理化；分离评 / 改职责更鲁棒。

#### 5.3.5 Voyager (Wang et al. 2023, Minecraft)
- **解决什么**: 开放世界 agent，自动 curriculum + skill library + iterative prompting。
- **我们采纳**: `tools/` 目录是 Voyager 风格的 skill library——每个 tool 是 named API，agent 通过调用名字使用；新增 tool = 新 skill。
- **v0.2 计划**: 加入 automatic curriculum——根据 trajectory log 找出系统"做得差"的 user_prompt 类型（比如 anime 风格、超长 video），自动生成更多这种 case 做 training。

#### 5.3.6 RAGEN (mll-lab-nu)
- **链接**: https://github.com/mll-lab-nu/RAGEN
- **解决什么**: agentic RL training framework，提出 StarPO 算法。
- **我们采纳**: trajectory log 格式与 RAGEN 完全兼容（JSONL，每行一个 step）；v0.2 直接用 RAGEN 训 EditorAgent。
- **我们改造**: RAGEN 默认 reward 来自 final episode return；我们的 reward 是 dense 的（每段 segment 都有 m1..m6 + judge）——RAGEN 的 trainer 支持 dense reward 直接用。

#### 5.3.7 verl
- **链接**: https://verl.readthedocs.io
- **解决什么**: 通用 LLM RL 训练库，支持 PPO / GRPO / RLOO。
- **我们采纳**: v0.3 当 EditingQualityRM 训好后，用 verl + GRPO 做 EditorAgent 的 RL fine-tune。
- **为什么先 RAGEN 再 verl**: RAGEN 上手快、对 agentic 特化好；verl 更通用、更适合后期大规模训练。

### 5.4 Reward Modeling 类工作

#### 5.4.1 Constitutional AI / RLAIF (Bai et al. 2022, Lee et al. 2023)
- **解决什么**: 用 AI 给的偏好数据替代人工标注。
- **我们采纳**: MLLMJudge 是 RLAIF 思路；v0.2 用 MLLMJudge 生成 (segment, score) 对，训 EditingQualityRM。

#### 5.4.2 Process Reward Models (Lightman et al. 2023 "Let's Verify Step by Step")
- **解决什么**: 对推理过程的每一步打分（PRM）比只对最终答案打分（ORM）训练更稳定、效果更好。
- **我们采纳**: trajectory log 的 segment-level reward 就是天然的 PRM 训练数据——每段都有 reward，而不是只在最后给一个总分。

#### 5.4.3 Skywork-Reward / RewardBench
- **解决什么**: 训练强 reward model 与建立 reward model benchmark。
- **v0.2 计划**: 当 EditingQualityRM 训出后，在 RewardBench-style hold-out set 上评估其对齐质量。

#### 5.4.4 G-Eval (Liu et al. 2023)
- **解决什么**: CoT prompt 引导 LLM 给文本质量打分，且校准。
- **我们采纳**: `prompts/reward_judge.txt` 完全是 G-Eval 风格——要求 1-10 评分 + reasons + accepted bool。

### 5.5 Perception 类（不是 evolution 核心，但数据流的基础）

| 真实库 | 用在哪 | 为什么 |
|---|---|---|
| **PySceneDetect** (`scenedetect`) | Shot 分割 | 业界事实标准，速度快，threshold tunable |
| **CLIP / open_clip** | Shot embedding | 跨模态文 / 图 retrieval 的事实标准 |
| **RAFT (torchvision)** | Optical flow | 速度精度平衡好，pre-trained 直接能用 |
| **U²-Net** | Saliency | 轻量 SOTA，给 m4 (framing) 提供输入 |
| **Qwen2-VL / GPT-4o** | Shot caption | Caption + 跨段 narrative，本质都靠 MLLM |
| **ShotVL** | Cinematography 标签 | 专门为镜头属性 fine-tune 过的 MLLM |
| **InsightFace** | 角色识别 | 工业级人脸 embedding |
| **All-In-One** | 音乐结构分析 | one-shot 同时给 bpm + beats + downbeats + sections，比单独跑多个 librosa 模块快 |

### 5.6 Video Generation 类

| 真实库 | 用在哪 | 为什么 |
|---|---|---|
| **OmniWeaving** (Tencent-Hunyuan) | 首选 video gen backend | 支持 text + multi-image + first/last frame + flow_field 条件，最适合"邻接 frame 作 anchor + character ref"用法 |
| **Wan2.6** | 备选 video gen | 开源、能本地 inference、quality 高 |
| **Veo (via google-genai)** | 兜底 API | 当本地推理不可用时的 fallback |

---

## 6. 我们 v0.1 已经为 Agentic Evolution / Self-Loop Evaluation 准备好的 "能力清单"

每一项都已经在代码里就位（不是 "todo"）：

1. **结构化 trajectory** — 每一次 agent 决策都写一行 JSONL，含 state / action / observation / reward。`outputs/demo_trajectory.jsonl` 就是样本。
2. **三层异构 reward** — m1..m6 闭式公式（最快） + MLLM judge（最稳健） + Orchestrator coherence check（覆盖跨段）。
3. **可替换 backbone** — 改 `configs/models/llm.yaml::aliases` 一处就换 LLM；改 `configs/models/video_gen.yaml::aliases` 一处就换视频生成模型。
4. **可解释 routing** — EditorAgent 的 retrieve / generate 决策由 `retrieval_feasibility` 触发，可被未来 learned policy 直接替代。
5. **可观察 plan→guidance→segment→shot 链** — trajectory 的 `segment_finalized` 摘要包含 semantic_query → metric_scores → accept 决策，可被 RM 直接消费。
6. **可重放 LangGraph-aligned state graph** — orchestration 用 PlanState TypedDict，等价于 RL 的 (state, action, next_state)。
7. **可外接 benchmark** — `benchmark/` 目录已建好 Mashup-Bench / CineBench 适配器接口，v0.2 接数据即可对比。
8. **可单元测试的小颗粒** — 67 个 pytest case 覆盖每个模块，agent / tool / metric 全可单独打分。

---

## 7. 路线图（v0.2 / v0.3 / v0.4）

| 阶段 | 时间盒 | 关键交付 | 验证 |
|---|---|---|---|
| **v0.1** ✓ 已完成 | 1 周 | 脚手架 + mock pipeline + 端到端可跑 + 67 测试绿 | `pytest tests/ -q` 全绿；`outputs/demo.mp4` 可播 |
| **v0.2** | 2-3 周 | (a) 关闭 mocks，接真 perception（CLIP、RAFT、Qwen-VL、All-In-One、InsightFace），(b) 接真 LLM（DeepSeek + Claude），(c) 接 OmniWeaving 真 video gen，(d) 跑 Mashup-Bench / CineBench 建 baseline | DIRECT 论文 metric 复现，error bar 内对齐 |
| **v0.3** | 4-6 周 | (a) 用 trajectory log 生成 (segment, guidance, score) 三元组数据集，(b) Fine-tune EditingQualityRM (从 Qwen-VL backbone 起步 LoRA)，(c) 把 ValidatorAgent 切到 fine-tuned RM | RM 在 hold-out trajectory 上 Spearman corr > 0.6 |
| **v0.4** | 8-12 周 | (a) 用 RAGEN / verl 跑 StarPO，目标 EditorAgent 的 retrieve/generate 决策策略，(b) reward = EditingQualityRM + benchmark metric 综合 | RL 训完的 EditorAgent 在 Mashup-Bench 上比 v0.2 baseline 有显著提升 |
| **v0.5** | 12-16 周 | personalisation：给 trajectory 加 user_id；按用户做 LoRA fine-tune（adaptive editing style） | Per-user A/B：用户偏好率 > 50% |

---

## 8. 给 reviewer / 给团队的 talking points

如果有人问"这个项目和 DIRECT 有什么本质不同"——

* DIRECT 是 retrieval-only + 单一 LLM call planner；我们是 hybrid retrieval+generation + 多 agent state graph。
* DIRECT 的评估是 hold-out metric；我们的评估是闭环 self-loop（trajectory → RM → next trajectory）。
* DIRECT 把 reward 当成 "evaluation tool"；我们把它当成 "training signal"。

如果有人问"为什么不直接用 GPT-4 / Claude 一次生成所有编辑指令"——

* 长视频 prompt 太长，单 LLM call 上下文塞不下整个 narrative memory（数千 shot × caption）。
* 多 agent 分工让每个 agent 的上下文只装它需要的子集（Screenwriter 看 summary，Director 看一个 SectionPlan，Editor 看一段 SegmentGuidance），上下文压缩比 ~100×。
* 多 agent 留下 trajectory 是为了未来 RL；单 LLM call 不留下中间状态，没法做 evolution。

如果有人问"为什么 self-eval 不全用 MLLM judge"——

* MLLM judge 慢（一次调用 1-5 秒）；beam search 内每个候选都打分会让 retrieve 阶段从秒级变小时级。
* MLLM judge 容易 reward hack（agent 学会针对 judge 的偏置生成内容）；用 m1..m6 闭式公式 + judge 多层 cross-check 是 reward hacking 的天然防御。

---

## 9. 结尾：为什么这个 codebase 值得继续投入

代码层面（v0.1）：脚手架完整、67 测试绿、端到端能跑、mock-first 让 CPU 也能开发。

研究层面（v0.2+）：

* 长视频剪辑是天然的 dense reward、long horizon、tool use、multi-modal、可评估 task——把一个 agent 系统在这上面练好，它在别的 agentic 任务上也很可能有迁移性。
* trajectory log 就是 dataset——每跑一次 case 就为后续 RL / RM 训练攒一份数据。
* 评估有 ground truth（Mashup-Bench / CineBench）和无 ground truth（用户偏好）两条路并行，可以同时验证 RM 质量与用户体验。

把这个项目做好的副产物，是一套**适用于任何 long-horizon agentic 任务的 evolution 平台**——剪辑只是它的第一个 instance。

---

## 10. Single-Agent vs Multi-Agent — 我们的选择与放大策略

### 10.1 答案是 multi-agent，并且能给出可证的理由

LongVideoEditAgent 在 v0.1 起就是 **multi-agent**，并且 v0.2（这一轮新增）扩展到 **6 个 agent**：

```
   ScreenwriterAgent → DirectorAgent → OrchestratorAgent → EditorAgent → ValidatorAgent
                                                                              │
                                                                              ▼
                                                            (post-run) CriticAgent
```

为什么不直接用一次 LLM 调用（single-agent）一口气出剪辑指令？三条硬约束：

**约束 #1 · 上下文预算**
一段 2 小时电影预处理后 ≈ 数千 shot × 每个 shot 含 caption + cinematography + character_ids + features 引用 + dialogue —— 仅 caption 文字就足以撑爆 200K context window。Multi-agent 让每个 agent 只看自己需要的子集：

| Agent | 它需要的上下文 | 大小量级 |
|---|---|---|
| Screenwriter | NarrativeMemory.summarize()（5-10 行） + MusicProfile.sections | 1-3 KB |
| Director | 单一 SectionPlan + 滚动 history | < 1 KB |
| Orchestrator | 全部 SegmentGuidance（结构化） + summary | 3-5 KB |
| Editor | 单一 SegmentGuidance + 当前 neighbor_context | < 1 KB |

每个 LLM call 的有效上下文压缩 ~100×；这是 multi-agent 不可替代的工程价值。

**约束 #2 · 异构 cognitive demands**
不同子任务的最优 LLM 不一样：
* Screenwriter 需要长程叙事创意 → 偏好大模型（Claude Sonnet 4.5）
* Director 需要稳定结构化输出 → 中模型即可（DeepSeek-V3）
* Validator 需要 multimodal 视觉理解 → MLLM（Qwen-VL）
* Critic 需要规则化扫描 + 偶尔 LLM 抽象 → 大部分非 LLM

单 agent 系统在这种异构需求下要么过 spec（全用最强 LLM，烧钱），要么欠 spec（全用一个折中模型，每件事都凑合）。

**约束 #3 · 异构 reward signals**
agent evolution 的核心是 credit assignment——把最终好/坏的结果归因到具体决策。Multi-agent 自带 credit assignment 边界：

* Screenwriter 的好/坏：是否每个 music section 都被覆盖、energy_level 是否对应实际能量曲线
* Director 的好/坏：semantic_query 在 memory 里 grounding 是否充分、pacing 是否合理
* Editor 的好/坏：retrieve/generate 路由是否正确、validator 接受率
* Validator 的好/坏：跟 m1..m6 mean 的 disagreement，跟人工偏好的 Spearman 相关性

Single agent 系统只能看到一个总 reward（"剪得好不好"），训练时被迫做巨大的 credit assignment 推断；multi-agent 则可以**对每个 agent 独立 fine-tune**——把一个难问题拆成 6 个独立 RL/SFT 问题。

### 10.2 我们在 v0.2 这一轮放大的 multi-agent 优势（已实现）

放大不是简单的"多加 agent"，而是在 multi-agent 的**结构属性**（多视角、独立判据、可组合）上做杠杆。三个新机制：

| 机制 | 利用的 multi-agent 属性 | 文件 |
|---|---|---|
| **EnsembleRewardModel** —— 同时跑 K 个 reward model，按 mean ± std 聚合 + quorum 投票 | 多视角 reduces single-judge bias；disagreement 本身是 active-learning 信号 | `models/reward/ensemble.py` |
| **CriticAgent**（第 6 个 agent） —— 跑完 pipeline 后扫 trajectory，提取 5 类失败模式，写 Lesson | 元 agent ≠ 任何执行 agent，避免 self-grading；独立判据 | `agents/critic.py` |
| **ScreenwriterAgent self-consistency** —— 跑 K 次，对 SectionPlan 按 energy_level 取众数 | 把 single-call 变 ensemble；对最关键决策（global plan）做共识 | `agents/screenwriter.py::_aggregate_plans` |

### 10.3 与领域内 multi-agent 工作的关系

| 工作 | 核心论点 | 我们采纳的部分 | 我们刻意没采的部分 |
|---|---|---|---|
| **Multi-Agent Debate** (Du et al. 2023) | N agents 多轮辩论比单 agent 鲁棒 | EnsembleRewardModel 是单 turn 版的 debate（多 judge 一次性表决） | 多轮辩论（cost > 收益；剪辑任务一段 segment 不值多轮 LLM 调用） |
| **DyLAN — Dynamic LLM Agent Network** (Liu et al. 2024) | agent 拓扑应该按 task 动态调整 | v0.2 的 `mocks.reward`/`config.compose.generation.backend` 等是轻量"动态拓扑"——按场景换 agent backbone | 完全动态加 / 减 agent；剪辑场景的 agent 角色相对稳定 |
| **MetaGPT** (Hong et al. 2024) | 给每个 agent 一个 SOP（standard operating procedure） | 每个 agent 一份 `prompts/*.txt` 就是它的 SOP | 严格的角色 hand-off 协议（我们用 state graph 而非纯消息） |
| **CAMEL** (Li et al. 2023) | 双 agent role-play 对话生成 instruction-following 数据 | 没采纳——我们的数据 pipeline 来自 trajectory + LessonBook，不是 role-play | 双 agent 对话本身 |
| **AutoAct** (Qiao et al. 2024) | 让 meta-agent 自动 instantiate 工作流 agent | CriticAgent 是 meta-agent 的雏形 | 完全 self-instantiation；我们的 agent 集合固定，演化的是策略不是拓扑 |
| **AgentVerse** (Chen et al. 2024) | 协作 + 竞争 multi-agent 评估框架 | 我们的 OrchestratorAgent vs ValidatorAgent 是隐含的竞争（不同判据） | 显式的 agent-vs-agent tournament（v0.3 可能加） |
| **Self-Discover** (Zhou et al. 2024) | meta-agent 组合自己的 reasoning structure | CriticAgent 用规则扫，未来 v0.3 可以让它生成自定义检查器 | LLM-driven 自合成 reasoning module |

---

## 11. Self-Loop Evaluation / Evolution —— v0.2 增强的 5 个机制

§3 介绍了我们原有的"三层 self-eval"，但那只是**单 run 内**的闭环。Agent evolution 真正需要的是**跨 run 的闭环**：每次 run 都留下可消费的数据资产，下次 run 更好。v0.2 这一轮新增 5 个机制，把这个闭环关上：

```
                    ┌─────────────────────────────────────────────┐
       │                                                          │
       │  run N                                                   │
       ▼                                                          │
  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐│
  │ Plan     │───▶│  Compose   │───▶│  Trajectory │───▶│ Critic   ││
  │ (lessons │    │ (ensemble  │    │  log       │    │ (rules→  ││
  │  loaded) │    │  judge +   │    │            │    │  lessons)││
  │          │    │  pref log) │    │            │    │          ││
  └──────────┘    └───────────┘    └────────────┘    └────┬─────┘│
       ▲                                                  │      │
       │              ┌─────────────────────────┐         ▼      │
       │              │  LessonBook (cross-run) │◀────────┘      │
       └──────────────┴─────────────────────────┘                 │
                         │                                        │
                         └──── run N+1 ─────────────────────────────┘

       PreferenceLogger  ──── (winner, losers) JSONL  ────▶ DPO / IPO RM training (v0.3)
```

### 11.1 详细对照表 — 每个机制 → 它落到哪个相关工作 → 我们改了什么

| # | 机制 | 文件 | 起源工作 | 我们的实现要点 |
|---|---|---|---|---|
| **#1** | **LessonBook** — 持久化跨 run 文本反思记忆，按 scope/keyword 检索注入 prompt | `memory/lessons.py` + `pipeline/plan.py` | **Reflexion** (Shinn et al., NeurIPS 2023) — 单 agent 跨 episode 的 verbal episodic memory | 我们把它**跨 run** 而不是跨 episode；按 keyword overlap 检索；schema 是 JSONL stable record，未来 RL/RM 训练直接消费 |
| **#2** | **CriticAgent** — post-hoc 扫 trajectory，识别 5 类失败模式，写 Lesson | `agents/critic.py` | **Self-Refine** (Madaan et al. NeurIPS 2023) + **Reflexion**，但**角色分离**思路来自 **Self-Discover** (Zhou et al. 2024) | 关键：critic 是**独立 agent**，不和执行 agent 共享 LLM 上下文，避免 self-grading bias；v0.1 用规则触发（5 个 trigger），v0.2 可接 LLM 抽象 |
| **#3** | **EnsembleRewardModel** — K 个 reward 模型 mean ± std + 多数票 quorum | `models/reward/ensemble.py` | **Multi-Agent Debate** (Du et al. 2023) + **LLM-as-judge bias** (Zheng et al. NeurIPS 2023) | 仅 1 轮（不像 MAD 多轮）保持 cost 可控；disagreement std 暴露给 caller 作为 active-learning signal |
| **#4** | **PreferenceLogger** — multi-candidate segment 自动写 (winner, losers) 三元组 | `utils/preferences.py` + `agents/editor.py` | **DPO** (Rafailov et al. NeurIPS 2023) + **IPO** (Azar et al. 2023) | 0 人工标注；schema 兼容 HuggingFace `trl.DPOTrainer`；每次 EditorAgent 跑都增量积攒 |
| **#5** | **Self-Consistency Screenwriter** — K 次采样 + 按 energy_level 取众数 | `agents/screenwriter.py` | **Self-Consistency Improves CoT** (Wang et al. ICLR 2023) | 把单 LLM call 变成 ensemble；只对最高 leverage 的 agent（Screenwriter）应用，控制总成本 |

### 11.2 为什么这 5 个组合起来能 close the loop

* **#1 + #2** 解决了"跨 run 学到东西"的问题：CriticAgent 把这 run 的失败提炼成 Lesson，Plan 阶段下次自动消费。
* **#3** 把单 judge → 多 judge，降低 reward hacking 风险；disagreement std 还顺便给 active learning 提供信号。
* **#4** 自动积累 DPO/IPO 训练数据；v0.3 RM fine-tune 不再需要人工标注 bootstrap。
* **#5** 把高代价决策（global structure）变成 ensemble，对应 Bayesian 视角下的"在最关键节点降方差"。

合起来：**evolve 不只是依赖单 run 的 reward**，而是依赖**跨 run 的 lesson、跨 judge 的共识、跨 candidate 的 preference**。三个独立的演化信号叠加，能避免任何一种走偏。

### 11.3 路线图 v0.3 / v0.4 / v0.5（每条都建立在这 5 个机制上）

| 阶段 | 用什么 | 训练什么 |
|---|---|---|
| **v0.3** RM training | accumulated `preferences.jsonl`（来自 PreferenceLogger #4） + `trajectory.jsonl` 的 segment_finalized 摘要 | EditingQualityRM（LoRA on Qwen-VL），用 DPO/IPO 目标 |
| **v0.4** RL training | 全套 trajectory + EditingQualityRM + LessonBook | EditorAgent policy 通过 RAGEN/StarPO 或 verl/GRPO 更新 |
| **v0.5** Personalisation | 每个 user_id 的 trajectory + 用户偏好（PreferenceLogger 加 user_id 字段） | per-user LoRA fine-tune EditorAgent + Screenwriter |

---

## 附录 A · 文档索引

| 文档 | 看什么 |
|---|---|
| [`README.md`](../README.md) | 快速安装 / quickstart / 模块图 |
| [`LongVideoEditAgent_DESIGN.md`](../LongVideoEditAgent_DESIGN.md) | 原始设计文档 |
| [`docs/SYSTEM_GUIDE.md`](./SYSTEM_GUIDE.md) | **本文档** — 数据流 + 核心机制 + 相关工作 |
| [`docs/architecture_tour.md`](./architecture_tour.md) | 设计文档 § ↔ 代码文件映射 |
| [`docs/decisions.md`](./decisions.md) | 14 条 design decision 取舍 |
| [`docs/dependencies.md`](./dependencies.md) | 每个模块用的真实开源库 |

## 附录 B · 主要引用（按出现顺序）

| 引用 | 类型 | 链接 |
|---|---|---|
| DIRECT (Mashup-Bench) | repo | https://github.com/AK-DREAM/DIRECT |
| FilmAgent | repo | https://github.com/HITsz-TMG/FilmAgent |
| MovieAgent | repo | https://github.com/showlab/MovieAgent |
| GLANCE | repo (announced) | https://github.com/ZihaoLinQZ/GLANCE-Video-Editing-Agent |
| OmniWeaving | repo | https://github.com/Tencent-Hunyuan/OmniWeaving |
| Wan2.6 | repo | https://github.com/Wan-Video |
| PySceneDetect | repo | https://github.com/Breakthrough/PySceneDetect |
| All-In-One | repo | https://github.com/mir-aidj/all-in-one |
| LangGraph | docs | https://langchain-ai.github.io/langgraph/ |
| RAGEN | repo | https://github.com/mll-lab-nu/RAGEN |
| verl | docs | https://verl.readthedocs.io |
| ReAct | paper | Yao et al., ICLR 2023 |
| Reflexion | paper | Shinn et al., NeurIPS 2023 |
| Self-Refine | paper | Madaan et al., NeurIPS 2023 |
| STaR | paper | Zelikman et al., NeurIPS 2022 |
| ReST | paper | Gulcehre et al., arXiv 2023 |
| ReST-EM | paper | Singh et al., arXiv 2023 |
| Voyager | paper | Wang et al., arXiv 2023 |
| Constitutional AI | paper | Bai et al., Anthropic, arXiv 2022 |
| RLAIF | paper | Lee et al., Google, arXiv 2023 |
| G-Eval | paper | Liu et al., arXiv 2023 |
| Multi-Agent Debate | paper | Du et al., arXiv 2023 |
| Process Reward Models | paper | Lightman et al., arXiv 2023 |
| AutoGen | repo | https://github.com/microsoft/autogen |
| InsightFace | repo | https://github.com/deepinsight/insightface |
| open_clip | repo | https://github.com/mlfoundations/open_clip |
| RAFT (torchvision) | docs | https://pytorch.org/vision/stable/models/raft.html |
| U²-Net | repo | https://github.com/xuebinqin/U-2-Net |
| Qwen2-VL | model | https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct |
| ShotVL/ShotBench | model | https://huggingface.co/Vchitect |
| Self-Consistency Improves CoT | paper | Wang et al., ICLR 2023 |
| DPO | paper | Rafailov et al., NeurIPS 2023 |
| IPO | paper | Azar et al., arXiv 2023 |
| Multi-Agent Debate | paper | Du et al., arXiv 2023 |
| LLM-as-judge bias | paper | Zheng et al., NeurIPS 2023 |
| DyLAN | paper | Liu et al., arXiv 2024 |
| MetaGPT | paper | Hong et al., ICLR 2024 |
| CAMEL | paper | Li et al., NeurIPS 2023 |
| AutoAct | paper | Qiao et al., arXiv 2024 |
| AgentVerse | paper | Chen et al., ICLR 2024 |
| Self-Discover | paper | Zhou et al., arXiv 2024 |
| Skywork-Reward | model | https://huggingface.co/Skywork |
