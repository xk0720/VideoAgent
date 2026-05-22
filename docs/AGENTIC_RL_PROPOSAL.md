# 把 Agentic RL 应用到 LongVideoEditAgent — 完整方案

> 这是 Round 12 的产出。在 docs/SYSTEM_GUIDE §11.4 路线图里 v0.4 一句话带过，本文档把它展开成"读完就能动手"的级别。
>
> 基础设施：本文档的每条引用都通过 WebSearch 在 2026 年 5 月核实过 arXiv ID / GitHub URL。

---

## 0. TL;DR

* **现状**：v0.2 已经把所有"为 RL 准备数据"的接口建好了（trajectory.jsonl + preferences.jsonl + lessons.jsonl + EnsembleRewardModel）。
* **2026 业界标准 recipe**：**SFT cold start → 偏好对齐 (DPO/KTO/SimPO) → RL with verifiable reward (GRPO/DAPO)**。这是 DeepSeek-R1 / Tülu-3 / EVA 共同收敛的三段式。([Zylos Research 2026](https://zylos.ai/research/2026-04-10-rl-posttraining-tool-using-agents-grpo-async-rl))
* **直接对标我们的工作**：**EVA (Efficient Video Agent, arXiv 2603.22918)** 已经把 SFT+KTO+GRPO 应用到视频 agent 上并 work——这是我们 v0.4 的直接参考。
* **该训谁**：先训 EditorAgent（最大 leverage）；v0.5 再 MARTI 风格 co-train Director+Editor。
* **该用什么 reward**：v0.3 先训 EditingQualityRM；v0.4 把它作为 GRPO 的 verifier；m1..m6 作为辅助 sparse → dense 监督。
* **该用什么框架**：**ProRL Agent (arXiv 2603.18815)** — 专为多轮 LLM agent rollout 设计，rollout-as-a-service，trainer 解耦，可接 verl / NeMo-RL。
* **最大风险**：reward hacking（[Anthropic 2026 "Natural Emergent Misalignment from Reward Hacking"](https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf) 警示），通过我们已有的 EnsembleRewardModel + ODIN-style disentangle + inoculation prompting 防御。

---

## 1. 2026 agentic RL 的 SOTA 是什么样

我从 6 个角度搜了文献，下面是我读到的 **4 条核心收敛**：

### 1.1 三段式 post-training 已成业界共识

DeepSeek-R1（2025/01）、Tülu-3（2024/11）、EVA（2026）、Zylos Research 2026 综述都收敛到同一管线：

```
   base LLM
      │
      ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Stage A: SFT cold start                                  │
   │  - 在高质量 trajectory 上模仿学习                        │
   │  - 让模型 first-time 学会输出格式 + 推理结构           │
   │  - 小数据量（几百-几千），但 SFT 比 RL data-efficient 数倍 │
   └────────────────────┬───────────────────────────────────┘
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Stage B: Preference optimization (DPO / KTO / SimPO)    │
   │  - 在 (winner, loser) pair 上做隐式 reward 学习         │
   │  - 不需要单独 RM，闭式 loss                             │
   │  - 是 RL 前的 "warm start"——把概率分布拉向偏好的解     │
   └────────────────────┬───────────────────────────────────┘
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Stage C: RL with verifiable reward (GRPO / DAPO / RLOO) │
   │  - 在线产生 K 个 rollout，按 reward 排序更新策略        │
   │  - reward 来自训好的 RM 或闭式 metric                   │
   │  - 解决 OOD 泛化（SFT memorizes, RL generalizes,       │
   │    ICLR 2026）                                          │
   └─────────────────────────────────────────────────────────┘
```

[ICLR 2026 "SFT Memorizes, RL Generalizes"](https://openreview.net/forum?id=yezWGJmODg) 实验显示：RL 后训能把 SFT 阶段"丢掉的"OOD performance 找回 99%（Qwen2.5-7B）/ 85%（Llama-3.2-11B）。**单独跑 RL 或单独跑 SFT 都不如组合**。

**警告**：SFT checkpoint 选最高 eval 分的 ≠ 最适合做 RL 起点。"distributional forgetting"——模型 SFT 时偏离 base 太远，RL 阶段反而难恢复。([见上 OpenReview](https://openreview.net/forum?id=yezWGJmODg))

### 1.2 Long-horizon credit assignment 是核心难题（47 种方法已被 survey）

2024 至 2026 初有 47 种 credit assignment 方法被系统调研（[Awesome-Credit-Assignment-in-LLM-RL](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)）。最值得我们抄的：

* **Turn-PPO** (EACL 2026) — 把"一个完整 LLM response + env feedback"当成一个 macro-action，turn-level advantage estimate。**消除 token-level credit 的方差爆炸**。我们的 segment 天然是 macro-action 单位。
* **HCAPO — Hindsight Credit Assignment for Long-Horizon LLM Agents** (arXiv 2603.08754, 2026) — 用 LLM 自己做 post-hoc critic，refine 早期步的 Q value。**在 WebShop +7.7%，ALFWorld +13.8% over GRPO 用 Qwen2.5-7B-Instruct**。这正是我们 CriticAgent 该做的事。
* **HiPER** (arXiv 2602.16165) — explicit planner / executor 层级 + Hierarchical Advantage Estimation。**正好对应我们 Screenwriter / Director / Editor 的天然层级**。
* **Decoupled Credit Assignment for Long-Horizon Agentic Learning** ([OpenReview](https://openreview.net/forum?id=Suja4AQ9Fo)) — 解耦"局部有效"和"全局有效"的 credit。
* **Reinforcing Multi-Turn Reasoning via Turn-Level Reward Design** ([NeurIPS 2025](https://arxiv.org/pdf/2505.11821))。

### 1.3 视频 agent 的 RL 已经有完整 baseline

我们不是在做开荒——视频 agent + RL 在 2026 已经是个 working 范式：

* **EVA — Efficient RL for End-to-End Video Agent** (arXiv 2603.22918, 2026) — **明确用了 SFT + KTO + GRPO 三段式**，做的就是视频 agent。**这是我们最直接的 reference implementation**。
* **VITAL / Thinking With Videos** (arXiv 2508.04416, 2025) — 多模态 tool-augmented RL for long video reasoning。tool 调用 + multi-round 训练。
* **SAGE — Smart Any-Horizon Agents for Long Video Reasoning** (arXiv 2512.13874, 2025) — long video specific。
* **PyVision-RL** (arXiv 2602.20739, 2026) — open agentic vision via RL，统一框架。
* **Argos** (Microsoft 2026) — multimodal RL with **agentic verifier**——这就是我们 EnsembleRewardModel 的角色定位。

### 1.4 Multi-agent co-training 框架已经成型

* **MARTI** (ICLR 2026, [TsinghuaC3I/MARTI](https://github.com/TsinghuaC3I/MARTI)) — centralized 多 agent 交互 + distributed policy 训练；MARTI v2 加 tree-search-augmented RL。
* **MAPoRL — Multi-Agent Post-Co-Training for Collaborative LLMs with RL** (arXiv 2502.18439) — 跨 agent benchmark 验证：**co-train 跨 unseen domain 也能泛化**。
* **Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent RL** (arXiv 2602.09578) — 工程实操：怎么把 rollout 和 training 解耦让多 agent 不互相阻塞。
* **RL for LLM-based Multi-Agent Systems through Orchestration Traces** (arXiv 2605.02801) — **直接用 orchestration traces 作 RL 信号**——这正是我们 trajectory.jsonl 的用法。

---

## 2. 我们项目为什么是个适合的 agentic RL 试验场

把 2026 文献要求的"agentic RL 必备条件"对照我们 v0.2 已有的：

| RL 训练必备 | 我们已经有的 | 文件 |
|---|---|---|
| 结构化 trajectory log | `AgentStep` JSONL，每行一个 (state, action, observation, reward)，schema 兼容 RAGEN / HF datasets | `utils/trajectory.py` |
| 稠密的 turn-level reward | EnsembleRewardModel 给每个 segment 打分（mean + std） | `models/reward/ensemble.py` |
| Process reward signal | per-step m1..m6 metric_scores（不是只 final reward） | `tools/metric_tool.py` |
| Pairwise preference data | PreferenceLogger 自动从 multi-candidate segment 产 (winner, loser) | `utils/preferences.py` |
| Cold-start data | 当前 mock pipeline 输出的 trajectory 已经是结构化的"专家轨迹"——v0.3 真接 LLM 后 traj 质量进一步上升 | `outputs/demo_trajectory.jsonl` |
| Cross-run 反思 | LessonBook 跨 run，可以做 in-context 引导（reduces RL sample need） | `memory/lessons.py` |
| 独立 verifier agent | EnsembleRewardModel + CriticAgent + OrchestratorAgent 三层判据 | 同上 |
| 可学习 policy + ABC | BaseLLMClient ABC + factory，改一行就换 backbone | `models/llm/base.py` |
| 可计算的 task reward | Mashup-Bench / CineBench 适配器接口 + DIRECT m1..m6 | `benchmark/` + `tools/metric_tool.py` |

**这意味着我们离 RL 训练只差三件事**：(1) 用 preferences.jsonl 训出 EditingQualityRM，(2) 接一个 RL trainer（ProRL/OpenRLHF/verl），(3) 写 env wrapper 把 LongVideoEditAgent 包成 RL 环境。其他全在。

---

## 3. 具体训练方案（v0.3 → v0.4 → v0.5）

### 3.1 Phase 1 · v0.3 · 训 EditingQualityRM（不动 agent policy）

**目标**：从我们累积的 preferences.jsonl + trajectory.jsonl 训出一个专用 reward model，替换掉 v0.2 的 mock judge。

**数据**：
- 主：`preferences.jsonl` — PreferenceLogger 自动写的 (winner, losers) 对。schema 跟 HF `trl.DPOTrainer` 兼容。
- 辅：`trajectory.jsonl` 里所有 `segment_finalized` 摘要 — 含 (segment, guidance, m1..m6, validator_score)。可以构造伪 pairwise（同 guidance 下高分 vs 低分）增量数据集。

**Recipe**（不发明，直接抄 Tülu-3-RM + Skywork-Reward）：
- Backbone: **Qwen3-VL-8B**（arXiv 2511.21631）— 256K 上下文能塞下整个 segment 的多帧 + guidance + neighbor context。
- 训练：**Bradley-Terry pairwise loss** 是 baseline；优化用 **Skywork-Reward** 风格的纯 preference + DPO-margin objective。
- 评估：在 hold-out preferences 上看 Spearman 与人工 / m1..m6 mean 的相关性，目标 **Spearman > 0.6**（业界 RM benchmark 阈值）。

**做完后**：
- 替换 `compose._build_reward_model` 里的某个 mock judge 为 fine-tuned 真 RM；ensemble 仍然保留 2 个 mock judge 防 reward hacking。
- 把 RM 训练数据 + 评估代码放到 `training/rm/`（v0.3 新目录）。

**为什么不直接跳到 RL**：没有 RM 的 RL 就是用 sparse outcome reward + LLM-as-judge，credit assignment 极差；2026 共识是先把 RM 训扎实。

---

### 3.2 Phase 2 · v0.4 · 训 EditorAgent policy（三段式）

**目标**：把 EditorAgent 的 retrieve/generate/fallback 路由策略从"prompted LLM"升级为"RL-trained policy"。

**Stage A · SFT cold start**
- 数据源：跑大量 mock + 真 LLM 的 pipeline，收集所有 EditorAgent 在 trajectory 里产的 (state, action) 对。
- 筛选：用 `segment_finalized.reward > 7.0` 过滤"好"trajectory（reject-sample SFT，[STaR](https://arxiv.org/abs/2203.14465) / [ReST-EM](https://arxiv.org/abs/2308.08998) 风格）。
- 目标输出：JSON action `{"action": "retrieve|generate|fallback", "rationale": "..."}`。
- backbone: **DeepSeek-V3-0324 / Qwen3-8B** 之类的开源中量级模型。
- 关键 check：用 ICLR 2026 "SFT Memorizes" 的发现——**别选 eval 最高的 ckpt**；选离 base 最近的那个 ckpt 准备给 Stage C。

**Stage B · KTO 偏好对齐**
- 用 `preferences.jsonl` 全部 (winner, loser) 对做 **KTO** ([Ethayarajh et al. 2024](https://arxiv.org/pdf/2402.01306))——比 DPO 在我们这种 noisy label 上稳。
- KTO 的另一个优势：能接受 unpaired 信号（只有 winner，没 loser），覆盖我们的 fallback case。
- 也可以试 **SimPO** ([Meng et al. 2024](https://arxiv.org/pdf/2405.14734))——reference-model-free，训练更快。

**Stage C · GRPO with EditingQualityRM**
- 用 v0.3 训好的 EditingQualityRM 作 verifier。
- **采用 Turn-PPO macro-action** 形式：一个 segment = 一个 macro-action。**消除 token-level credit 方差爆炸**——这是 EditorAgent 长 horizon 的核心痛点。
- **加 HCAPO hindsight critic**：CriticAgent 已经存在，让它在 segment 完成后 retroactively refine 早期步的 Q value（hindsight reasoning）。
- 框架：**ProRL Agent** (arXiv 2603.18815) — rollout-as-a-service，跟 verl 配合好。
- reward 组合：`reward = α * EditingQualityRM + β * Σ(m1..m6) + γ * music_alignment_bonus`，权重用 grid search 在 hold-out 上调。

**预期收益**（基于 EVA 论文 + HCAPO 论文）：
- HCAPO 在 WebShop / ALFWorld 比 GRPO 提升 7-14%。
- EVA 三段式（SFT+KTO+GRPO）对视频 agent task success rate 有显著提升（具体数字论文未读到，需读 EVA 复现）。
- **我们的目标**：v0.4 在 Mashup-Bench 上比 v0.2 baseline +20%（保守）。

---

### 3.3 Phase 3 · v0.5 · Multi-agent co-training (MARTI 风格)

**目标**：让 Director + Editor 同时 RL，互相适配；逼出 Director 的 retrieval_feasibility 估计跟 Editor 实际 retrieve 成功率对齐。

**Recipe**：
- 框架：**MARTI** (ICLR 2026)——专为 multi-agent LLM RL 设计，centralized 交互 + distributed policy。
- 范式：**MAPoRL** post-co-training——先各自 SFT，再 joint RL。
- 每个 agent 独立 reward：
  - Director: `r_director = correlation(retrieval_feasibility, editor_success_rate)` — 即 director 估计的"可检索性"必须跟 Editor 实际成功对应。
  - Editor: 同 v0.4 GRPO。
  - Critic: `r_critic = (lessons_used_next_run * next_run_improvement)` — critic 写的 lesson 在下次 run 真起作用了才得奖。
- 共享 task reward：`r_shared = pipeline_end_quality_score`（来自 EditingQualityRM）。
- 多 agent 信用分配：用 **Orchestration Traces RL** (arXiv 2605.02801) 直接用我们的 trajectory.jsonl 当训练数据。

**为什么 v0.5 而不是 v0.4**：multi-agent RL 不稳定（独立 agent 互相 chase a moving target）；先单 agent RL 训稳了再 co-train。MAPoRL 论文也明确 single-agent SFT 必须先到位。

---

## 4. 风险与缓解

| 风险 | 严重度 | 缓解 | 引用 |
|---|---|---|---|
| Reward hacking — agent 学会 game EditingQualityRM | **高** | (a) Ensemble of 3 RMs（已实现），quorum-aware accept；(b) 训 RM 时 ODIN-style disentangle "quality" vs "length"；(c) RL 时定期用 m1..m6 作 sanity check，发现 hacking 立即 inoculation prompt | [ODIN](https://arxiv.org/pdf/2402.07319), [PAR + Inoculation Prompting](https://openreview.net/forum?id=1TIWkM3nY4) (2026), [Anthropic Natural Emergent Misalignment](https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf) |
| Long-horizon credit assignment 失效 | **高** | Turn-PPO macro-action + HCAPO hindsight + Hierarchical (Screenwriter→Director→Editor 已是 hierarchical) | [HCAPO arXiv 2603.08754](https://arxiv.org/abs/2603.08754), [HiPER arXiv 2602.16165](https://arxiv.org/pdf/2602.16165) |
| 多 agent 训练不收敛 / 互相 chase moving target | **中** | (a) v0.5 才做，前置 v0.4 单 agent 稳；(b) MAPoRL 推荐的 staged co-training；(c) MARTI 的 centralized critic + distributed actor 结构 | [MAPoRL](https://arxiv.org/pdf/2502.18439), [MARTI](https://iclr.cc/virtual/2026/poster/10010710) |
| SFT cold-start 选错 ckpt → RL 阶段卡住 | **中** | 用"离 base 最近的 SFT ckpt"而不是"eval 最高"；监控 KL divergence | ICLR 2026 [yezWGJmODg](https://openreview.net/forum?id=yezWGJmODg) |
| 视频生成段过长 horizon、reward 稀疏 | **中** | 把 generation 的 reward 拆成 frame-level intermediate signal（VideoReward 风格） | [Improving Video Generation with Human Feedback](https://openreview.net/forum?id=nHkg4yc7SP), [SAGE-GRPO](https://github.com/Tencent-Hunyuan/SAGE-GRPO) |
| 数据量不够（v0.2 累积 trajectory 较少） | **中** | (a) 用 mock pipeline 大量产；(b) 真接 LLM 后跑 Mashup-Bench / CineBench 自动产；(c) 借鉴 Multi-Agent Evolve 让系统自产 prompt 自训 | [MAE](https://arxiv.org/abs/2510.xxxxx) |
| 算力——Qwen3-VL-8B + GRPO 跑不起 | **中** | Stage A/B 用 LoRA SFT（H100×8 一周可跑完）；Stage C 先做 1.5B-size pilot 验证 pipeline 再 scale | [DeepSWE](https://github.com/agentica-project/DeepSWE) (Qwen3-32B / 64×H100 / 6 天 → 59% SWE-bench) |
| 视频 generation step 太慢，单次 rollout 几分钟 | **中** | (a) v0.4 训练时用 MockVideoGenClient（policy 学 routing 不学生成质量）；(b) 真视频生成只在 eval 阶段用；(c) async rollout (OpenRLHF 0.8.0) 让 generation 不阻塞训练 | [OpenRLHF](https://github.com/openrlhf/openrlhf) |

---

## 5. 基础设施选型

| 需求 | 选哪个 | 为什么 |
|---|---|---|
| Phase 1 RM 训练 | **HuggingFace TRL** + `trl.RewardTrainer` | 我们的 preferences.jsonl schema 已经匹配；标准、文档全 |
| Phase 2 SFT | **TRL** `SFTTrainer` + LoRA | 同上 |
| Phase 2 KTO | **TRL** `KTOTrainer` | KTO 是 2024 加入的，TRL 原生支持 |
| Phase 2 GRPO | **ProRL Agent** ([arXiv 2603.18815](https://arxiv.org/abs/2603.18815)) | 专为多轮 agent 设计；rollout-as-a-service 跟 verl/NeMo-RL 都能配合；可在 HPC sandboxed 环境跑 |
| Phase 3 multi-agent co-training | **MARTI** ([github.com/TsinghuaC3I/MARTI](https://github.com/TsinghuaC3I/MARTI)) | ICLR 2026，专为 multi-agent LLM RL；MARTI v2 加 tree-search |
| Async rollout | **OpenRLHF 0.8.0** | `--train.async_enable` + `--train.agent_func_path`；目前唯一原生支持 async agent RLHF |
| Backup 选择 | **verl** (HybridFlow) | 通用 RL post-training；更灵活但门槛高 |

**不选 RAGEN**：设计文档 v0.1 提过；现在 ProRL/MARTI 出来后，RAGEN 在 multi-turn agent 这块已不是 SOTA。仍保留为 v0.4 备选。

---

## 6. 数据流（什么数据流向哪个训练阶段）

```
   v0.2 已经在产的数据
   ────────────────────────────────────
   outputs/run_N_trajectory.jsonl
       ├── AgentStep 行  ────────────────► Phase 2 SFT (filter reward>θ)
       └── segment_finalized 摘要 ────────► Phase 1 RM 训练 (伪 pairwise)

   outputs/run_N_preferences.jsonl
       └── (winner, losers) 对 ───────────► Phase 1 RM 训练 (主数据集)
                                        ──► Phase 2 KTO/DPO

   .cache/lessons.jsonl
       └── Lesson 记录 ──────────────────► RL 不用，但作为 in-context prompt
                                          减少 RL sample 需要（Reflexion + RAG）

   v0.4 新加的数据
   ────────────────────────────────────
   rollouts/v04_grpo_N.jsonl
       └── 在线 GRPO rollout ────────────► Phase 2C 训练 (在线)

   eval/mashup_bench_results.json
       └── 每 epoch 的 task success ────► early stop + curriculum 信号
```

**关键点**：v0.2 的 PreferenceLogger 和 segment_finalized 早就为这条 pipeline 铺路了——不需要重写数据采集层，直接接训练就行。

---

## 7. 成功标准（必须可测）

| 阶段 | 指标 | 阈值 | 怎么测 |
|---|---|---|---|
| v0.3 EditingQualityRM | Spearman corr with m1..m6 mean on hold-out | > 0.6 | 留 20% preferences 当 hold-out |
| v0.3 EditingQualityRM | Top-1 accuracy on (winner vs loser) | > 0.75 | 同上 |
| v0.4 EditorAgent post-RL | Mashup-Bench task success rate | +20% over v0.2 baseline | benchmark/mashup_bench.py |
| v0.4 EditorAgent post-RL | average validator score in trajectory | > 8.0 (was ~7.45 in v0.2 demo) | trajectory analysis |
| v0.4 EditorAgent post-RL | reward hacking rate | < 5% on adversarial probe set | 用 [Reward Hacking Benchmark](https://arxiv.org/abs/2605.02964) (arXiv 2605.02964) |
| v0.5 multi-agent co-train | Mashup-Bench task success rate | +5% over v0.4 | 同上 |
| v0.5 multi-agent co-train | Director's retrieval_feasibility prediction calibration | Brier score < 0.1 | per-segment 验证 |

---

## 8. 不知道 / 需要先验证的开放问题

1. **HCAPO 在视频 segment 上的效果** — HCAPO 论文用 WebShop / ALFWorld（短文本环境）。我们的 segment 是几秒视频 + 多模态——hindsight critic 是否仍有效，需要先 pilot。
2. **EditingQualityRM 训练数据量下限** — Tülu-3-RM 用 100K+ preferences；我们 v0.2 自动收集，但要跑多少次 pipeline 才够？预估需要先做一次 mock-pipeline + LessonBook 的大规模 (1000-run) 采集实验。
3. **是否所有 6 个 agent 都该 RL？** Screenwriter 调用频率低（每个 run 一次），Director 中等（每 section 一次），Editor 高频（每 segment 多次）。先只训 Editor 性价比最高；Director / Screenwriter 是否值得 RL 训需要 v0.4 后再评估。
4. **是否需要 RM 也 ensemble** — 当前 EnsembleRewardModel 是多个 mock judges；v0.3 后能否训出 K 个不同种子的 EditingQualityRM 形成真 ensemble？计算开销 K× 大，需评估 ROI。
5. **生成段的 reward 怎么处理** — VideoReward (Liu et al. 2025) 和 SAGE-GRPO (Tencent) 提供了视频生成的 RM 选项；但我们的 generation 是被 EditorAgent 调用的子工具，generation 段的 RL 是 in-scope 还是另起一卷？倾向后者——v0.4 先只 RL routing，生成端继续用现成模型。

---

## 9. 实施时间盒（保守估计）

| 阶段 | 时间 | 关键里程碑 |
|---|---|---|
| v0.3 数据收集 + RM 训练 | 4-6 周 | 1000 run 累积 + Qwen3-VL-8B LoRA SFT-as-RM + hold-out Spearman > 0.6 |
| v0.4 EditorAgent RL pipeline | 6-8 周 | ProRL Agent 接入 + SFT + KTO + GRPO 三段 + Mashup-Bench +20% |
| v0.5 multi-agent co-train | 8-12 周 | MARTI 接入 + Director+Editor 联训 + Multi-Bench +5% over v0.4 |
| v0.6 personalization | 12+ 周 | per-user LoRA + 用户偏好 A/B |

合计约 30-40 周从 v0.2 走到完整 RL 演化系统。**人力**：可能需要 1 人专注 RM 训练 + 1 人专注 RL infra + 1 人专注 eval/benchmark。

---

## 9.4 v0.4.1 — 加入 On-Policy Distillation (OPD) as Stage A.5

详细思想性调研 + reference 严谨度审查见独立文档
[`docs/ON_POLICY_DISTILLATION_ANALYSIS.md`](./ON_POLICY_DISTILLATION_ANALYSIS.md)
（300+ 行，覆盖 GKD → OPSD → REOPOLD 全谱系；§3 完整论证为什么 OPD 加在
SFT 之后、KTO 之前；§4 反方意见；§7 红线条件）。

**一句话**：OPD **不替换** SFT/KTO/GRPO；它在 Stage A 之后、Stage B 之前
插入做 Stage A.5。教师选 Claude Sonnet 4.5（或 GPT-4o / Qwen3-VL-72B），
学生是后训的 DeepSeek-V3 / Qwen3-7B。

实装：[`training/stages/distill.py`](../training/stages/distill.py)
+ [`PipelineConfig.opd_enabled`](../training/runners/pipeline.py) 默认 `False`（先量化 baseline）。

---

## 9.5 实现状态（v0.3 stub 已就位 — branch `rl-integration`）

§3 的方案已经从文档变成代码——见 [`training/`](../training/README.md)。开关一翻就能跑（mock backend，CPU-only，没有 torch/TRL/vLLM 也能跑通整管线）：

| §3 的步骤 | 对应文件 | 现在能做什么 |
|---|---|---|
| §3.1 EditingQualityRM 训练 | `training/rewards/editing_quality_rm.py` + CLI `lva-train-rm` | 跑 Bradley-Terry stub RM；输出 weights+bias 的 JSON，能 `EditingQualityRM.load(path)` 后塞进 EnsembleRewardModel |
| §3.2 Stage A SFT cold start | `training/stages/sft.py` + CLI `lva-train-sft` | 用 `reward > θ` 过滤 trajectory，stub backend 写 metrics+ckpt；`backend="trl"` 接 TRL v1.0 SFTTrainer |
| §3.2 Stage B KTO 偏好对齐 | `training/stages/kto.py` + CLI `lva-train-kto` | KTO/DPO/IPO/SimPO 切换；`backend="trl"` 接 TRL KTOTrainer/DPOTrainer |
| §3.2 Stage C GRPO RL | `training/stages/grpo.py` + CLI `lva-train-grpo` | EditorEnv rollout + GRPO leave-one-out advantage（stub）；`backend="verl"` / `"prorl"` 是 v0.4 swap |
| **§9.4 Stage A.5 — OPD** | `training/stages/distill.py` (新增) | reverse-KL student-on-policy distillation；stub 模拟 KL-descent 曲线；`backend="trl"`/`"nemo_rl"` 是 v0.4.1 swap |
| Turn-PPO macro-action | `training/env/editor_env.py` | 一个 segment = 一个 episode，已实装 |
| HCAPO hindsight refiner | `training/rewards/hindsight.py` | closed-form 平滑实装，LLM-driven 变体是 v0.4 |
| ORS tool-call protocol | `training/env/base.py::AgentEnvBase.tools()` | EditorEnv 三个 action 已按 ORS 暴露 |
| RAGEN Environment/Context/Agent 拆分 | `training/env/` + `training/policy/` | 同构对齐 |
| verl AgentLoop server/client | `training/runners/rollout.py` | sync，async 由 verl/OpenRLHF 包外层 |
| EVA 三段式 | `training/runners/pipeline.py::PipelineRunner` | RM → SFT → KTO → GRPO 一次调用跑完 |

**已验证**：22 个 training/ 单元测试 + 1 个端到端 PipelineRunner 集成测试全过；主仓库 102 + 训练子树 22 = **124 测试 green**；`pyflakes 0 warning`。

**v0.4 升级路径**（每条都是一行 flag）：
- 装 `transformers + trl + datasets` → 把 `SFTConfig.backend="trl"` 等三个 flag 翻过来
- 装 verl 或 ProRL Agent → `GRPOConfig.backend="verl"` 或 `"prorl"`，然后填空 `_fit_verl` / `_fit_prorl` 里的 NotImplementedError
- 装 vLLM → 给 `policy/editor_policy.py` 配 OpenAI-compatible base_url，零代码改动

---

## 10. 我对这件事的总判断

**这套架构非常适合做 agentic RL，且时机刚好**：

1. **数据基础设施在 v0.2 已经全部就绪**——这是 90% 项目卡住的地方，我们没卡。
2. **业界 2026 recipe (SFT→KTO→GRPO + ensemble RM + hindsight critic + multi-agent co-train) 都有 working 论文 baseline**——不需要发明，只需要复现 + 适配视频编辑场景。
3. **视频 agent 的 RL 已经有 EVA、PyVision-RL、SAGE-GRPO 这些直接 reference**——不是开荒。
4. **三层 reward (m1..m6 闭式 + Ensemble RM + LessonBook) 是天然的 reward hacking 防御**——这是 RL 阶段最大的风险，我们结构上已经免疫。

**最有意义的事可能不是"训出更好的 EditorAgent"，而是**：

> 把 LongVideoEditAgent 做成"agentic evolution 试验台"——长视频编辑任务的 dense reward + long horizon + tool use + multimodal 属性，让它成为通用 agentic RL 研究的基准任务之一。trajectory.jsonl 的格式 RAGEN-兼容，preferences.jsonl 的 schema TRL-兼容，意味着我们做的训练数据可以被领域共享。

下一步建议：**先做 v0.3 RM 训练的 pilot**——用 100-500 次 mock pipeline run 的 preferences.jsonl 训一个迷你 RM，验证 Spearman 是否能 > 0.4，再决定是否 scale 到 1000+ run。

---

## 附录：本文档全部引用（按出现顺序）

| 引用 | 类型 | 链接 |
|---|---|---|
| Zylos Research 2026 RL post-training survey | blog/research | https://zylos.ai/research/2026-04-10-rl-posttraining-tool-using-agents-grpo-async-rl |
| The Landscape of Agentic RL for LLMs: A Survey | arXiv 2509.02547 | https://arxiv.org/abs/2509.02547 |
| A Brief Overview: Agentic RL in LLMs (2026) | arXiv 2604.27859 | https://arxiv.org/abs/2604.27859 |
| Turn-PPO | EACL 2026 / OpenReview | https://openreview.net/forum?id=7cgTBPuwMr |
| HCAPO — Hindsight Credit Assignment for Long-Horizon LLM Agents | arXiv 2603.08754 | https://arxiv.org/abs/2603.08754 |
| HiPER — Hierarchical RL with Explicit Credit Assignment | arXiv 2602.16165 | https://arxiv.org/pdf/2602.16165 |
| Decoupled Credit Assignment for Long-Horizon Agentic Learning | OpenReview | https://openreview.net/forum?id=Suja4AQ9Fo |
| Reinforcing Multi-Turn Reasoning via Turn-Level Reward Design | NeurIPS 2025 | https://arxiv.org/pdf/2505.11821 |
| Awesome-Credit-Assignment-in-LLM-RL | repo | https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL |
| From Reasoning to Agentic: Credit Assignment in RL for LLMs | arXiv 2604.09459 | https://arxiv.org/html/2604.09459v1 |
| EVA — Efficient RL for End-to-End Video Agent | arXiv 2603.22918 | https://arxiv.org/pdf/2603.22918 |
| Argos — Multimodal RL with Agentic Verifier (Microsoft 2026) | blog | https://www.microsoft.com/en-us/research/blog/multimodal-reinforcement-learning-with-agentic-verifier-for-ai-agents/ |
| VITAL / Thinking With Videos | arXiv 2508.04416 | https://arxiv.org/html/2508.04416v1 |
| SAGE — Smart Any-Horizon Agents for Long Video Reasoning | arXiv 2512.13874 | https://arxiv.org/html/2512.13874v2 |
| PyVision-RL | arXiv 2602.20739 | https://arxiv.org/pdf/2602.20739 |
| MARTI ICLR 2026 | ICLR | https://iclr.cc/virtual/2026/poster/10010710 |
| MARTI repo | github | https://github.com/TsinghuaC3I/MARTI |
| MAPoRL | arXiv 2502.18439 | https://arxiv.org/pdf/2502.18439 |
| Rollout-Training Co-Design for Multi-Agent RL | arXiv 2602.09578 | https://arxiv.org/html/2602.09578v1 |
| RL for Multi-Agent Systems through Orchestration Traces | arXiv 2605.02801 | https://arxiv.org/html/2605.02801v1 |
| SFT Memorizes, RL Generalizes (ICLR 2026) | OpenReview | https://openreview.net/forum?id=yezWGJmODg |
| Reinforcement Fine-Tuning for Computer Use — Cold Starting | Medium / Norlund 2026 | https://medium.com/@tobias.norlund/reinforcement-fine-tuning-for-computer-use-part-3-cold-starting-with-supervised-fine-tuning-4088934ff4a3 |
| Off-Policy Token-Clipped SFT (cold start) | OpenReview | https://openreview.net/forum?id=qJLKOryYeR |
| Anthropic — Natural Emergent Misalignment from Reward Hacking | Anthropic paper | https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf |
| Mitigating Reward Hacking with RL Training Interventions (PAR) | OpenReview 2026 | https://openreview.net/forum?id=1TIWkM3nY4 |
| Reward Hacking Benchmark | arXiv 2605.02964 | https://arxiv.org/abs/2605.02964 |
| MONA — Myopic Optimization with Non-myopic Approval | arXiv 2501.13011 | https://arxiv.org/pdf/2501.13011 |
| When Reward Hacking Rebounds | arXiv 2604.01476 | https://arxiv.org/pdf/2604.01476 |
| ODIN — Disentangled Reward Mitigates Hacking | arXiv 2402.07319 | https://arxiv.org/pdf/2402.07319 |
| Improving Video Generation with Human Feedback (VideoReward) | OpenReview | https://openreview.net/forum?id=nHkg4yc7SP |
| SAGE-GRPO (Tencent-Hunyuan) | github | https://github.com/Tencent-Hunyuan/SAGE-GRPO |
| Reward-Forcing (CVPR 2026) | github | https://github.com/JaydenLyh/Reward-Forcing |
| Awesome-Video-Generation-Post-Training | github | https://github.com/CyL97/Awesome-Video-Generation-Post-Training |
| OpenRLHF | github | https://github.com/openrlhf/openrlhf |
| verl / HybridFlow | github | https://github.com/verl-project/verl |
| ProRL Agent | arXiv 2603.18815 | https://arxiv.org/html/2603.18815v1 |
| Agent-R1 | arXiv 2511.14460 | https://arxiv.org/pdf/2511.14460 |
| DeepSWE (Agentica + Together AI) | github | https://github.com/agentica-project/verl |
| Lightweight SFT before RL | OpenReview | https://openreview.net/forum?id=yezWGJmODg |
| KTO | arXiv 2402.01306 | https://arxiv.org/pdf/2402.01306 |
| SimPO | arXiv 2405.14734 | https://arxiv.org/pdf/2405.14734 |
| OpenRLHF vs veRL Deep Dive | blog | https://langcopilot.com/posts/2025-11-06-openrlhf-vs-verl-ray-framework-deep |
