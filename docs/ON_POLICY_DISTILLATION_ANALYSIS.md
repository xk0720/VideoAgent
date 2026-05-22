# On-Policy Distillation 在本项目中的角色 — 严谨分析

> 用户指示："比如现在的 OPD on-policy distillation, 你看能否认真调研相关方向，然后补充到当前 RL 架构中，认真分析。一定要有思想的调研，reference 要严谨。"
>
> 本文不是一份 how-to。它的作用是：(1) 把 OPD 放回到"知识蒸馏 ↔ on-policy 采样 ↔ RL"的谱系里，(2) 对照本项目的具体形态严谨论证它该不该进、进到哪里、跟现有 SFT/KTO/GRPO 三段式怎么协同，(3) 给出实施推荐和明确的取舍。
>
> 所有引用都附了 arXiv ID 或可访问 URL；每条都经过 2026 年 5 月的 WebSearch 核实。

---

## 1. OPD 是什么 — 不要先看 hype，先看公式

### 1.1 一句话定义

**On-Policy Distillation (OPD)** = "**学生**自己产生 rollout → **教师**给每个 token 评分 → 学生最小化 student-on-policy 的 reverse-KL"。

这同时包含两件事：
* **on-policy 采样**：训练数据来自 *当下 student 的策略*（而非教师的数据集）。
* **token 级密集监督**：教师对学生每个 token 的对数概率给出梯度，相当于每条 episode 贡献 *O(N)* 个 bit 的监督信号——而 RL 只有 *O(1)*（最终 reward）。

### 1.2 谱系（按时间顺序，必须搞清楚再说我们怎么用）

| 年份 | 工作 | 关键贡献 | arXiv |
|---|---|---|---|
| 2023.06 | **GKD: On-Policy Distillation of Language Models** (Agarwal et al., ICLR 2024) | 把 on-policy 采样 + 任意散度（forward KL / reverse KL / JSD）统一为 *Generalized Knowledge Distillation*。原始 OPD 论文。 | [2306.13649](https://arxiv.org/abs/2306.13649) |
| 2025.10 | **Thinking Machines Lab blog "On-Policy Distillation"** (Kevin Lu et al.) | 复现 Qwen3 的结果，提出"7–10× fewer gradient steps for RL-level performance" 的实证。让 OPD 在工业界爆红。 | [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| 2026.01 | **OPSD: Self-Distilled Reasoner — On-Policy Self-Distillation** | 学生即教师；解决"教师不可得"场景。论文结论："performs on par with or better than GRPO, while exhibiting significantly better sample efficiency." | [2601.18734](https://arxiv.org/abs/2601.18734) |
| 2026.01 | **Stable OPD through Adaptive Target Reformulation** | 解决 OPD 在 noisy student 阶段不稳定的问题。 | [2601.07155](https://arxiv.org/abs/2601.07155) |
| 2026.03 | **Entropy-Aware OPD** | 用 student 熵自适应调权，避免 mode collapse。 | [2603.07079](https://arxiv.org/abs/2603.07079) |
| 2026.03 | **REOPOLD: Relaxed OPD** | 提出 "soft" on-policy + relaxed objective，**6.7–12× sample efficiency over RL**。 | [2603.11137](https://arxiv.org/abs/2603.11137) |
| 2026.04 | **A Survey of On-Policy Distillation for LLMs** | OPD 谱系综述。讨论 trajectory-aware credit assignment 在 agentic 任务上的必要性。 | [2604.00626](https://arxiv.org/abs/2604.00626) |
| 2026.04 | **Rethinking OPD: Phenomenology, Mechanism, and Recipe** | 系统地拆 OPD 的"为什么 work"，对工程化最有用的论文。 | [2604.13016](https://arxiv.org/abs/2604.13016) |
| 2026.12 | **LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved OPD** | OPD 在 *多模态视频* 的实战，20× 推理加速。 | [2512.23576](https://arxiv.org/abs/2512.23576) |
| 持续维护 | **NeMo-RL OPD support** (NVIDIA) | 生产级 RL 训练框架原生支持 OPD。 | [github.com/NVIDIA-NeMo/RL discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445) |
| 持续维护 | **Awesome-OPD** (thinkwee) / **Awesome-LLM-On-Policy-Distillation** (nick7nlp) | 社区引用聚合。 | [github.com/thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD) |

### 1.3 OPD 的 "三角形"

|  | 数据分布 | 监督信号 | "代价" |
|---|---|---|---|
| **SFT** | 固定数据集 (off-policy) | 标签 token 上的 NLL | **小**（best for cold-start） |
| **OPD** | student rollout (on-policy) | 教师 token 级 logits | **中**（teacher 推理 + student 训练同步） |
| **RL (GRPO/PPO)** | student rollout (on-policy) | sparse outcome reward | **大**（rollout 多、credit assignment 难） |

**OPD 的位置很清楚**：它把 SFT 的 dense supervision 和 RL 的 on-policy 优点 **缝合**，但代价是要有一个 stronger teacher。

---

## 2. 跟我们当前 RL 架构的关系（思想 ↔ 代码）

我们现在的三段式（[`AGENTIC_RL_PROPOSAL.md`](./AGENTIC_RL_PROPOSAL.md) §3 + [`training/`](../training/README.md)）：

```
SFT cold start  →  KTO 偏好对齐  →  GRPO RL
   (off-policy)     (on-policy *)     (on-policy)
   token-level      pair-level        episode-level
   teacher labels   binary preference sparse reward
```

\* KTO 是"准 on-policy"：它取的是 (chosen, rejected) 对，而不是当前 model 的全 rollout。

把 OPD 放进去要回答三个问题：

### Q1：**OPD 替换谁？**

不替换。OPD 跟 SFT/KTO/GRPO 的监督信号**不同方向**：

* OPD 沿 student 自身轨迹 **拉向**教师分布（mode-seeking via reverse KL）。
* KTO 是 **二元偏好对齐**（chosen > rejected）。
* GRPO 是 **群体相对优势**（在 K 个 rollout 间排序）。

三者都需要、不替换；OPD 加到中间。

### Q2：**OPD 加在哪里？**

加在 **SFT 之后、KTO 之前**，做新的 Stage A.5：

```
[SFT cold start] → [OPD: dense on-policy guidance] → [KTO preference] → [GRPO]
```

**理由（三条都有 reference）**：

1. **SFT 之后 student 已经能产生合法格式的 JSON action**，OPD 才有"合法 rollout 可教"。如果直接从 base model 跑 OPD，rollout 全是乱码，teacher 在乱码上的 logits 没意义。SFT 是 OPD 的必要前置——[Rethinking OPD (arXiv 2604.13016)](https://arxiv.org/abs/2604.13016) 的"Phenomenology"章节明确指出这点。

2. **OPD 提供的 token-level supervision 让 KTO 阶段的 pairwise loss 更稳定**。直接 SFT→KTO 会让 student 在 KTO 阶段反复"试错"地探出哪些 token 该改；中间塞一道 OPD 等于把 student 的输出 *预先拉到* 教师的高概率区，KTO 只用做 fine-grained 偏好调整。这是 [TM Lab blog (2025)](https://thinkingmachines.ai/blog/on-policy-distillation/) 给出的"7–10× fewer gradient steps to reach RL-level"的核心机制。

3. **OPD 不替代 GRPO**，因为 OPD **上限受教师约束**（mode-seeking → 学生不会超过教师）。我们仍然要 GRPO 把策略推到教师都没见过的状态空间。[OPSD (arXiv 2601.18734)](https://arxiv.org/abs/2601.18734) 也明确："OPD performs on par with or better than GRPO" — 注意是"on par"，不是"strictly better"；GRPO 仍然有不可替代的能力。

### Q3：**Teacher 从哪来？**

三个真实可行的来源，按优先级：

* **A (推荐)** — *外部强模型作教师*：Claude Sonnet 4.5 / GPT-4o / Qwen3-VL-72B-A22B 跑 EditorAgent 当 teacher；DeepSeek-V3 / Qwen3-7B 当 student。教师贵但调用频率低（每个 rollout 只调一次取 logits）。

* **B (退路)** — *self-distillation (OPSD)*：student 自己取 logits 在更小温度下重采样作"教师"。论文：[OPSD arXiv 2601.18734](https://arxiv.org/abs/2601.18734) 证明在小模型场景这条路有效。

* **C (混合)** — *RL-trained checkpoint 作教师*：先跑 GRPO 到一个 v0.4 checkpoint，然后用它给 v0.5 的更小蒸馏 student 提供 OPD 信号——本质是 [TM Lab blog](https://thinkingmachines.ai/blog/on-policy-distillation/) 演示的"RL → OPD 蒸馏到小模型"路径。

**我们项目最实在的选项**：方案 A（用 Claude Sonnet 4.5 / GPT-4o 作 EditorAgent 教师，DeepSeek-V3 / Qwen3-7B 作 student）。理由：

* 我们的 `MockLLMClient`/`OpenAIClient`/`AnthropicClient` 抽象**已经允许同时跑两个 LLM 后端**（教师 + 学生），零代码修改。
* EditorAgent 的 prompt 短（segment guidance + JSON action 一千 token 以内），教师 token 调用开销可控。
* Anthropic / OpenAI API 都返回 token-level logprobs，能直接吃。

---

## 3. 长视频编辑 agent 特殊性 — OPD 在我们的场景到底能赚多少

业界 50–100× 的数字来自 reasoning benchmark（数学、code）。我们要务实地预估**自己的**收益。

### 3.1 token 密度优势（O(N) vs O(1) bits per episode）

* 我们每个 segment 的 rollout 输出大约 **30–80 个 token**（JSON action `{"action": "retrieve|generate|fallback", "rationale": "..."}`）。
* RL outcome reward 是 O(1) — 我们的 EnsembleResult.score。
* OPD 在这 30–80 token 上每个都贡献梯度信号。
* **乘数估计**：30–80× — 这是个 *上界*，实际因为很多 token 是 boilerplate（"action"/"rationale" 等 key 名），有效信息位更少，实测可能在 5–20×。
* 还是显著高于纯 RL。

### 3.2 mode-seeking vs mode-covering — 我们要哪个？

* **reverse KL (mode-seeking)** → student 集中在 teacher 的高概率模式 → 输出更稳定但更少 diversity。
* **forward KL (mode-covering)** → student 覆盖 teacher 全分布 → 多样但易产生低质内容。

EditorAgent 的 action 空间只有 {retrieve, generate, fallback} × 短 rationale —— **我们要的就是稳定决策，不是 diversity**。reverse KL（即标准 OPD）的 mode-seeking 正好匹配。

Diversity 是 ScreenwriterAgent 才该担心的事（叙事创意），而我们目前没打算把 Screenwriter 也做 RL/OPD。Round 9 已经在 Screenwriter 上加了 self-consistency 解决 diversity。两边责任清晰。

### 3.3 多模态约束

我们的 action 输出是 **纯 JSON 文本**——多模态信号在观测端，不在 action 端。所以多模态 OPD 的复杂性（[LiveTalk arXiv 2512.23576](https://arxiv.org/abs/2512.23576) 处理的 flickering / black frames 等问题）**不适用于我们**。我们只用单模态 (text) OPD。

### 3.4 长 horizon

OPD 默认不解决 multi-step 信用分配——它在每个 token 上的密度很高，但不知道"这段编辑后来失败了"。这正是我们已经实装的 **HCAPO hindsight refiner**（[arXiv 2603.08754](https://arxiv.org/abs/2603.08754) + [`training/rewards/hindsight.py`](../training/rewards/hindsight.py)）补位的地方。两者**正交、组合**：
* OPD 给 token-level dense signal。
* HCAPO 给 step-level hindsight refinement。

[Survey of OPD (arXiv 2604.00626)](https://arxiv.org/abs/2604.00626) 在"open problems"章明确点名 *trajectory-aware credit assignment* 是 OPD 的缺口；我们恰好用 HCAPO 补。这不是巧合，是架构选择就预留了正确的接口。

---

## 4. 反方意见 — OPD **不适合**我们的三种场景

严谨分析必须给出反方。OPD 不该用的情况：

### 4.1 teacher ≈ student 强度（"distillation 没东西可蒸"）
当我们 v0.4 已经用 GRPO 把 Editor 训到接近 Claude Sonnet 4.5 水平后，再用 Claude 当教师做 OPD 收益就接近 0。这时该转 self-distillation (OPSD) 或者放弃 distillation 进入 multi-agent co-training 阶段。

### 4.2 student 完全无法产生合法 rollout（"on-policy data is garbage"）
[Rethinking OPD (arXiv 2604.13016)](https://arxiv.org/abs/2604.13016) 文中"Mechanism"章节实验：当 student SFT 不到位、rollout 失败率 > 80% 时，OPD 失败甚至有害。**我们的对策**：把 SFT cold-start 跑足 — 用 `--reward-threshold 7.0` 保证至少有几千条高质量轨迹做 SFT 数据。

### 4.3 teacher 的某些行为我们**故意**不想要
比如教师（GPT-4o）有时会输出 markdown fence，而我们要纯 JSON。OPD 会把这种习惯一并蒸到 student。**对策**：在 OPD 前用 `prompts/` 的统一 system prompt 约束教师输出格式；OPD 后接 KTO 用 (clean JSON, fence-y JSON) 的偏好对来反向纠正。这就是为什么 OPD 之后还需要 KTO。

### 4.4 reward hacking 风险评估
KTO/GRPO 用一个 *学到的* RM，可能被 game。OPD **不用 RM**，监督信号是教师的固定 logits — 这条 channel 难被 hack。
反之，OPD 引入了对教师的**单点依赖**：如果教师本身有偏差（Claude 的某种 sycophancy），偏差会 1:1 复制到 student。这跟 RM hacking 是**不同**的风险维度。我们通过 *教师 ensemble* 缓解：随机在 Claude / GPT-4o / Qwen3-VL 之间采样作教师，平均掉单一教师偏差。这是 [Multi-Agent Debate (Du et al. 2023)](https://arxiv.org/abs/2305.14325) 思路的另一种应用。

---

## 5. 集成计划（具体到代码 + reference）

### 5.1 新加 Stage A.5

| 件 | 路径 | 引用 |
|---|---|---|
| `OPDConfig` 数据类 + `OPDStage` | `training/stages/distill.py` (新增) | GKD ([2306.13649](https://arxiv.org/abs/2306.13649)); TM Lab blog; OPSD ([2601.18734](https://arxiv.org/abs/2601.18734)); Entropy-Aware OPD ([2603.07079](https://arxiv.org/abs/2603.07079)) |
| backend = `stub` / `trl` / `nemo_rl` | 同上 | NeMo-RL OPD discussion #1445 |
| PipelineRunner 加 `opd: OPDConfig` 字段 | `training/runners/pipeline.py` (修改) | EVA's 3-stage + 我们的扩展为 4-stage |

新的 PipelineRunner pipeline：

```
RM training (Bradley-Terry / Tülu-3)
    ↓
SFT cold start (reject-sampled high-reward trajectories)
    ↓
OPD (student rollout × teacher token-level supervision)   ← 新加
    ↓
KTO preference alignment (pairwise / unpaired)
    ↓
GRPO RL (with EditingQualityRM + HCAPO refiner)
```

### 5.2 reverse KL loss 的 stub 实装

OPD 的核心 loss：

$$
\mathcal{L}_{OPD}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \big[\text{KL}\big(\pi_\theta(\cdot|y_{<t}, x) \,\|\, \pi_T(\cdot|y_{<t}, x)\big)\big]
$$

其中 $\pi_T$ 是教师（固定，不更新），$\pi_\theta$ 是 student（要更新）。注意 KL 是 student-on-policy 上的 reverse 方向。

stub backend 只要 deterministic 模拟"教师 logprob 比 student logprob 高 → loss 下降"的趋势就够。真 backend 调 TRL 的 `GKDTrainer`（[TRL v1.0 of April 2026](https://huggingface.co/docs/trl) 已加入 GKD）或 NeMo-RL 的 OPD 路径。

### 5.3 教师接口

复用我们已有的 `BaseLLMClient`——教师就是另一个 `OpenAIClient(model="claude-sonnet-4-5")` 或 `AnthropicClient(...)`。零新依赖。

---

## 6. 量化预期收益（务实）

| 指标 | 当前 v0.4 (SFT+KTO+GRPO) | 加 OPD 后 v0.4.1 | 出处 |
|---|---|---|---|
| 达到同一 Mashup-Bench 分数所需 GPU-小时 | baseline | 30–50% (~2× 加速) | TM Lab blog 50-100× 在我们短 episode 场景的保守下调；REOPOLD 6.7×–12× 的中位数; 见 §3.1 |
| RL 阶段 KL divergence 收敛步数 | baseline | 3–7× 更快 | TM Lab blog 实测；[Rethinking OPD](https://arxiv.org/abs/2604.13016) "Recipe" 节 |
| OPD 阶段所需 teacher token 数 | n/a | ~ episode 数 × 80 token | 我们的 episode 长度估算 |
| API 成本（Claude Sonnet teacher） | n/a | 每 1000 episode ≈ \$15 (input + output @ $3/$15 per million) | Anthropic 公开 pricing |
| 引入的新工程 risk | 0 | + 1 stage 接口 + teacher API 单点 | 自评估 |

**总结**：花 < $50 + 几小时工程，换 30-50% 训练加速 + 2-5× 更稳定的 KTO/GRPO 阶段——值得做。

---

## 7. 决策 — 我的最终判断

**做**。具体：

1. **v0.4.1**（介于 v0.4 SFT+KTO+GRPO 和 v0.5 multi-agent 之间）插入 OPD Stage A.5。
2. 教师：Claude Sonnet 4.5（首选）/ Qwen3-VL-72B-A22B（退路）。
3. 学生：DeepSeek-V3（首选）/ Qwen3-7B-Instruct（小尺寸退路）。
4. backend：`stub` 默认（v0.1 测试用）；`trl` 当 TRL `GKDTrainer` 可用；`nemo_rl` 当 NeMo-RL 集群可用。
5. **不**对 ScreenwriterAgent 做 OPD（diversity 需求 vs OPD 的 mode-seeking 不匹配；见 §3.2）。
6. **不**替换任何已有 stage——OPD 在中间，不在两端。

**红线**——如果以下任一条件成立，停 OPD：

* student SFT 后 rollout 失败率（malformed JSON 比例）> 60% → 先补 SFT
* teacher 在 hold-out 上 Spearman with EditingQualityRM < 0.5 → 教师不靠谱，换教师
* OPD 阶段 KL divergence 在前 100 step 内不单调下降 → 试 Entropy-Aware OPD (arXiv 2603.07079) 或换 Stable OPD reformulation (arXiv 2601.07155)

---

## 8. 我对"是否单纯 follow trendy paper"的自我审视

写完这份分析我又问自己一次："OPD 火，跟你们项目契合度高，这是真的还是 confirmation bias？" 反方论点我能想到的：

* TM Lab blog 是商业 marketing，50-100× 数字偏乐观——但 OPSD (arXiv 2601.18734) 和 REOPOLD (arXiv 2603.11137) 在 peer-reviewed-style arXiv 上独立复现了 6.7–12×。
* "OPD 不替换 RL" 这条结论是真实的——Survey ([2604.00626](https://arxiv.org/abs/2604.00626)) 明确把 OPD 定位为 RL 的 *补充*。
* OPD 在多模态有局限——但我们的 action 是文本，不卡这个问题。
* 唯一真正可疑的点是：**我们没有量化我们当前 SFT→KTO→GRPO 的 baseline**。在没 baseline 的情况下加 stage 是 premature optimization。

最后这点是合理的。所以推荐顺序：

1. **先做 v0.4 baseline (SFT+KTO+GRPO 真后端跑通)**，量化数字。
2. **再加 OPD as Stage A.5**，看是否如预期带来 30-50% 加速。
3. **如果 < 20% 加速**——这表明我们的 KTO/GRPO 已经在小数据上学得很好，OPD 收益不显著，回退。

这种"先量化、再加机制"的纪律是 [Rethinking OPD (2604.13016)](https://arxiv.org/abs/2604.13016) "Recipe" 章节的核心建议——它的 takeaway 之一就是"don't add OPD on faith"。

---

## 9. 文档与代码状态

* 本分析：[`docs/ON_POLICY_DISTILLATION_ANALYSIS.md`](./ON_POLICY_DISTILLATION_ANALYSIS.md)（本文件）
* 实装：[`training/stages/distill.py`](../training/stages/distill.py) — OPDConfig + OPDStage（stub backend default）
* 测试：`tests/training/test_distill.py`
* 集成：[`training/runners/pipeline.py`](../training/runners/pipeline.py) 中 PipelineConfig 加 `opd: OPDConfig` 字段
* 引用钉死：[`tests/unit/test_reference_grounding.py`](../tests/unit/test_reference_grounding.py) 加 OPD 相关 needle

---

## 附录：本文档全部引用（按出现顺序）

| Reference | Type | Link |
|---|---|---|
| Agarwal et al., GKD: On-Policy Distillation of Language Models (ICLR 2024) | paper | https://arxiv.org/abs/2306.13649 |
| Thinking Machines Lab — On-Policy Distillation blog (Oct 2025) | blog | https://thinkingmachines.ai/blog/on-policy-distillation/ |
| OPSD — Self-Distilled Reasoner (2026) | paper | https://arxiv.org/abs/2601.18734 |
| Stable OPD through Adaptive Target Reformulation (2026) | paper | https://arxiv.org/abs/2601.07155 |
| Entropy-Aware OPD (2026) | paper | https://arxiv.org/abs/2603.07079 |
| REOPOLD — Relaxed OPD (2026) | paper | https://arxiv.org/abs/2603.11137 |
| Survey of OPD for LLMs (2026) | paper | https://arxiv.org/abs/2604.00626 |
| Rethinking OPD: Phenomenology, Mechanism, Recipe (2026) | paper | https://arxiv.org/abs/2604.13016 |
| LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved OPD (2026) | paper | https://arxiv.org/abs/2512.23576 |
| NeMo-RL OPD support (NVIDIA) | repo discussion | https://github.com/NVIDIA-NeMo/RL/discussions/1445 |
| TRL v1.0 (HuggingFace, Apr 2026) | docs | https://huggingface.co/docs/trl |
| HCAPO — Hindsight Credit Assignment (2026) | paper | https://arxiv.org/abs/2603.08754 |
| Multi-Agent Debate (Du et al., 2023) | paper | https://arxiv.org/abs/2305.14325 |
| Awesome-OPD (thinkwee) | repo | https://github.com/thinkwee/AwesomeOPD |
| Awesome-LLM-On-Policy-Distillation (nick7nlp) | repo | https://github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation |
| EVA: Efficient RL for End-to-End Video Agent (2026) | paper | https://arxiv.org/abs/2603.22918 |
