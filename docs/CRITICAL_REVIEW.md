# Critical Review of Rounds 1–15

> "把之前所有做的步骤，继续发现错误" — 用户指示。
>
> 这份文档不是 changelog。是诚实清单：**14 轮里我做了哪些看起来工作、实际上只是在自我闭环的事**。它在主分支 (`main`) 上，作为
> [`docs/CSA_FRAMEWORK.md`](./CSA_FRAMEWORK.md) 的动机依据存在。

---

## 1. 总判断

`v0.2 (main)` + `rl-integration` 分支表面上交付了一个"多 agent、self-loop 闭环、RL 就绪"的长视频剪辑系统。**真实情况是：所有 self-loop 信号都来自我硬编码的几个数字，整条管线没有任何一处真的判断过 "这次剪辑好不好"。** 测试全绿 ≠ 系统有意义。

这不是 bug，是 *表征性错误* — 代码看起来在做某件事，实际上它没在做。下面按 round 列具体例子。

---

## 2. 按 round 的诚实清单

### Round 3 — RetrievalTool beam search "真实滑窗"
- **声明**：实现了 DIRECT §4.3 的 dynamic sliding-window trimming。
- **实际**：`_slide_trim` 在我加完所谓"motion + framing-aware 排序"后，仍然取的是最简版本，三行就完了。motion-aware 部分是用 `feats.avg_flow_magnitude`（一个标量）做加权——这不是 motion-aware，是 magnitude-aware。论文里说的是 flow field 方向 + 大小的联合，这里没有方向，因为 mock flow 是随机噪声。
- **后果**：beam search 的"排序质量"不可能比随机更好。

### Round 4 — GenerationTool 的 metric_scores
- **声明**：m1..m6 metric_scores 反映 generation 是否有 anchor 输入。
- **实际**：固定常数。`has_first_frame` 决定 m2 是 0.75 还是 0.50，`has_flow` 决定 m3 是 0.70 还是 0.45。两条线性公式。
- **B 阶段 R16-1 之前**：R→G 和 G→G 边界在 metric_scores 上 *完全相同*——因为 `previous_source` 没传进来。这意味着 hybrid 这条架构断言 14 轮里从未被任何信号触碰过。
- **B 阶段 honesty fix (D-021)**：我在第 16 轮才加上 `previous_source` 信号。**这本应是 round 1 的事**。

### Round 9 — EnsembleRewardModel 的"3 个 judge"
- **声明**：3 个独立的 mock judge，权重不同，模拟 multi-judge ensemble + disagreement。
- **实际**：3 个 judge 全部消费 *同一份 metric_scores dict*，只是用 3 个不同的线性权重做加权和。这不是 ensemble — 这是同一信号的 3 个仿射变换。它们的 disagreement 是用户输入决定的（取决于 metric_scores 的分布），不是 judge 自身视角差异。
- **真正的 ensemble** 应该让每个 judge 看 *不同的信号源*（一个看视频帧、一个看音频、一个看 transcript）。我的实现做不到这点，因为我没有真的去读视频。
- **后果**：`EnsembleResult.disagreement` 在 v0.4 RL 训练时会被当 active-learning 信号用 — 但它的方差完全来自我设的权重表，不是世界。

### Round 9 — LessonBook 跨 run 反思
- **声明**：CriticAgent 写 Lesson；下次同 user_prompt 时 Screenwriter 注入。
- **实际**：CriticAgent 的 5 类 trigger 都是规则扫 trajectory.jsonl 字段。但 trajectory.jsonl 里的 `reward` 字段是 MockRewardModel 的输出 — 而 MockRewardModel 又是 metric_scores 的线性组合 — 而 metric_scores 是我在 generation_tool.py / retrieval_tool.py 里硬编码的。
- **整条因果链是**：我写常数 → mock judge 做线性组合 → critic 用阈值扫 → lesson 写进 book → 下次注入 prompt。所有"反思"的真实信源是我设的几个常数。
- **从未测试过**：注入 lesson 后下次 run 的结果是否真的不一样。

### Round 9 — PreferenceLogger
- **声明**：捕获 (winner, losers) 三元组，HuggingFace TRL 兼容，供 v0.3 DPO 训练。
- **实际**：winner 是 MockRewardModel score 最高的候选。所以训出来的 RM 是在拟合"MockRewardModel 喜欢什么"，不是"人类喜欢什么"。
- **更糟糕**：MockRewardModel 的权重几乎可以用 `argmax` 直接表达 — 一个 7B LoRA 拟合不到 1 小时就会完美复现它。我会得到一个"准确率 99%"的 RM，然后宣布 v0.3 成功，但这个 RM 在真实人类标注上的表现可能是随机的。

### Round 9 — Self-Consistency Screenwriter
- **声明**：K 次采样取众数提升稳健性。
- **实际**：MockLLMClient 每次调用走的是 `_stub_for(messages)` 里 `if "screenwriter" in joined` 分支，**输出确定**。K 次采样得到 K 个一样的输出。`_aggregate_plans` 退化成恒等映射。
- **测试 `test_self_consistency_k_gt_1_still_emits_one_plan` 仍过**，因为它只 assert "rationale 字段提到 self-consistency 字样"——而这字样是我在 `_aggregate_plans` 里写死的。
- **后果**：Self-Consistency 在 mock 模式下是 no-op，但测试看起来工作。真接 LLM 之前，这个机制就是装饰。

### Round 11 — 2024-2026 引用刷新
- **声明**：每个核心模块 docstring 都点名最新工作。
- **实际**：大部分 docstring 注释是 *叙述性的* — 说"我们这里的 reverse-KL 思路来自 XX 2024"——但代码本身没实现 reverse-KL。Docstring 把它说成"这就是那个工作的实现"是夸大的。
- **`test_reference_grounding` 钉死引用串** 防止后人 *删* 引用，但它没法防止我 *滥用* 引用。

### Round 13 — `training/` 子树
- **声明**：mock-first 的 SFT → KTO → GRPO 三段式可用。
- **实际**：所有 `stages/*.py` 的 stub backend 都走 `_stub.py::stub_train`，它的"loss"是 `base / (2 ** i)`——一个 hard-coded 指数衰减。**没有任何梯度更新**。`metrics.json` 里写的 final_loss 是这个常数表达式的结果。
- 这是"测试架构骨架"的合理 mock，但我在 SYSTEM_GUIDE + AGENTIC_RL_PROPOSAL 里把它包装成"v0.4 训练就绪"——这是误导。

### Round 14 — OPDStage 的 KL 曲线
- **声明**：stub backend 模拟 OPD 的 KL-to-teacher 单调下降。
- **实际**：`kl = 0.3 + 1.2 * exp(-0.6 * step)`。**完全硬编码**。我在 docs/ON_POLICY_DISTILLATION_ANALYSIS.md 里把这写成"模拟真实 OPD 行为"，**但它什么都没模拟，它就是一个 4 数字的常数序列**。
- 测试 `test_kl_to_teacher_decreases_monotonically` 是 tautology——它在验证我写的 exp 是单调下降的。

### Round 15 / B 阶段 — `scripts/measure_baseline.py`
- 这个我相信是干净的。它真的会跑 27 次 `run_pipeline`、真的会聚合，verdict 是真的从数据计算的。
- **但它能算出的"verdict"** 取决于 R→G 边界是否在 metric_scores 上和 R→R 边界拉得开——而这又取决于我刚加的 `previous_source` 系数（D-021）。所以即使跑出 ✅ 也只是"我设的系数让 hybrid 看起来好"。
- 真正有意义的 baseline 必须在真 LLM + 真 video-gen 下跑。**目前的 baseline 工具是一个 *测得动* 的脚手架，不是 *测得出真相* 的实验。**

---

## 3. 真实状态总结

我们有：

* ✅ 完整的代码骨架，所有 pytest 全绿
* ✅ 一致的 dataclass 类型系统、配置加载、trajectory 日志
* ✅ ffmpeg 真的会拼出 mp4（这是唯一真做了的事）
* ✅ mock 后端在不爆炸的前提下能跑完整 pipeline
* ✅ 一个清晰的 baseline 实验框架（R16.B 阶段写的）

我们没有：

* ❌ 任何机制真的判断过 "这次剪辑好不好"
* ❌ 任何 self-loop 信号不是我硬编码常数的下游产物
* ❌ 任何 ensemble 真的有视角差异
* ❌ 任何 RL 训练发生过梯度更新
* ❌ 任何 OPD 蒸馏发生过
* ❌ 任何 Reviewer 看过一帧视频

---

## 4. 反复出现的元错误

这些错误不是技术 bug，是**思维模式**的错误：

### 元错误 #1 · "测试绿 = 工作"
单元测试验证的是函数返回值符合我写的 assertion。当 assertion 和实现都是我写的，绿测试只证明我前后一致 —— 它**不证明系统做了任何外部世界关心的事**。

### 元错误 #2 · "mock 越来越完整 = 更接近真"
我加了 EnsembleRewardModel、PreferenceLogger、OPD、HCAPO refiner——每一层都是在前一层 mock 输出上再造一层 mock。这是 mock 套娃，不是逐步接真。

### 元错误 #3 · "引用最新 paper = 框架先进"
我在 Round 10-11 把所有 docstring 升级到 Qwen3-VL / HunyuanVideo / Tülu-3。代码里没有任何一行真的用了这些。引用最新 paper 是文档工作，不是工程工作。

### 元错误 #4 · "差异化 = 集成更多 paper"
Round 9 的 5 个机制、Round 13 的 7 个 RL 模块、Round 14 的 OPD —— 每次我都说"这跟其他工作的差异化在于我们把 X+Y+Z 都做了"。"做了" = 加了 stub。整合更多 paper 不是差异化，是 buffet。

---

## 5. 这份审计的用处

这份审计是 `docs/CSA_FRAMEWORK.md` 的依据。CSA 框架的合法性来自于：

> "你已经 14 轮证明：在 segment-level metric_scores + multi-agent + 模拟 RL 的设计空间里，没有差异化可言。所有现有 paper 都在这个空间里，我们叠加它们就是模仿。**差异化必须来自换原语和换尺度。**"

CSA 的换法：把原语从 *segment* 换成 *cut*；把判别尺度从 *segment-level (m1..m6)* 加上 *arc-level (whole-script)*；把 retrieve/generate 的选择从 *EditorAgent 的 top-down 决策* 改成 *intent-driven 的 lookup 副作用*。

详细见 [`docs/CSA_FRAMEWORK.md`](./CSA_FRAMEWORK.md)。

---

## 附录 · 哪些 round 是真做了对的事

为了不变成自我鞭笞，列出真有意义的几件：

| Round | 做对了什么 |
|---|---|
| 4 (基础设施) | `Config` dataclass 树、`TrajectoryLogger` JSONL、`load_prompt` 三件确实是 reusable 的工程产物 |
| 5 (memory/) | `MemoryStore` 用 SQLite + npz + 可选 FAISS 的三层存储是干净的，没掺水 |
| 7 (orchestration/) | LangGraph fallback 的 state machine 是真在切换状态、真在 conditional edge 上转，不依赖 mock 信号 |
| 10 (assembly) | ffmpeg 真的会拼真 mp4。这是整个项目里唯一一处"真做了某件世界关心的事"的地方 |
| 15-B (measure_baseline) | 实验脚手架本身写得清晰，等真 backend 上来后是可用的 |

剩下的 95% 是 mock 框架 + 文档。
