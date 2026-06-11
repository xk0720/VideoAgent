# Maestro 创新点重构方案（2026-06-11，待用户确认）

> 输入：用户三点思路 —— (1) 以 **skill** 为形式实现"素材 + 用户指令 → agent 自动化视频生成"的核心故事；
> (2) 增强生成视频的 **physics** 能力（review 也是一种 skill）；(3) **memory** 与长视频生成结合的创新（也是一种 skill）；
> 整体做成 **self-improve** 形式。
>
> 依据：四份新调研（`survey_self_improve_2026_06.md`、`survey_memory_2026_06.md`、
> `survey_physics_2026_06.md`、`survey_skills_audio_2026_06.md`，共 ~80 篇 2024–2026 文献）
> + 仓库现状（父项目 `docs/CRITICAL_REVIEW.md`、Maestro v0.3 C1–C8）。

---

## 0. 现状诊断："链路跑不通"的真实含义

两套代码都测试全绿（父项目 109、Maestro 113），所以问题不是代码报错，而是：

1. **信号链路没接到真实世界**（父项目 CRITICAL_REVIEW 的结论）：所有 self-loop 信号
   （metric_scores → reward → lesson → preference）都是硬编码常数的下游。测试绿 ≠ 系统判断过"这次视频好不好"。
   Maestro 的 mock 也有同样风险——C2/C5 自改进、C7 技能、C8 记忆都没在真实像素上闭环过。
2. **两套框架互不相通**：父项目有素材链路（检索+生成混合、ffmpeg 真拼接、SQLite+FAISS 记忆库），
   Maestro 有创新链路（物理 oracle、HSI、技能库、多层记忆），但"素材 + 指令 → 长视频"的完整故事
   在任何一套里都不完整。
3. **创新点表述分散**：C1–C8 各自成立，但缺少一个统一的"核心故事原语"。用户的直觉
   （一切创新以 skill 的形式组织）正好是文献里完全空白的统一抽象（见 §2）。

**因此"跑通链路"的定义应当是**：在服务器上，素材 + 指令进 → 至少一条**真实信号**
（真 VLM critic / 真 tracker / 真生成模型）穿过 plan → generate → review → improve → memory 全链 → 真 mp4 出。

---

## 1. 文献给出的总空白图（四份调研的交集）

| # | 空白（截至 2026-06 无人占据） | 来源调研 |
|---|---|---|
| G1 | **skill 习得线（Voyager/SkillWeaver/AWM/AutoSkill）与视频 agent 线（UniVA/CutClaw/Crayotter/MovieAgent）零交集** —— 所有视频 agent 都是静态手注册工具 + 角色 prompt，程序性知识困在 prompt 里每次重新推导 | skills_audio |
| G2 | skill 抽象不适配视频域：现有 skill 验证 = 代码执行成功（廉价确定 oracle）；视频工具调用昂贵、随机、感知评判 —— **无人做 VLM 评审的技能准入（"skill CI"）、昂贵随机调用下的技能打磨、以素材为参数的技能** | skills_audio |
| G3 | 物理验证：**"测量得到的、可解释的、按物体定位的残差，同时驱动 best-of-N 选择与定向重生成，training-free"** 这一交集无人占据（PSIVG/PhyRPR 开环注入；WMReward 闭环但不可解释；PhyT2V 闭环但文本瓶颈；Morpheus/PISA 测量但只用于评测） | physics |
| G4 | **无人量化 tracker 在生成视频上的可靠性**（CoTracker3 等都在真实视频上训练，对变形/闪烁内容会输出"看起来合理的轨迹"）—— verifier-of-the-verifier 是独立可发表的缺口 | physics |
| G5 | 生成侧记忆：**无感知↔生成闭环读写**（记忆存的是"打算生成什么"，从不是"输出里验证到什么"）；**无身份/状态分离**（EntityMem 冻结、VideoMemory 自由漂移，无人做 canonical identity ⊕ evolving state + 类型化转移日志） | memory |
| G6 | **所有生成记忆随项目死亡**：跨视频/跨会话的角色库、风格库、偏好记忆从未接到视频生成器上 | memory |
| G7 | 自改进：**视频生成的"验证过的失败教训库"几乎无人认领**（VISTA 每次会话清零；MemoGen 只做图像；TTOM 存参数不存语义） | self_improve |
| G8 | **跨阶段归因（blame assignment）不存在**：MAViS 等逐阶段局部审查，坏成片可能源于剧本，但只有视频阶段被重试 | self_improve |
| G9 | **最廉价充分保真度级联验证不存在**：script → storyboard → keyframe → latent → clip 各级验证只被孤立利用过，无端到端自适应分配 | self_improve |
| G10 | **一致性架构与 agentic 闭环未结合**：StoryMem 等记忆库不加检查地吞下生成内容（错误跨镜头复利传播）—— critic 门控的记忆写入无人做 | self_improve + memory |
| G11 | 音频：**无 agent 闭环**"生成音频 → 感知混流结果 → 评审 AV 语义/时序契合 → 重新下 brief"；跨镜头声学连续性（所有 V2A ≤8-10s，接缝处氛围/音乐跳变）无人管理 | skills_audio |

---

## 2. 统一框架提案："一切可改进之物皆是 skill"

把用户的直觉形式化为框架核心原语：

**Skill = 类型化、经验证准入、版本化、可进化的程序单元**，统一三类：
- **创作技能**（generation skills）：从自身轨迹蒸馏的工作流（"卡点蒙太奇(素材,音轨,情绪)"、"对话场景正反打"），对应 G1/G2；
- **评审技能**(review skills)：物理验证器（C6 oracle 各档位）、AV 一致性 critic、实体一致性 critic —— **review 本身是技能**，按 prompt 路由（G3 的 S3 路由器）；
- **记忆技能**（memory skills，借 MemSkill 框架）：写入门控、状态转移、淘汰策略。

**Self-improve = skill 生命周期**：蒸馏（从轨迹）→ **验证准入**（VLM 评审 + 回归测试，"skill CI"，对应 G2）→ 检索 → 执行 → 评估（EMA 性能）→ 进化/淘汰。
agent 唯一可学习的基质就是技能库——可审计、可版本化、可回滚（避开 Gödel-Agent 式任意自改写的不安全性）。

这一步使三个创新点共享同一生命周期与同一评审证据链，而不是三个并列模块。
Maestro v0.3 的 C7（PhysicsTyped SkillLibrary）是雏形，需推广为上述三类。

---

## 3. 三个创新点的具体落点

### 3.1 创新点一：素材+指令驱动的技能化创作核心（占 G1/G2/G8/G9）
- 任务定义（合并父项目前提）：输入 = 1+ 源素材（视频/图片/角色参考）+ 自然语言指令 + 可选音轨；
  输出 = 检索+生成混合的长视频（含音频）。
- **Skill 习得**：从自身外化轨迹（trajectory.jsonl 已有）蒸馏带参技能；**VLM 评审准入**替代
  Voyager 的执行成功 oracle；昂贵调用下用"少样本验证 + 范例缓存"替代 SkillWeaver 的反复练习。
- **跨阶段归因 + 级联验证**：把 C5 HSI 升级为全链归因（成片失败时判定责任在剧本/分镜/关键帧/片段哪一级，
  回到最便宜的责任层重做），配合 script→storyboard→keyframe→latent→clip 逐级验证（关键帧便宜、成片昂贵的不对称性）。

### 3.2 创新点二：物理 = 验证 oracle 技能族（占 G3/G4）
保持 sketch-as-verifier 定位（已是正确站位），按 S1–S5 强化：
- **S2 优先**（tracker 可靠性门控 + 跨 tracker 一致性，副产品"tracker 分歧本身是物理不合理线索"可独立成文，填 G4）；
- **S4 优先**（残差定向重生成：只重生成违规物体的时空管道，把 best-of-N 变成类梯度修正，直接回应"昂贵拒绝采样"质疑）；
- S1（参数后验 + 守恒残差）、S3（可验证性路由器：刚体→仿真，参数运动→方程发现，流体/生物→V-JEPA 2 reward，语义违规→VLM）、S5（级联早退 + 反循环评测协议）随后。
- 物理评审器作为**带类型签名的 review skill** 注册进技能库，由路由器按场景选用。

### 3.3 创新点三：验证门控的双寄存器实体记忆（占 G5/G6/G10）
- **双寄存器**：每个实体 = 不可变身份寄存器（验证过的参考 crop + 频域 ID 特征）⊕ 可变状态寄存器
  （服装/损伤/情绪/位置/光照），状态变化只能通过导演 agent 签发的**类型化转移日志**进入 —— 连续性可审计。
- **验证门控写入**："只提交渲染中确认的内容"——VLM critic 对照记忆检查每个生成镜头，
  确认才写入状态更新；不符则触发重生成或显式记忆修正条目。这同时是 G10 的解
  （critic 门控记忆写入 = 记忆创新与自改进创新的天然汇合点）。
- **跨项目层**（G6）：用户级角色库/风格库/偏好记忆，从隐式信号（保留 vs 重生成的 take）更新。
- 评测：EntityBench (2605.15199) 长复现间隔切分 + VideoPhy-2 + 人评。

### 3.4 配套：音频闭环（用户的音乐需求，占 G11）
- 后置多模型路由（联合 AV 模型对 agent 不可行：仅给自生成视频配音且 ≤8s）：
  foley → MMAudio/ThinkSound（ThinkSound 的指令编辑级 = 评审闭环的执行器）；
  音乐 → agent 写语义 brief + V2M-Zero 管时序，或 CutClaw 式检索+卡点；混音 → ducking/响度确定性工具。
- **AV 一致性评审环**（无人做）：混流 → critic 评语义契合 + AV-Align/JavisBench 时序 → 结构化批评 → 重 brief。
- **跨镜头声学连续性**：氛围床、乐句跨切点规划。
- 注意 AV-Align 对音乐无效（节拍≠运动），音乐用 beat-alignment 指标。

---

## 4. 待确认决策清单（用户逐项确认后开工）

| # | 决策点 | 建议 |
|---|---|---|
| D1 | **代码底座**：以 Maestro 为底座合并（移植父项目的 ffmpeg 真拼接、SQLite+FAISS 记忆存储、场景检测、CSA arc-judge 思想），父项目转为遗留参考？还是继续双轨？ | 合并到 Maestro（父项目背着 CRITICAL_REVIEW 的 mock 套娃债务） |
| D2 | **任务定义**：确认核心故事 = "素材+指令 → 检索/生成混合长视频（含音频）"（即吸收父项目前提），而非 Maestro 现在的纯文生视频？ | 确认混合 |
| D3 | **统一 skill 抽象**（§2）：创作/评审/记忆三类技能 + 蒸馏→准入→进化生命周期，作为框架第一公民？ | 采纳——这是与所有现有工作区分的"换原语" |
| D4 | **物理优先级**：S2（tracker 门控）+ S4（残差定向重生成）先做，S1/S3/S5 排后？ | 是 |
| D5 | **记忆方案**：双寄存器实体记忆 + 验证门控写入为主线，跨项目用户记忆为系统级差异化，RL 学习保留策略（Angle 3）不做？ | 是（Angle 3 计算成本过高） |
| D6 | **音频范围**：v 下一版就加 sound-director 规划 + AV 评审环 + 模型路由？还是先占位（接口+mock），等视觉链路真实化之后？ | 先占位，视觉链路真实信号优先 |
| D7 | **"跑通"验收标准**：服务器上最小真实链 = 真 LLM 规划 + 真生成 backbone（OmniWeaving/Wan 任一）+ 真 VLM critic + 真 CoTracker 物理验证 + 真 ffmpeg 出片，端到端一条指令出一个 ≥3 镜头视频？ | 以此为第一里程碑，**先于**任何新框架代码 |
| D8 | **评测组合**：EntityBench、VideoPhy-2、Physics-IQ 协议、UniVA-Bench、AV-Align/JavisBench + 小规模人评？ | 是，并加反循环协议（独立评测轴 + 反作弊正则） |

---

## 5. 风险与诚实声明（提前防 reviewer / 防自欺）

- **元错误防线**（来自父项目 CRITICAL_REVIEW）：任何新机制必须先指明它消费的**真实信号源**；
  禁止在 mock 输出上再叠 mock。每个创新点配一个"信号溯源"测试。
- 物理线的收益上界 = 基础生成器的样本多样性（N 个里没有一个物理正确时，选择无能为力）——
  S4 定向重生成是答案，必须显式消融。
- 自评闭环的循环性：选择用自家 oracle、评测必须用独立轴（VideoPhy-2 AutoEval、人评），
  并在 oracle 里加反作弊正则（慢动作、静态场景会平凡地"守恒"）。
- 技能库的坏习惯固化风险（AutoSkill 的教训）：准入测试 + 回归测试 + EMA 淘汰是硬要求。
