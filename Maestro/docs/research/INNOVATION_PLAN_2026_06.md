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

**Training-free 硬约束**（用户确认 2026-06-11）：本版本全程不做任何训练/微调。
"蒸馏"特指 **LLM 把成功轨迹总结为结构化技能条目**（参数化的工具调用序列 + 触发条件
+ 准入测试，AWM/MemoGen 式，零梯度）；"准入"是推理期 VLM 评审；"进化"是条目改写
与 EMA 淘汰。所有需要梯度的备选（RL 学记忆保留策略、DPO 物理后训练、V2M-Zero 微调）
一律不实现，只能作为外部预训练模型被调用。

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

### 3.2 创新点二：参考自由的"像素物理验证器"（占 G3/G4）——已实现 v0.4

**Sketch 线已彻底废弃**（用户判定 + 文献支持：草图既不能控制冻结的生成模型，
"对比单条仿真轨迹"又预设了单图不可知的质量/摩擦/尺度参数）。新定位把问题改成
**参数自由**的提法：

> "观察到的运动是否存在**任何**物理一致的解释？"

实现（`maestro/physics/`，全部 training-free，125 tests passing）：
1. `annotate.py` — 规划期标注：哪些实体会动、什么**运动类别**（ballistic/rigid/fluid/
   agentive/static）、盯哪些失效模式。只是验证种子，不含任何轨迹/控制信号。
2. `router.py`（S3）— **可验证性路由器**：每个实体分配能检查它的最强档位
   （measurement / world_model / vlm / none），coverage 显式上报——部分验证绝不冒充全量验证。
3. `tracks.py` + `track_extractor_backends.py` — CoTracker/TAPIR 从生成像素恢复观测轨迹。
4. `reliability.py`（S2）— **先认证 tracker 再相信 verdict**：前后向一致性、跨 tracker
   分歧、抖动检测；"tracker 分歧本身就是不合理线索"（G4，可独立成文）。
5. `laws.py` — 核心：对观测轨迹拟合被动运动定律族（static / 匀速 / 匀加速，重力向量
   自由拟合故无需尺度标定），**最佳拟合残差 = 违例度**；再叠加离散异常检测
   （teleport→物体恒存、mid-air reversal→重力惯性、energy_gain→守恒、jerk spike→碰撞），
   全部按实体+帧段定位。
6. `verifier.py` + `critics/physics_consistency.py` — verdict 驱动 best-of-N 选择 +
   HSI 定向修复（S4 的第一形态：违例的帧段+实体+模式直接生成修复指令），
   p2_law_consistency 独立计分（source="law_verifier" 与 VLM 评审的 p1 分离）。

**站位**（survey_physics §SYNTHESIS）：PSIVG/PhyRPR 开环注入不验证；WMReward 闭环但
不可解释；PhyT2V 闭环但文本瓶颈；Morpheus/PISA 测量但只评测——
"**测量得到、可解释、按实体定位、驱动选择与定向重生成、training-free**"的交集仍然无人占据，
且新定位不再背负"你的仿真器才是错的"攻击。
后续：S1（守恒残差强化——能量/动量比值检验，无需绝对质量）、S5（级联早退 + 反循环评测协议）。
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

## 4. 决策清单（2026-06-11 用户裁定后的状态）

用户三点裁定：(1) sketch 线废弃 → 物理重新思考（已落地为 §3.2 参考自由验证器）；
(2) 本版本严格 training-free，不设计任何训练/微调（已写入 §2 硬约束）；
(3) 链路不通参考根目录 `univa/`（工程图谱见 `univa_engineering_map.md`，
WaveSpeed API 后端已实现进 `models/video_gen_backends.py`）。其余按建议开工。

| # | 决策点 | 状态 |
|---|---|---|
| D1 | 代码底座：以 Maestro 为底座，移植父项目真实资产（ffmpeg 拼接、SQLite+FAISS、场景检测、CSA arc-judge 思想） | 按建议执行 |
| D2 | 任务定义 = "素材+指令 → 检索/生成混合长视频（含音频）" | 按建议执行 |
| D3 | 统一 skill 抽象（创作/评审/记忆三类 + 生命周期），严格 training-free | **已落地**：`memory/skill_admission.py`（证据门+回归门+评审门）、三类技能注册、版本化 |
| D4 | 物理：~~sketch-as-verifier~~ → **参考自由像素验证器**（v0.4：laws/reliability/router/verifier） | **已落地，定位重写**；README/COMPARISON 已同步 |
| D5 | 记忆：双寄存器实体记忆 + 验证门控写入；跨项目用户记忆；RL 保留策略不做 | **已落地**：`EntityIdentity⊕EntityState`+类型化转移日志（状态由日志重放=全程可审计）、`write_gate.py`（只提交渲染中验证到的）、`reentry_context` 长间隔复现负载（151 tests） |
| D6 | 音频先占位（接口+mock），视觉链路真实信号优先 | 按建议执行 |
| D7 | "跑通"= 最小真实链（真 LLM + 真生成后端[WaveSpeed API 最快] + 真 VLM critic + 真 CoTracker + 真 ffmpeg） | WaveSpeed 后端已实现，待服务器配 key 验证 |
| D8 | 评测：EntityBench / VideoPhy-2 / Physics-IQ / UniVA-Bench / AV-Align + 反循环协议 | 按建议执行 |

---

## 5. 风险与诚实声明（提前防 reviewer / 防自欺）

- **元错误防线**（来自父项目 CRITICAL_REVIEW）：任何新机制必须先指明它消费的**真实信号源**；
  禁止在 mock 输出上再叠 mock。每个创新点配一个"信号溯源"测试。
- 物理线的收益上界 = 基础生成器的样本多样性（N 个里没有一个物理正确时，选择无能为力）——
  S4 定向重生成是答案，必须显式消融。
- 自评闭环的循环性：选择用自家 oracle、评测必须用独立轴（VideoPhy-2 AutoEval、人评），
  并在 oracle 里加反作弊正则（慢动作、静态场景会平凡地"守恒"）。
- 技能库的坏习惯固化风险（AutoSkill 的教训）：准入测试 + 回归测试 + EMA 淘汰是硬要求。
