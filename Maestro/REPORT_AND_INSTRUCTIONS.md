# Maestro — 物理可信 · 自改进 · 多模态 Agentic Video Generation

**Report & Build Instructions（写代码前的总纲）**

> 代号 **Maestro**（可改）。一个 **training-free、多 agent、自闭环自改进、物理可信(physically-grounded)** 的 **视频生成(generation)** 智能体框架。用户提供自然语言指令 + 可选多模态素材（视频/图像/音乐），系统通过"多 agent 规划 → 生成 → 多维 metric 评审 → 关键帧级局部修正 → 物理校验"的闭环产出视频。
>
> 本文是给后续写代码的人（包括 Claude Code）的总纲：先讲清楚**为什么这么做（差异化创新点）**，再讲清楚**怎么搭（架构 + 数据结构 + agent 规格 + 实现优先级）**。
>
> 关系说明：本项目是**全新独立 project**，但**复用旧 `VideoAgent`(LongVideoEditAgent) 仓库中被验证有用的资产**——hierarchical narrative memory、感知模块(CLIP/光流/显著性/角色ID/镜头属性)、6 个量化 metric、beat-sync 音乐分析、结构化 trajectory log、配置驱动工程范式。旧项目重心是 *editing/montage*；Maestro 重心是 *generation*。

---

## 第一部分 · Report（分析与创新点）

### 1. 目标重述

把重心从"长视频剪辑(editing)"转到"视频生成(generation)"，并满足四个硬约束：

1. **Self-improving 优先**：框架的灵魂是自闭环。生成不是一次性的，而是"生成→评审→定位问题→局部修正→再评审"的单调改进循环。
2. **Generation 为主 + 多模态素材输入**：用户除指令外，可提供视频/图像/音乐作为素材，生成需被这些素材 *grounded*（身份一致、风格一致、音乐驱动结构）。
3. **Physical grounding**：物理合理性是**一等公民**，不是附带评分项。
4. **Training-free**：不微调底层生成/理解模型，只做推理时(test-time)的编排、评审、反馈、记忆。能力来自 agent 闭环，而非重训。

### 2. 现有框架横向对比

基于对开源代码与论文的真实调研（出处见第三部分参考文献）：

| 框架 | 定位 | 多 agent | Understanding | Generation | Self-improve | 物理 | Training-free | 关键短板 |
|---|---|---|---|---|---|---|---|---|
| **UniVA** (univa-agent) | 全能视频通才(产品级编排) | Plan+Act 双agent + MCP工具 | ✅ | ✅(任意条件→视频) | 工作流级自反思 | ❌ | ✅ | 物理缺位；自反思不沉淀能力；前端重、研究可复现性弱 |
| **VideoAgent** (HKUDS) | 理解+编辑+二创 | 图驱动 Agentic Graph Router | ✅强 | ⚠️只做remix/改编 | 两步自评估(0.95成功率) | ❌ | ✅ | 不做从零原创生成；无长期记忆；自评估只评"工作流是否可执行" |
| **ViMax** (HKUDS) | idea→长视频生成 | 导演/编剧/制片/生成器 | ❌ | ✅强(Idea/Novel/Script2Video) | 一致性检查(生成多张+VLM挑) | ❌ | ✅ | 不做理解/编辑；Agents Loop 仍是 TODO；一致性靠启发式；无量化基准/论文 |
| **VISTA** (Google) | test-time 自改进视频agent | Planner+锦标赛Judge+3维critic+改写agent | — | ✅(改prompt驱动现成T2V) | ✅最完整(多维critic+迭代+early-stop) | ⚠️软约束(MLLM评"物理常识") | ✅ | 只在 **prompt 层** 黑盒**整段重生成**；物理无硬保证 |
| **M3** (UIUC) | 高保真 T2**I** 组合生成 | Planner/Checker/Refiner/Editor/Verifier | — | ✅(图像) | ✅最细粒度(checklist+单调Verifier+escape hatch) | ❌(纯静态语义) | ✅ | 只做静态图像；无时序、无运动、无物理 |
| **Event-Graph** (2604.10383) | 文本→可执行事件图→引擎渲染 | Director+SceneBuilder+Relation子agent | — | ✅(3D引擎确定性执行) | ❌(实验证明事后refine失败,改靠"构造保证") | ✅唯一硬物理(引擎,physical validity 58% vs 神经 20-25%) | ✅ | 非照片级真实；表达力受引擎/GEST约束 |

### 3. 关键空白点（差异化的土壤）

调研得到的、所有框架**共同**的空白：

- **空白①｜物理不是一等公民**。要么完全没有（UniVA/VideoAgent/ViMax/M3），要么只是某个泛 critic 里的一句"physical commonsense"软评分（VISTA），要么只能靠牺牲视觉真实度的纯引擎渲染（Event-Graph）。**没人把"神经生成器的视觉真实度"与"物理引擎/事件图的硬约束"结合起来。**
- **空白②｜自改进是"整段黑盒"而非"局部精修"**。VISTA 在 video 上做到了闭环，但停在改 prompt + 整段重生成，昂贵且不可控。M3 在 *image* 上做到了"checklist→局部编辑→Verifier 单调改进→escape hatch"的精修范式，但**没人把这个 image 范式扩展到 video（关键帧级编辑 + 时序一致性约束）**。
- **空白③｜自改进是"任务内一次性"而非"跨任务能力沉淀"**。所有框架的自反思都只在单次生成内做局部修正，**没人把"这一轮犯的物理/一致性错误 + 成功修法"沉淀成可检索的经验库**，供未来生成复用。
- **空白④｜评审缺"可定位→可执行"的桥**。现有物理 benchmark 能打分/标 glitch（VideoPhy-2/PhyGenBench/Physics-IQ 显示 SOTA 物理合理性仅 ~22–24%，守恒律/因果最弱），但 agent 拿到分数后只能笼统改 prompt——**缺"哪条物理定律在第几帧违反 → 映射到哪个具体生成干预"的机制**。
- **空白⑤｜多模态素材 grounding 浅**。素材输入大多停在"参考图喂给生成器"，缺"用户素材 → 结构化资产记忆 → 跨镜头身份/风格/音乐结构一致性"的统一机制（而这恰是旧 VideoAgent 仓库已有积累的强项）。

### 4. Maestro 的差异化创新点（核心 6 + 增强 4）

> 设计哲学一句话：**把 M3 的"局部精修单调改进"范式扩展到 video，把 Event-Graph 的"硬物理约束"降维成 training-free 的物理 critic + 草图层，把 VISTA 的"多维 critic 迭代"接上一个跨任务沉淀的经验记忆，再在生成层引入 *adaptive scope* 的 HSI 多档升级、把物理草图升级为前向控制 + 后向一致性的双向闭环——这些空白的交集，就是 Maestro。**

**核心创新（必须做）**

- **C1 · Physics-as-first-class：双层物理 grounding（oracle 层 + critic 层）**
  > ⚠️ **2026-06 重新定位**（依据 `PHYSICS_LITERATURE_REVIEW.md`）：原"草图当 control signal 去 condition 冻结生成器（引擎管物理，扩散管渲染）"的表述**撤销**——文献调研证实该路线无人验证且有结构性硬伤（轨迹欠定物理、OOD 合成轨迹服从性空白、轨迹控制需训练专门分支）。新表述如下。
  - *oracle 层（Physics Sketch 作为验证预言机）*：LLM 把场景规划成轻量"事件图/物理草图"（物体、初速度、受力、碰撞），用**轻量仿真**算出**期望轨迹**——它不喂给生成器，而是作为 **ground-truth 参考**：用点追踪/光流（v0.3: CoTracker/RAFT）从**生成出来的视频**抽取**观测轨迹**，做 PISA 式归一化 Trajectory-L2 对比 + Morpheus 式守恒检查（`physics/oracle.py`）。仿真的显著点（最高点/接触瞬间）兼作**关键帧锚点提示**，经 I2V 首/末帧——冻结模型唯一可靠服从的条件通道——进入生成。training-free：仿真器是评测工具，不训练底模。物理提升的真正引擎是 **best-of-N/tournament + world-model reward（对标 WMReward 2601.10553，PhysicsIQ Challenge 第一）+ 单调 Verifier** 的 test-time 搜索。
    > **草图保真度（v0.3 落地，`physics/sim_wrapper.py`）**：oracle 的判别力取决于 ground-truth 轨迹有多对，因此 `MockSimulator` 从纯抛物线升级为**带碰撞响应的半隐式欧拉积分器**——地面平面（恢复系数反弹、不穿地）、单面墙（横向反弹）、支撑约束（静止不下坠）、并记录**接触帧事件**(`ground_bounce`/`wall_bounce`，供 critic/Refiner 定位修复帧)。v0.4 换 MuJoCo/Newton，oracle 数学与管道不变。
  - *critic 层（物理失败模式定位）*：专门的 **PhysicsCriticAgent**，不输出一个笼统分数，而是按**失败模式分类法**(穿模/重力惯性/碰撞/流体/物体恒存(object permanence)/形变/守恒律)逐项检查，输出"**第 t 帧违反 X 定律 + 严重度 + 建议干预**"的结构化反馈。复用 PhyGenEval 式分层 VLM 评测(单帧→多帧→全视频)作为零训练评测器。

- **C2 · Keyframe-level 局部自改进（把 M3 扩到 video）**
  - 不做 VISTA 式整段重生成。把视频分解为**关键帧 + 帧间过渡**；评审定位到具体关键帧/区段；只对失败的关键帧做**局部编辑 (image-edit 模型)**，再用 first/last-frame 条件的视频生成做**局部 regeneration + 时序传播**。
  - 借鉴 M3 三件套保证收敛：**Checklist**(把指令+物理+一致性拆成可验证 yes/no 项) → **Verifier 单调改进**(只接受"相对上一最优更好"的修改，不退化) → **Escape hatch**(单项重试 K 次仍失败则跳过，防死循环)。

- **C3 · 自闭环 = 多 agent review × metrics design（用户明确要的点）**
  - 一套**可量化 metric 套件**驱动闭环（不是黑盒打分）：语义对齐、时序一致性、**物理合理性(分失败模式)**、身份一致性、音乐同步、美学。复用旧仓库已有的 6 个 metric（m1 prompt 相关性 … m6 能量对应）并新增物理维度。
  - **多 agent review**：生成 agent ↔ 多个 critic agent（语义/物理/一致性/节奏）↔ 仲裁/改写 agent，闭环直到所有 metric 过阈或 early-stop。这是相对 ViMax "Agents Loop（TODO）" 和 VideoAgent "只评工作流可执行性" 的实质差异。

- **C4 · 跨任务经验记忆（capability-level self-improvement）**
  - 一个 **Lesson/Constraint Library**：把每轮"物理/一致性失败 + 成功修法 + 触发条件"沉淀成可检索条目；新生成任务在规划阶段检索相关 lesson，提前注入约束（"涉及倒水→强制流体连续性检查 + 降速 prompt"）。
  - 这让自改进从"任务内一次性"升级为"越用越好"，是所有现有框架都没有的"能力级"而非"工作流级"自提升。
  - **v0.2.1 修正**：沉淀的 mode 不再固定写 `expected_modes[0]`，而是取**初始 verdict 集合 − 终态 verdict 集合 − escape-hatch 集合**——即真正被本轮修复的 mode，避免 lesson 库被"挂名"条目污染。

- **C5 · Hierarchical Self-Improvement (HSI) — adaptive scope 多档升级（v0.2.1 新增）**
  - 现状：VISTA 永远整段重生（昂贵），M3 永远局部 patch（scope 受限），二者中间没人填。Reflexion/Self-Refine 只在单一 scope 内迭代。
  - Maestro 的 HSI：生成层把自改进拆成 **由便宜到昂贵的多档**，由 critic 给出的失败信号驱动 *按需升级 scope*：

    | tier | 动作 | 借鉴 | cost 量级 |
    |---|---|---|---|
    | 0 | Refiner 选定关键帧 → image-edit → 用作 first-frame 做局部 regen | M3 | ⭐（一次 image edit + 一次 local I2V） |
    | 1 | PhysicsPlanner.replan 把 sketch 的初速降到 0.55x → 重出 control signal → regen | C1 草图层 | ⭐⭐（重出 control + regen） |
    | 2 | Director.refine_spec：cinematography 降速放宽 + prompt 注入 `plan-fix:<worst hint>` → regen | VISTA prompt rewrite | ⭐⭐⭐（rewrite spec + regen） |
    | 3 | escape hatch：丢弃最严重 verdict 并刷新 metric | M3 escape | ⭐ |

  - **每档**都跑 `k_retries` 次，**每次**都过 Verifier 单调改进闸（任一档失败到下一档，决不接受退化候选）；接受候选后**复位到 Tier 0**，下一轮再从最便宜档开始（cost-amortized）。
  - report 里的 `tier_used` / `escalations` 暴露这一动态档位行为，trajectory 里能看到 `replan_sketch` / `refine_spec` action。这是相对 VISTA/M3 的实质性 paradigm 扩展。

- **C6 · 物理草图 ↔ 视频 oracle 一致性（v0.2.1 新增，v0.3 随 C1 一并重定位）**
  - 现状：VISTA 用 VLM 评一句"physical commonsense"（无中间物理参考可比），Event-Graph 用引擎直接渲染（牺牲真实度）；**没人拿一个物理正确的参考轨迹去核验生成视频实际的运动**。
  - Maestro 的 `PhysicsConsistencyCritic`：把 sketch **只当后向验证基准**（与 C1 reframe 一致——不再声称前向控制）。生成 clip 后用 track extractor 抽**观测轨迹**，与 oracle 的**期望轨迹**做 PISA 式归一化 Trajectory-L2（`physics/oracle.py:TrajectoryOracle`）；divergence 超阈值发 `CONSERVATION`-mode verdict 进 Review Board，**定位到最差实体 + 严重度**，驱动 HSI 修正（接触帧事件给 Refiner 精确锚点）。
    > **track extractor 已可换真实后端**（`physics/track_extractor_backends.py` + `oracle.build_track_extractor` 工厂，配置 `models.track_extractor.name`）：默认 `mock-track`（CPU 无依赖）；`cotracker`/`tapir` 懒加载 torch，解码真 mp4 帧 → 每实体起点设 query → 点追踪 → 归一化回 [0,1] 屏幕空间。优雅降级：mock 管道产出的文本占位"视频"无像素 → 解码返回 None → oracle 静默（诚实：非视频无法追踪）；若显式配置真后端但库/权重缺失则**响亮报错**，绝不静默给出虚假完美 p2。
  - MetricTool 拆出 `p1_physics`（原生失败模式）vs **`p2_sketch_consistency`**（观测 vs 期望轨迹的偏差），两类失败可独立诊断、独立加权。可选再叠 `wm_reward`（world-model reward，`models/world_reward.py`）作为第三个物理信号。
  - 核心立论从"我们能用草图控制生成"改为"**我们把物理正确性变成一个 test-time 可搜索的、可定位的验证目标**"——既诚实（承认轨迹控制不可靠），又把贡献立在文献支持的验证+搜索范式上（PISA / Morpheus / WMReward）。

**增强创新（强烈建议，差异化加分）**

- **E1 · 多模态素材 → 统一资产记忆 grounding**。复用旧仓库的 narrative memory + 感知栈：用户上传的视频→拆 shot 建记忆；图像→身份/风格 anchor；音乐→beat/段落驱动生成的镜头时长与节奏(沿用 m5 beat-sync / m6 energy)。生成全程被资产记忆约束，保证跨镜头身份/风格/节奏一致。
- **E2 · 物理草图作为"可执行中间表示"**。借 Event-Graph 的 GEST 思想，但只用它做**约束与可解释性**(可视化"系统以为的物理")，渲染交给神经生成器——兼顾可控与真实。
- **E3 · 锦标赛式候选选优 + 双向消偏**。借 VISTA 的 binary tournament（双向交换比较消除 MLLM token bias）做候选挑选，比 ViMax "生成多张 VLM 挑一张"更鲁棒。
- **E4 · 全程结构化 trajectory log**。每个 agent 决策、metric、修正都落 JSONL（沿用旧仓库范式）。即便 training-free，也为**未来可选**的 reward model 微调 / agentic RL 预留干净数据接口（不在 v1 做）。

### 5. 一句话定位差异

> **UniVA 全而不深、VideoAgent 不做原创生成、ViMax 闭环还是 TODO、VISTA 整段黑盒重生成、M3 只到图像、Event-Graph 牺牲真实度。Maestro = 把"物理一等公民 + 关键帧级局部精修 + 多agent metric 闭环 + 跨任务经验记忆"四者合一的 training-free 视频生成 agent。**

---

## 第二部分 · Instructions（新项目搭建蓝图）

### 6. 工程原则（沿用旧仓库被验证的好习惯）

- **配置驱动**：agent 角色、模型选择、metric 权重、物理失败模式阈值全部走 YAML，不 hardcode。
- **离线/在线分离**：素材预处理(建资产记忆)离线缓存；agent 推理只读 cache。
- **接口先行**：所有跨模块通信用 dataclass/TypedDict，不用裸 dict。
- **模型包 wrapper**：LLM / MLLM / 视频生成 / 图像编辑 / 物理仿真都包稳定 wrapper，换模型只改 wrapper。
- **mock-first**：v0.1 所有重模型(视频生成/MLLM/仿真)先 mock，CPU 可跑通端到端；v0.2 再 swap 真实模型。
- **prompt 独立成文件**：`prompts/*.txt`，不写进 `.py`。
- **结构化日志**：所有决策/metric/修正写 JSONL trajectory。
- **单调改进是硬规则**：任何自改进步骤必须经 Verifier 确认"不退化"才接受。

### 7. 系统总览

```
┌──────────────────────────── Inputs ────────────────────────────┐
│  User Prompt(指令)   │  Materials(可选: 视频/图像/音乐)          │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────▼─────────────────────┐
         │ Stage 0 · Material Understanding (offline)│  复用旧仓库感知栈
         │  视频→shot记忆 / 图像→身份·风格anchor /   │
         │  音乐→beat·段落 →  AssetMemory            │
         └────────────────────┬─────────────────────┘
                              │
         ┌────────────────────▼─────────────────────┐
         │ Stage 1 · Planning (multi-agent)          │
         │  ScreenwriterAgent → 叙事/分镜脚本         │
         │  DirectorAgent     → 每镜头 ShotSpec       │
         │  PhysicsPlannerAgent → Physics Sketch      │ ← C1草图层 / E2
         │  (检索 Lesson Library 注入约束)            │ ← C4
         └────────────────────┬─────────────────────┘
                              │
         ┌────────────────────▼─────────────────────┐
         │ Stage 2 · Generation + Self-Improve Loop  │  ← 框架灵魂
         │  for each Shot:                            │
         │   GeneratorAgent → 候选(草图条件下生成)    │
         │   ┌── Review Board (并行 critic) ──┐       │ ← C3
         │   │ SemanticCritic / PhysicsCritic │       │ ← C1 critic层
         │   │ ConsistencyCritic / RhythmCritic│      │
         │   └────────────┬───────────────────┘       │
         │   Verifier(单调改进) → 接受/拒绝            │ ← C2
         │   若拒绝: 定位关键帧 → 局部编辑/regeneration│ ← C2
         │   Tournament 选优 + Escape hatch            │ ← E3
         │   沉淀 Lesson → Library                     │ ← C4
         └────────────────────┬─────────────────────┘
                              │
         ┌────────────────────▼─────────────────────┐
         │ Stage 3 · Assembly (ffmpeg)               │
         │  拼接 + 叠音乐 + transition → output.mp4   │
         └────────────────────┬─────────────────────┘
                              ▼
              Output Video + Metric Report + Trajectory Log
```

### 8. 核心数据结构（最先写，`types.py`）

```python
# ── 素材 / 资产记忆 ──
@dataclass
class AssetMemory:
    """用户素材的结构化记忆。复用旧仓库 NarrativeMemory 思路。"""
    video_shots: dict[str, Shot]          # 上传视频拆出的 shot(可复用旧 Shot)
    identity_anchors: dict[str, Identity] # 图像/视频里的人物/物体身份(face/clip embedding)
    style_anchors: list[StyleRef]         # 风格参考(图像 CLIP embedding + 描述)
    music_profile: Optional[MusicProfile] # beat/downbeat/段落(复用旧 MusicProfile)

# ── 规划产物 ──
@dataclass
class ShotSpec:
    """DirectorAgent 输出，一个镜头的完整规格。"""
    shot_idx: int
    duration: float
    prompt: str                           # 文本指令(给生成器)
    cinematography: CinematographyTags     # 复用旧 dataclass
    identity_refs: list[str]              # 引用 AssetMemory.identity_anchors 的 key
    style_refs: list[str]
    rhythmic_pacing: list[int]            # 按 beat 的剪辑节奏(复用旧思路)
    physics_sketch: Optional["PhysicsSketch"] = None
    injected_lessons: list[str] = field(default_factory=list)  # C4 注入的经验

# ── 物理 ──
@dataclass
class PhysicsSketch:
    """C1 草图层 / E2。场景的轻量物理表示 + 仿真得到的控制信号。"""
    entities: list["PhysEntity"]          # 物体: 质量/初速/受力
    interactions: list["PhysInteraction"] # 碰撞/支撑/流体/约束
    control_signal: Optional[Path] = None  # 仿真渲染的轨迹/depth/flow 控制图(喂生成器)

class PhysFailureMode(str, Enum):
    PENETRATION = "penetration"           # 穿模
    GRAVITY_INERTIA = "gravity_inertia"   # 重力/惯性
    COLLISION = "collision"
    FLUID = "fluid"
    OBJECT_PERMANENCE = "object_permanence"
    DEFORMATION = "deformation"
    CONSERVATION = "conservation"         # 守恒律(最弱项)

@dataclass
class PhysicsVerdict:
    """PhysicsCriticAgent 输出: 可定位→可执行。"""
    mode: PhysFailureMode
    frame_range: tuple[int, int]          # 第几帧违反
    severity: float                       # 0-1
    suggested_intervention: str           # 映射到具体修正动作

# ── 评审 / 自改进 ──
@dataclass
class Checklist:
    """C2/M3 范式: 指令+物理+一致性拆成可验证项。"""
    items: list["ChecklistItem"]          # 每项: 问题 + 类型 + 是否 pass + 修复指令

@dataclass
class CandidateClip:
    shot_idx: int
    video_path: Path
    keyframes: list[Path]                 # 抽出的关键帧(供局部编辑)
    metric_scores: dict[str, float]       # metric 套件打分
    physics_verdicts: list[PhysicsVerdict]
    checklist: Checklist
    accepted: bool = False
    revision: int = 0                     # 第几轮修正

@dataclass
class Lesson:
    """C4 经验库条目。"""
    trigger: str                          # 触发条件("倒水/抛物/快速转身"…)
    failure_mode: Optional[PhysFailureMode]
    fix: str                              # 成功修法
    embedding: Optional[np.ndarray] = None # 供检索

# ── trajectory(E4) ──
@dataclass
class AgentStep:
    timestamp: float; agent_name: str
    state_snapshot: dict; action: str
    action_input: dict; observation: dict
    reward: Optional[float] = None        # 预留, v1 可空
```

### 9. Multi-Agent 规格

每个 agent 是 `BaseAgent` 子类，`run(state)->partial_state`，prompt 从 `prompts/` 读，每步落 trajectory。

| Agent | 输入 | 输出 | 职责 | 借鉴 |
|---|---|---|---|---|
| **ScreenwriterAgent** | prompt + AssetMemory.summary + music | 叙事结构/分镜大纲 | 把 idea 拆成场景/镜头序列；音乐驱动结构锚定 | ViMax 编剧 / 旧仓库 Screenwriter |
| **DirectorAgent** | 分镜 + AssetMemory | `list[ShotSpec]` | 每镜头定 prompt/镜头语言/身份风格引用/节奏 | ViMax 导演 / 旧 Director 三段CoT |
| **PhysicsPlannerAgent** | ShotSpec | `PhysicsSketch` | 抽物体/受力/交互→仿真→控制信号；检索 Lesson 注入 | Event-Graph(GEST) + C1/C4 |
| **GeneratorAgent** | ShotSpec + sketch + anchors | `CandidateClip` | 调视频生成工具(草图/首帧/参考图条件);多候选 | VISTA/ViMax |
| **SemanticCritic** | clip + ShotSpec | checklist 语义项判定 | 属性/计数/空间/文本对齐 | M3 Checker |
| **PhysicsCritic** | clip | `list[PhysicsVerdict]` | 按失败模式分层 VLM 定位物理错误 | PhyGenEval + C1 |
| **ConsistencyCritic** | clip + anchors + 邻镜头 | 一致性分 + 定位 | 身份/风格/跨镜头连续性 | 旧 m2/m4 |
| **RhythmCritic** | clip + music | 节奏分 | beat-cut 同步/能量对应 | 旧 m5/m6 |
| **VerifierAgent** | 上一最优 vs 新候选 | accept/reject | **单调改进**仲裁，只接受不退化的修正 | M3 Verifier |
| **RefinerAgent** | verdicts + checklist 失败项 | 修正动作(局部编辑/regen/改prompt) | 把"可定位"翻译成"可执行" | M3 Refiner + C2 |

> 编排建议：用 LangGraph（state 持久化、条件边、易扩展）或自写 mini state machine。Review Board 的多 critic **并行**调用。

### 10. Self-Improving 闭环（Stage 2 伪码，框架灵魂）

**v0.2.1 起为 HSI 多档（C5）**：

```
for shot in shot_specs:
    # 初始候选（在草图条件 + identity refs 下出多个）
    candidates = [GeneratorAgent.generate(shot, seed=s) for s in range(n_candidates)]
    for c in candidates: ReviewBoard.review(c)      # critic + metric_tool
    best = Tournament.select(candidates)            # E3 双向消偏选优
    initial_modes = {v.mode for v in best.physics_verdicts}    # snapshot 给 C4 用

    for r in range(1, max_revisions + 1):
        if board.all_passed(best): break

        accepted = None

        # ── Tier 0 · keyframe-level 局部精修 (M3-style, 最便宜) ──
        plan = RefinerAgent.plan(best)
        for k in range(k_retries):
            ff = ImageEdit.edit(best.keyframes[plan.kf_idx], plan.edit_instr)
            cand = Generator.generate(shot, first_frame=ff, extra_prompt=plan.extra)
            ReviewBoard.review(cand)
            if VerifierAgent.is_better(cand, best):    # 单调改进硬约束
                accepted = cand; break

        # ── Tier 1 · 物理草图加严（重出 control signal） ──
        if accepted is None and physics_planner:
            PhysicsPlanner.replan(shot, strictness=0.55)
            for k in range(k_retries):
                cand = Generator.generate(shot)
                ReviewBoard.review(cand)
                if VerifierAgent.is_better(cand, best):
                    accepted = cand; break

        # ── Tier 2 · ShotSpec 重写（VISTA-style 但局限于本镜头） ──
        if accepted is None and director:
            Director.refine_spec(shot, hint=worst_verdict.intervention)
            for k in range(k_retries):
                cand = Generator.generate(shot)
                ReviewBoard.review(cand)
                if VerifierAgent.is_better(cand, best):
                    accepted = cand; break

        # ── Tier 3 · escape hatch（防死循环） ──
        if accepted is None:
            escape_hatch(best)                          # 丢一个最严重 verdict
            board.recompute_metrics(best)               # ★ 修陈旧 metric: 让 Verifier 下一轮比较用最新分
        else:
            best = accepted                              # 接受后下一 revision 重新从 Tier 0 开始

    Lesson.distill(initial_modes − final_modes − skipped)  # ★ C4: 用实际被修复的 mode
    LessonLibrary.add(spec.prompt, fix, failure_mode)
    script.append(best)

assemble(script, music) → output.mp4 + metric_report + trajectory
```

设计要点：
- **adaptive scope**：失败才升级，便宜档优先，cost-amortized。
- **monotonic 贯穿全档**：Verifier 在每一档每一次重试都把关，不会因为换档而放水。
- **接受后复位 Tier 0**：下一轮 revision 从最便宜档重新开始（不锁档），避免被一次困难带高 cost。
- **Tier 1/2 为可选依赖**：`generate_shot(..., physics_planner=None, director=None)` 时直接退化为单档 M3 模式，向后兼容。
- 停止条件：所有 checklist 项 pass/skip、达 `max_revisions`、或最优连续 m 轮不变(early-stop)。

### 11. Metric 套件设计（C3 的量化基础）

复用旧仓库 6 个 + 新增物理/身份维度：

- `m1` 语义对齐：CLIP(clip, prompt) / VLM checklist 通过率
- `m2` 时序一致性：相邻帧/镜头 embedding 连续性
- `m3` 运动连续性：光流幅度+方向相似度（旧）
- `m4` 构图/取景一致性：显著性分布距离（旧）
- `m5` beat-cut 同步：剪点与节拍距离（旧）
- `m6` 能量对应：运动幅度与音乐 RMS 相关（旧）
- **`p1` 物理合理性 — 原生失败模式**(C1)：PhyGenEval 式分层 VLM，按失败模式加权惩罚；只看 PENETRATION/GRAVITY/COLLISION/FLUID/OBJECT_PERMANENCE/DEFORMATION verdict
- **`p2` 物理一致性 — oracle 轨迹偏差**(C6)：`PhysicsConsistencyCritic` 用 `TrajectoryOracle` 比较**观测轨迹 vs 仿真期望轨迹**（PISA 式归一化 Trajectory-L2）；只看 CONSERVATION verdict。和 p1 分开报，让"运动偏离物理预期"这类失败可独立诊断
- **`wm_reward` 世界模型奖励**(可选, C1 oracle 增强)：`models/world_reward.py`，对标 WMReward(2601.10553)/VJEPA-2 Reward(2510.21840)；未配置时 MetricTool 输出与 v0.2.2 完全一致
- **`id1` 身份一致性**(新)：跨帧/跨镜头 face/object embedding 方差

每项可在 `configs/default.yaml` 的 `metrics.weights` 改权重；不同镜头类型用不同 preset（沿用旧 heuristic presets 思路）。

### 12. 工具层 `tools/`

| Tool | 功能 | v0.1 mock / v0.2 真实 |
|---|---|---|
| `video_gen_tool` | 文本/首帧/参考图/控制图条件视频生成 | mock 返回固定 mp4 / Wan·OmniWeaving·Veo API |
| `image_edit_tool` | 关键帧局部编辑 | mock / Qwen-Image-Edit 类 |
| `physics_sim_tool` | 轻量仿真→轨迹/depth/flow 控制信号 | mock 几何先验 / MuJoCo·Newton·粒子仿真 |
| `vlm_critic_tool` | 分层 VLM 评测(物理/语义) | mock 打分 / Qwen-VL 等 |
| `retrieval_tool` | 资产记忆/Lesson 检索 | FAISS(可真实, 轻量) |
| `assembly_tool` | ffmpeg 拼接+叠乐+transition | **真实**(否则没法出 mp4) |
| `metric_tool` | 算 metric 套件，供 agent 主动查 | 真实(CPU 可算) |

> 视频生成器优先选**支持多条件(text+首帧+参考图+控制信号)** 的，否则草图层无意义。候选：OmniWeaving / Wan / Veo·Sora(API 兜底)。

### 13. 评测设计（差异化也要可证明）

- **物理**：对接 VideoPhy-2 / PhyGenBench / Physics-IQ；报告 `p1` 与人评相关性。
- **一致性/叙事**：跨镜头身份方差、人评叙事连贯。
- **自改进有效性**(核心卖点)：画"revision 轮数 vs metric"曲线，证明单调改进 + 收敛；对比 VISTA 式整段重生成的**成本(生成调用次数)**，证明关键帧局部精修更省。
- **跨任务记忆**(C4)：开/关 Lesson Library 的 A/B，证明"越用越好"。
- **消融**：草图层开关、物理 critic 开关、Verifier 单调约束开关。

### 14. 仓库结构

```
maestro/
├── README.md
├── REPORT_AND_INSTRUCTIONS.md        # 本文
├── pyproject.toml / requirements.txt / .env.example
├── configs/
│   ├── default.yaml
│   ├── agents/{screenwriter,director,physics_planner,generator,
│   │            verifier,refiner}.yaml
│   ├── critics/{semantic,physics,consistency,rhythm}.yaml
│   ├── models/{llm,mllm,video_gen,image_edit,physics_sim}.yaml
│   └── metrics.yaml                  # metric 权重 + 物理失败模式阈值
├── src/maestro/
│   ├── types.py                      # 第8节所有 dataclass(最先写)
│   ├── config.py / logging.py / trajectory.py
│   ├── memory/                       # 复用旧仓库: schema/store/builder/retriever
│   │   └── lesson_library.py         # C4 经验库(新)
│   ├── perception/                   # 复用旧仓库感知栈(素材理解)
│   ├── physics/                      # ⭐新: sketch.py / failure_modes.py / sim_wrapper.py
│   ├── agents/                       # 第9节各 agent
│   ├── critics/                      # Review Board 各 critic
│   ├── tools/                        # 第12节
│   ├── models/                       # LLM/MLLM/video_gen/image_edit/physics wrappers
│   ├── orchestration/                # LangGraph state/graph/messages
│   ├── pipeline/                     # understand / plan / generate_loop / assemble / run
│   └── prompts/*.txt
├── benchmark/                        # videophy / phygenbench / selfimprove_eval
├── scripts/                          # preprocess / build_memory / run_pipeline / eval / viz_trajectory
└── tests/{unit,integration,fixtures/tiny_clip.mp4}
```

### 15. 实现优先级（按此顺序写，接口+单测先行）

1. `types.py`（全员依赖）→ `config.py` → `logging.py` + `trajectory.py`
2. `memory/`（复用旧仓库）+ `lesson_library.py`（C4 接口，可先空实现）
3. `perception/`（复用旧仓库，mock 优先）→ `pipeline/understand.py`（建 AssetMemory，跑通 fixture）
4. `agents/base.py` + `critics/base.py` + `tools/base.py`（ABC 接口）
5. `physics/`（**新模块，差异化核心**）：failure_modes 分类法 → sketch 数据结构 → sim_wrapper(mock 几何先验)
6. `prompts/*.txt`（占位，能让 mock LLM 跑通）
7. `orchestration/`（LangGraph state + graph）
8. agents：Screenwriter → Director → PhysicsPlanner → Generator（逐个 mock 跑通）
9. critics：Semantic → Physics → Consistency → Rhythm（并行 Review Board）
10. `agents/verifier.py` + `refiner.py`（**自改进闭环 + 单调约束 + escape hatch**）
11. `tools/`：metric_tool(真实) → assembly_tool(真实 ffmpeg) → 其余 mock
12. `pipeline/generate_loop.py`（第10节伪码）→ `assemble.py` → `run.py`
13. `scripts/*` + `tests/*`

### 16. v0.1 验收标准（脚手架）

- `pytest tests/unit/` 全绿（mock 实现即可）
- `pytest tests/integration/test_end_to_end.py` 跑通：输入 `tiny_clip.mp4` + 一句 prompt + 一段音乐，所有重模型 mock，产出一个 `.mp4` + metric 报告 + trajectory JSONL
- **自改进闭环可见**：trajectory 里能看到 ≥1 次"critic 定位失败 → refiner 局部修正 → verifier 接受/拒绝"的完整回合（即便是 mock 分数）
- **物理模块可见**：PhysicsSketch 生成 + PhysicsVerdict 输出 + 失败模式分类（mock）
- 改 `configs/*.yaml` 某权重，运行时能反映
- `README.md`：介绍 + 安装 + quickstart + 模块图

### 17. 反模式（避免）

- ❌ 自改进做成"整段黑盒重生成"（那就退化成 VISTA，丢了 C2 差异化）
- ❌ 物理只给一个笼统分数（必须分失败模式 + 定位帧 + 可执行建议）
- ❌ Verifier 缺席导致越改越差（单调改进是硬规则）
- ❌ v0.1 依赖 GPU 真实模型（全 mock，CPU 跑通）
- ❌ prompt 写进 `.py`、跨模块用裸 dict、把整个 memory 塞进 state
- ❌ 把视频生成的多 GPU 调度写进 agent（agent 只调 tool）

---

## 第三部分 · 参考文献（带 URL，供加引用）

**开源框架**
- UniVA — Universal Video Agent. GitHub: https://github.com/univa-agent/univa ｜ arXiv: https://arxiv.org/abs/2511.08521 ｜ 官网: https://univa.online/
- VideoAgent (HKUDS) — All-in-One Understanding & Editing. GitHub: https://github.com/HKUDS/VideoAgent
- ViMax (HKUDS) — Agentic Video Generation（截至调研**无公开论文**，信息来自 README）. GitHub: https://github.com/HKUDS/ViMax

**自改进 / 多 agent 生成**
- VISTA — A Test-Time Self-Improving Video Generation Agent (Google). arXiv: https://arxiv.org/abs/2510.15831 ｜ 项目页: https://g-vista.github.io/
- M3 — High-fidelity T2I via Multi-Modal, Multi-Agent, Multi-Round Visual Reasoning (UIUC). arXiv: https://arxiv.org/abs/2602.06166
- Agentic Video Generation: From Text to Executable Event Graphs via Tool-Constrained LLM Planning. arXiv: https://arxiv.org/abs/2604.10383

**物理 grounding 评测与方法**
- VideoPhy-2 — action-centric physical commonsense. arXiv: https://arxiv.org/abs/2503.06800 ｜ https://videophy2.github.io/
- PhyGenBench / PhyGenEval (OpenGVLab, ICML'25). arXiv: https://arxiv.org/abs/2410.05363 ｜ https://phygenbench123.github.io/
- Physics-IQ (DeepMind+INSAIT). arXiv: https://arxiv.org/abs/2501.09038 ｜ https://physics-iq.github.io/
- VBench-2.0 — intrinsic faithfulness(含 physical realism). arXiv: https://arxiv.org/abs/2503.21755

> 注：部分较新论文(M3 2602.x / Event-Graph 2604.x / 个别物理方法 2512–2603.x)编号显示为近未来日期，引用前请核验原文；上文逐一标 URL 的 benchmark(VideoPhy-2/PhyGenBench/Physics-IQ/VBench-2.0)与 VISTA 为已抓取或多源交叉确认的可靠条目。ViMax 无论文，勿引用为 paper。

---

*本文为 Maestro 项目的总纲。下一步：经你确认创新点与架构后，按第 15 节优先级生成 v0.1 脚手架代码。*

---

## 第四部分 · 版本演进 (changelog)

### v0.2.2（当前）· UniVA 借鉴：广度补齐 + 服务器化、零创新点改动
对照 **UniVA — Universal Video Agent**（arXiv:2511.08521 / github.com/univa-agent/univa）的 4 个值得借鉴的 pattern，把"广度"补上但保留 Maestro 的"深度"（C1-C6 完全不动）：

| UniVA pattern | Maestro 对应实现 | 文件 |
|---|---|---|
| MCP tool servers + 自描述工具 | `tools/base.py:ToolRegistry/ToolSpec/BaseTool.spec`；in-process，不上 wire 协议 | `src/maestro/tools/base.py` |
| Analysis / Generation / Editing / Tracking 四类工具 taxonomy | 在 UniVA 的 4 类基础上加 Maestro 自己的 `physics/metric/retrieval` 三类，共 7 类 | 同上 |
| Plan / Act dual-agent | 新 `ActAgent`：吃 `list[ToolCall]`、走 registry、写 `tool_call` 进 trajectory | `src/maestro/agents/act.py` |
| `univa_server.py` + `/health` | FastAPI server（`/health` 兼容 UniVA shape + `/tools` 工具清单 + `/generate` 异步 job + `/jobs/{id}`） | `src/maestro/server.py` |

新增的 9 个内置工具：

| 名 | 类别 | 说明 | 真后端 hook |
|---|---|---|---|
| `video_probe` | analysis | duration / fps / 分辨率 | ffprobe；缺则启发式 |
| `frame_extract` | analysis | 抽指定时间戳的帧 | ffmpeg；缺则写 placeholder |
| `caption` | analysis | 图/视频 → 自然语言描述 | v0.3：Qwen-VL |
| `detect_objects` | tracking | bbox + label + score | v0.3：Grounding-DINO/SAM |
| `image_ops` | editing | resize / crop | PIL；缺则文件拷贝 |
| `video_concat` | editing | 多片段 → 单 mp4 | ffmpeg；缺则 manifest |
| `assemble` | editing | 高阶拼接 + 音乐 + transition | 既有，加 category 标注 |
| `audio_gen` | generation | TTS / 音乐 / SFX | v0.3：Bark/MusicGen |
| `compute_metrics` | metric | metric 套件（m1/m2/p1/p2/id1/m5/aesthetic） | 既有，加 category |

可部署组件：
- **`maestro` CLI**：`smoke`（健康检查）/ `serve`（启 FastAPI）/ `run-once`（批量生成），由 `pyproject.toml [project.scripts]` 注册到 PATH。
- **`server.py`**：`/health` + `/tools` + `/generate` + `/jobs/{id}`；jobs 跑在 `ThreadPoolExecutor`（v0.3 → Redis/Celery，interface 稳定）。fastapi 是 optional dep，缺则 server 模块 import 不崩，只是无法 `create_app()`。
- **`Dockerfile`**：`python:3.11-slim + ffmpeg`；`HEALTHCHECK` 命中 `/health`；k8s/compose 探活开箱即用。GPU 升级路径在文件注释里。
- **`requirements.txt`** 锁版本 + 拆 optional extras（`server` / `image` / `all`）。
- **`.env.example`** 扩到 server / sandbox / 真后端 keys 全列。

测试：`pytest -q` → **54 passed**（CPU，~0.6s）。新加 `tests/unit/test_tools_and_act.py`（11 项工具 + Plan→Act handoff）+ `tests/unit/test_server.py`（4 项；fastapi 缺则自动 skip 整文件）。

> 设计取舍：**不**移植 MCP 的 wire 协议（v0.2.2 单 replica in-process 用不上），**不**移植 UniVA 前端（隔离关注点）。只取它的"工具自描述 + 4 类 taxonomy + Plan/Act 抽象 + FastAPI shape"四件套，这是真正让框架"完善 + server 化"的核心。

### v0.2.1 · paradigm 深化、零模型改动
不改 mock 模型，专门把"自改进 paradigm + 物理 grounding"做深，对应 §4 的 C5/C6：

- **C5 HSI 多档升级**：`pipeline/generate_loop.py` 重写为 Tier 0/1/2/3 escalation；
  新增 `agents/physics_planner.py:replan` 和 `agents/director.py:refine_spec`；
  `SelfImproveResult` 多出 `tier_used` / `escalations`，report 同步暴露。
- **C6 Sketch↔Video 一致性 critic**：新建 `critics/physics_consistency.py`；
  `tools/metric_tool.py` 拆 `p1_physics` / `p2_sketch_consistency`；默认权重重平衡。
- **逻辑硬化**：
  - `critics/board.py:recompute_metrics` 让 escape hatch 后 Verifier 的对比基准刷新到最新。
  - `pipeline/generate_loop.py:_distill_lesson` 沉淀的 mode = `初始 verdict 集 − 终态 verdict 集 − escaped`，不再盲取 `expected_modes[0]`。
- **测试**：新增 `tests/unit/test_hsi_and_consistency.py`（7 项），全套 `pytest -q` 35 通过、CPU < 0.2s。
- **向后兼容**：`generate_shot(physics_planner=None, director=None)` 自动退化为 v0.1 单档行为。

### v0.2 · 真模型 backend 骨架 + plan-level 自改进
- `models/video_gen_backends.py`：OmniWeaving / Wan / Veo 客户端骨架（同一 `generate(...)` 契约）。
- `physics/control_render.py`：sketch JSON → backend-agnostic `ControlSpec`，让一套 sketch 喂多种 backend。
- `planning/event_graph.py`：GEST-style IR + `validate_event_graph`，executable-by-construction。
- `agents/plan_validator.py`：plan-stage Critique-Correct-Verify，把不可 ground 的 ShotSpec 拦在生成前。
- `critics/tournament.py`：VISTA-style 双向消偏 binary tournament。
- `tools/retrieval_tool.py`：identity / style / 源 shot 检索 → 给 generator 当 reference。

### v0.1 · CPU-only mock-first scaffold
§7–§16 描述的最小可跑闭环：types → memory → physics 模块 → agent ABC → critics → loop → assemble → tests。所有重模型 mock，CPU 跑通 end-to-end，pytest 全绿。
