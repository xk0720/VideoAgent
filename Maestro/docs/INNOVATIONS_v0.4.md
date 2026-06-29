# Maestro v0.4 创新点全解：来源依据 · 实现原理 · 可行性分析

> 2026-06-12。本文是三轮对抗式自我审查（3 个独立审查代理，42 项发现）与三轮修复
> （29 项审查修复 + mock 信号内容化改造）之后的定稿描述——每个创新点写清楚
> **它从哪些已有工作来、它们为什么不够、我们的代码具体怎么做、以及在真实部署
> 下成立的条件与已知边界**。当前状态：`pytest -q` → **192 passed**；
> 端到端 mock 流水线收敛（修复后：1 轮修复即收敛，分数 0.454→0.894，
> 收敛由"修复是否被应用到产物"驱动而非 revision 计数）。
>
> 调研底稿：`docs/research/survey_{self_improve,memory,physics,skills_audio}_2026_06.md`
> （~80 篇 2024–2026 文献，每篇记录方法/局限/留下的空白）；
> 决策记录：`docs/research/INNOVATION_PLAN_2026_06.md`。

---

## 0. 框架一句话与统一原语

**Maestro**：training-free、多 agent、自改进的"素材+指令 → 长视频（含音频）"生成框架。
三个创新点不是并列模块，共享一个统一原语：

> **一切可改进之物皆是 skill**——agent 唯一可学习的基质是一个类型化、经验证准入、
> 版本化的技能库；物理评审器、记忆写入策略、创作工作流都是技能，走同一个
> 蒸馏 → 准入 → 检索 → 执行 → 评估 → 进化 的生命周期。

支撑这一切的工程纪律（来自父项目 `docs/CRITICAL_REVIEW.md` 的 14 轮教训）：

> **信号诚实性原则**：mock 可以模拟"世界"（一个会响应修复指令的生成器），
> 但评审器和指标必须读**产物内容**，绝不许读 revision 计数器。
> 回归卫兵 `test_loop_signal_is_content_derived`：一个无视修复指令的生成器
> 永远不收敛（缺陷持续、escape hatch 兜底、converged=False），
> 而诚实生成器收敛——证明循环是反馈系统，不是时钟。

---

## I1 · 统一技能生命周期（创作/评审/记忆三类技能 + "skill CI"准入）

### 来源依据

| 借鉴 | 内容 | 它的不足（我们要补的） |
|---|---|---|
| Voyager (2305.16291) | 技能=可执行代码库，执行成功即验证 | 验证依赖**廉价确定 oracle**（代码跑通/物品到手）；视频产出是感知评判的，没有这种 oracle |
| SkillWeaver (2504.07079) | 网页 agent 自主发现技能、反复练习打磨成 API | 练习预设环境**可重置、调用免费**；视频生成每次调用 GPU 数分钟且随机，练到鲁棒经济上不可行 |
| AWM (2409.07429) | 从轨迹归纳可复用 workflow 注入上下文 | workflow 是**文本动作痕迹**，无类型签名、无前后置条件、无失败处理；自评成功错了会固化坏 workflow |
| AutoSkill (2603.01145) | 终身技能自进化插件层 | **没有任何准入验证机制**——坏习惯被固化（我们"skill CI"的直接动机） |
| MemSkill (2602.02474) | 把记忆操作本身变成可进化的"记忆技能" | 只管文本记忆管理；进化信号来自有 ground-truth 的 QA 基准 |
| 综述 2512.16301 / 2604.08224 | 把 skills 与 memory/post-training 分立的外化能力 | 两篇都明确指出：**多模态生成管线的技能库是无人区** |

**核心空白**（survey_skills_audio §SYNTHESIS）：技能习得线与视频 agent 线
（UniVA 2511.08521 / CutClaw 2603.29664 / Crayotter 2606.07636 / MovieAgent 2503.07314）
截至 2026-06 **零交集**——所有视频 agent 都是静态手注册工具 + 角色 prompt。

### 实现原理（`memory/skill_library.py` + `memory/skill_admission.py`）

- **类型**：`Skill{skill_class: creation|review|memory, version, admission, physical_signature,
  triggers, entities, cinematography_preset, acceptance_thresholds, perf_score(EMA), …}`。
- **蒸馏（training-free）**：HSI 循环成功收敛后，`generate_loop` 把该 episode 固化为技能条目
  （prompt 触发词 + 物理签名 + 镜头预设 + 验收门槛）。技能名用
  `hashlib.md5(prompt)[:5]`（修复：`hash()` 进程加盐导致跨运行 id 不稳定、库里堆重复技能）。
- **准入"skill CI"**（`SkillAdmission`，三道门，全部推理期）：
  1. **证据门**：消费 episode 的真实输出——`converged`（真收敛，**escape-hatch 兜底的
     episode 被拒**，修复：之前被 hatch 丢弃的缺陷会冒充"已解决"进入证据）、
     `weighted_total ≥ 0.5`、`escalations = 0`（只有 Tier-0 干净解决的 episode 够格）；
  2. **回归门**：新技能的验收门槛不得低于库内同物理签名的现行最高门槛
     （修复：声明空门槛绕过该门的漏洞——缺省声明按缺失键拒绝）；
  3. **评审门**：MLLM 评审条目自洽性（触发词非空、签名与运动类别相容）。
- **三类技能落地**：创作技能 = 蒸馏所得；评审技能 = 物理验证三档位
  （measurement/world_model/vlm）注册条目，路由选择计入使用账（`mark_used`）；
  记忆技能 = 技能库自身的 EMA 保留策略作为可审计条目注册。
- **生命周期闭环已接线**：`record_outcome`（匹配技能用后按 episode 总分更新 EMA）与
  `age_and_evict`（每 run 结束）都有生产调用点（修复：之前是只有 API 没有调用的"声明式生命周期"）。
- 持久化：JSONL，原子写（`.tmp` + `os.replace`，修复 truncate-then-write 的崩溃丢库问题）。

### 可行性分析

- **mock 已验证**：生命周期全链（蒸馏→拒绝/通过→版本号递增→检索命中→EMA→淘汰）
  有 13 个专项测试；第二次同任务运行检索命中技能（`test_v03_skill_is_retrieved_on_second_run`）。
- **真实部署条件**：评审门换成真 MLLM（接口已是 `BaseMLLMClient` 注入式）；
  证据门消费的 weighted_total 来自真 VLM critic 后自动成立。无需任何训练。
- **诚实边界**：版本是计数器而非历史链（回滚需未来工作，文档已不夸大）；
  技能此刻参数化在 prompt/镜头预设层面，"以素材为参数的可执行工作流"
  （如 卡点蒙太奇(素材,音轨,情绪)）是下一步——需要先把工具层（UniVA 式 ToolRegistry，已有）
  的调用序列纳入蒸馏对象。
- **Reviewer 攻击面**：①"准入是不是又一个 LLM 自评闭环？"——防御：证据门消费的是
  metric/verifier 的客观 episode 输出，评审门只做格式自洽，真实像素下 metric 来自独立
  VLM+tracker；②"坏技能固化"——AutoSkill 的教训正是设计动机，三道门 + EMA 淘汰 +
  回归门是直接回应。

### Capability routing is a skill, not config（能力路由是技能，不是配置）

把"WHEN 调用 WHICH 生成能力（t2v/i2v/flf2v/edit）"建模为**三层**，关键论断是
第三层属于技能基质而非静态配置：

| 层 | 含义 | 落地 |
|---|---|---|
| **adapter** | 怎么调一个模型 | 后端方法（`models/video_gen_backends.py` 的 `generate`/`frame_to_frame`/`edit_video`） = **工具**（I5 工具脊柱） |
| **provider binding** | 哪个后端支撑某能力 | `models.video_gen.*` = **配置** |
| **capability routing** | 这一镜需要哪种能力 | `agents/capability_router.py` = **技能** |

- **为什么是技能**：选哪种能力依赖镜头意图与"相似镜头上次什么成功"——是程序性
  知识，不是常量。创作技能因此**记录**它的物理/视觉签名下成功的
  `gen_capability + gen_params`（`Skill.gen_capability/gen_params`，随 JSONL 持久化、
  缺键向后兼容默认 t2v）；下一相似镜头检索命中该技能即**复用**该路由决策
  （`CapabilityRouter.route` 路径 (a)："the skill decides which model"）。
- **冷启动启发式（诚实声明的 bootstrap）**：技能尚不存在时，
  `route` 走确定性意图启发式（路径 (b)）——源视频被标记编辑→edit；首尾关键帧齐备→flf2v；
  有身份/参考锚→i2v；否则 t2v。该启发式是 bootstrap，会被学到的技能**逐步替换**
  （真实部署可把它换成 LLM 导演，`route(...)->(capability,params)` 契约不变）。
- **诚实边界（绝不冒称能力）**：路由结果**只在后端 `capabilities()` 内取值**
  （mock 仅 {t2v,i2v}）；想要 edit/flf2v 但后端不支持时降级到 i2v（有锚）或 t2v 并
  **记录降级**（`route_capability` 轨迹事件 + GeneratorAgent 的 `capability_downgrade` 日志），
  没有静默的能力声明。
- **补的空白（vs UniVA 2511.08521）**：UniVA 每次运行都让 Act-LLM **重新临时决定**路由，
  决策即用即弃；Maestro 把验证收敛过的路由决策**蒸馏进技能**并复用——
  能力路由从每运行重算变成可积累、可审计的程序性知识。
- **mock 已验证**：13 个专项测试（`test_capability_router.py`）覆盖技能复用、四个启发式
  分支、降级永不越界（mock 后端 edit→t2v/i2v 且降级被报告）、GeneratorAgent 派发到正确
  WaveSpeed 方法（stub，无网络）/在 mock 后端回落、以及"蒸馏能力跨持久化往返并在第二次
  计划被复用"。端到端 mock 流水线全部路由到 t2v/i2v，既有行为不变。

---

## I2 · 参考自由的"像素物理验证器"（physics-from-pixels）

### 来源依据（为什么是这个形态）

演化路径（两次否定）：
1. **sketch-as-controller**（v0.1-0.2）：用户正确否定——没有现有模型能把合成草图当
  condition 有效控制冻结的视频生成器（文献核实：轨迹条件化全是 trained-only —
  PhysCtrl 2509.20358、Force Prompting 2505.19386；且轨迹欠定物理）。
2. **sketch-as-verifier**（v0.3）：对比"仿真预期轨迹 vs 观测轨迹"——仍预设仿真器知道
  质量/摩擦/恢复系数/尺度，单图不可知。Reviewer 必杀："你的仿真器才是错的"。
3. **v0.4 参考自由**：把问题改成**参数自由**的提法——
   > "观测到的运动是否存在**任何**物理一致的解释？"

| 借鉴 | 内容 | 不足 |
|---|---|---|
| Morpheus (2504.02918) | 守恒律残差不需要参考视频 | 只用于评测基准，从不驱动选择/重生成 |
| PISA (2503.09595) | 轨迹残差可作 reward | 对仿真 ground truth、单一现象 |
| 方程发现 (2507.06830) | 对观测轨迹拟合参数化动力学 | 用于预测，不用于验证 |
| WMReward (2601.10553) | 冻结世界模型当测试时物理 reward（ICCV'25 Physics-IQ 冠军） | **不透明标量**——说不出哪条定律、哪里、差多少；只能重掷不能定向修 |
| PSIVG (2603.06408) / PhyRPR (2601.09255) | 仿真入环生成 | **开环**——注入后从不验证生成器实际做了什么 |
| PhyT2V (2412.00596) | 生成→VLM描述→LLM推理→改prompt 闭环 | oracle 是有损文本瓶颈，执行器是弱杠杆 prompt |
| CoTracker3 (2410.11831) / SpatialTrackerV2 (2507.12462) | 观测轨迹提取 | 在真实视频上训练；**无人量化它们在生成视频上的可靠性** |
| TRAVL (2510.07550) / PhysBench (2501.16411) | VLM 物理评判可靠性证据 | VLM 普遍弱于动力学——是我们 vlm 档位的诚实上限依据 |

**站位**：「测量得到 + 可解释 + 按实体/帧段定位 + 同时驱动 best-of-N 选择与定向修复 +
training-free」的交集无人占据，且参数自由化后不再有"仿真器错了"的攻击面。

### 实现原理（`physics/`，六个模块）

```
annotate.py   实体 + 运动类别(ballistic/rigid/fluid/agentive/static) + 预期失效模式
   ↓ 只是验证种子——不含轨迹、不含控制信号
router.py     可验证性路由：每实体分配最强可行档位 measurement/world_model/vlm/none
   ↓ 覆盖率显式上报（部分验证绝不冒充全量验证）
tracks.py     观测轨迹提取（mock 确定性合成；CoTracker/TAPIR 在 backends 同约定）
   ↓           实体种子=检测定中心：GroundingDINO 在 frame0 检测每个实体→bbox 质心→
   ↓           CoTracker 在该质心播种（跟踪"真正的那个实体"，非任意像素；
   ↓           检测不到→回退均匀播种 + WARNING：该实体判据不可靠）
reliability.py  先认证 tracker 再相信 verdict（S2）：
   ↓             完整性/抖动(churn>0.55 拒)/跨tracker一致性；不可读=clip_unreadable
laws.py       核心："是否存在任何物理解释"
   ↓           ① 被动定律族拟合 static/匀速/匀加速（重力向量自由拟合→无需尺度标定），
   ↓              最简定律优先（更复杂定律须好 20% 才换），最佳拟合残差=违例度
   ↓           ② 离散异常定位：teleport→物体恒存 / mid-air reversal→重力惯性 /
   ↓              energy_gain→守恒 / jerk_spike→碰撞，全部带帧段
verifier.py   组装 + 失信降级（认证失败的 measurement 实体降到 vlm 档，覆盖率按最终档重建）
   → critics/physics_consistency.py：按实体/帧段/模式的 verdict（source="law_verifier"）
   → 驱动 best-of-N、HSI 定向修复、p2_law_consistency 独立计分（与 VLM 评审 p1 分离）
```

物理正确性的关键细节（审查修复后）：
- 自由落体加速**不是**守恒违例（重力做功）——energy_gain 仅在无定律可解释时计入；
- teleport/jerk 检测的 median 归一在"最干净的违例"上退化为 0（静止物体瞬移）——
  加绝对阈值兜底（ABS_TELEPORT=0.15 屏幕单位/帧）；
- teleport 严重度按双触发约定分级（触发点 0.5，双倍饱和 1.0），不再恒为 1；
- <4 帧轨迹标 `indeterminate` 而非"static=可解释"；
- mid-air reversal 的单地平面假设与两类边界（全局最深点处的反转不可与弹跳区分→漏检；
  高台弹跳→误检）**写进文档**而非假装解决；
- 残差高但无异常定位时用 `UNEXPLAINED` 模式（不冒充"重力违例"分类）。

### 可行性分析

- **mock 已验证**：违例轨迹（含 tracker 式噪声底）被定位检出（severity≈0.5、帧段非全片），
  应用物理修复指令后的再生成通过；负对照（不带修复指令的再生成）保持被标记。
- **真实部署条件**：CoTracker3（torch.hub 一行加载，已接）或 TAPIR；
  **实体种子点已改为检测定中心**（GroundingDINO，`models/detection_backends.py`，
  HF transformers 零样本检测）：每个 prompt 实体在第 0 帧被检测→取 bbox 质心→
  CoTracker 在该质心播种，从而跟踪"真正的那个实体"而非任意像素——
  数据流 `prompt 实体名 → detect(frame0) → 质心 → CoTracker → 定律`。
  检测不到时回退到旧的均匀带状位置并 WARNING（该实体判据不可靠），ABC 契约不变。
  world_model 档位挂 V-JEPA 2 reward（`models/world_reward.py` 接口已留）。
- **已知边界（诚实声明）**：
  1. 覆盖率：measurement 档只覆盖刚体/抛体——这正是路由器存在的理由，缺口显式交给
     world_model/vlm 档并上报，**部分验证透明化本身是贡献**（对比 PSIVG 假装全覆盖）；
  2. tracker 在生成视频上会"对变形物体输出貌似合理的轨迹"——可靠性门控 + 跨 tracker
     分歧（本身就是不合理线索）是直接回应，量化研究（G4）可独立成文；
  2b. 检测播种的剩余边界=**检测质量**：GroundingDINO 在真实图像上训练，对生成视频的
     零样本检测可能漏检/误定位，且需 GPU+权重；检测不到时回退均匀播种且该实体判据被
     标记为不可靠（诚实声明）。MockDetector（仅按名词出 bbox、忽略像素）是 CPU 冷启动路径，
     此路径下播种本质仍是启发式——单实体判据在真实视频上不可靠；
  3. 随机游走式漂移噪声仍可能通过抖动门（文档已注明，平滑性检验是后续工作）；
  4. p2=1.0 的语义是"无测量违例"而非"已验证"——提取失败时 WARNING 日志 + 覆盖率记录，
     不会无声扮演完美分数；
  5. 收益上界=基础生成器的样本多样性（N 个里没有一个物理正确时选择无能为力）——
     定向重生成（verdict→修复指令→再生成）是答案，已是 Tier-1 的实现语义。
- **评测反循环协议**（计划，D8）：选择用自家 oracle、评测用独立轴
  （VideoPhy-2 AutoEval 2503.06800、Physics-IQ 协议 2501.09038、人评），
  oracle 内置反作弊（慢动作/静态场景会平凡"守恒"→ dynamism 下限正则）。

---

## I3 · 双寄存器实体记忆 + 验证门控写入

### 来源依据

| 借鉴 | 内容 | 不足 |
|---|---|---|
| EntityMem (2605.15199) | 生成前验证每实体参考并持久复用；EntityBench 长间隔基准 | 参考**冻结**——无法表达剧情驱动的外观演化（湿了/受伤/换装），生成后不再验证 |
| VideoMemory (2601.03655) | 实体描述库逐镜头更新 | 更新是**未经像素验证的 LLM 描述**——错误写进记忆向后复利传播 |
| StoryMem (2512.19539) | 关键帧记忆库注入单镜头扩散模型 | 关键帧无差别吞下生成内容（无写入质量门）；外观级，无状态语义 |
| A-MEM (2502.12110) / MIRIX (2507.07957) | 链接式笔记进化 / 六类型存储 | 文本域；"多模态"=截图转描述，无法支撑视觉再 grounding |
| 记忆综述 (2512.13564) | 明确把"多模态+生成侧记忆"列为开放前沿 | —— |

**两个结构性空白**（survey_memory §5）：①无"感知↔生成"闭环读写——记忆存的是
"打算生成什么"，从不是"输出里验证到什么"；②无身份/状态分离——冻结派不能演化，
自由更新派会漂移，没人做 canonical identity ⊕ evolving state + 类型化转移日志。

### 实现原理（`memory/entity_store.py` + `memory/write_gate.py`）

- **双寄存器**：`EntityIdentity`（frozen dataclass：id、名字、验证过的参考图路径、描述
  ——注册后无变更 API；注册期允许对**空**字段做一次性回填）⊕ `EntityState`
  （attributes dict + version），状态**只能**通过 `StateTransition`
  （entity/shot/field/old/new/cause/status/evidence/run_id）进入。
- **状态由日志重放**：加载时从 append-only 转移日志重放出当前状态——每个状态版本
  可溯源到一条具体转移（审计完整性测试钉死）；修正（correction）也是显式日志条目。
- **验证门控**（"只提交渲染中验证到的"）：转移在规划期提出（mock 为确定性线索词导演
  替身，文档言明），**只有** 镜头 `converged`（非 escape-hatch 兜底）、产物证据含新值、
  且无一致性/物理缺陷被 hatch 跳过时才 commit；否则 rejected 并记录依据。
  审查修复：证据只读 clip 产物（不再把 spec.prompt 混进证据造成同义反复）；
  mock 下产物仍回显 prompt 的残余循环性**在 docstring 里全文声明**，真实路径=注入
  MLLM 看解码帧，同契约。
- **跨运行作用域**：转移带 `run_id`，去重与待提交选择按本 run 作用域（run 1 被拒的转移
  run 2 可重提）；状态本体跨运行持久（复现角色是产品特性），泛指兜底实体 "subject"
  排除在注册之外。
- **再入条件**：`reentry_context(entity_id)` = 身份参考 + 当前已验证状态 →
  长间隔后续镜头的条件化负载（EntityBench 式评测的直接接口）。

### 可行性分析

- **mock 已验证**：13 个专项测试覆盖 提出→门控提交/拒绝×3路径→修正→重放审计→
  长间隔再入→持久化往返→旧版兼容；流水线级测试覆盖未收敛=全拒。
- **真实部署条件**：写入门的感知确认换成真 MLLM 看解码帧（注入点已留）；
  身份寄存器的参考图来自素材链路（identity anchors 已接，注册顺序修复后参考路径不再丢失）；
  ConsisID 式频域 ID 特征作为紧凑身份键是升级项（接口预留 description/embedding 位）。
- **诚实边界**：mock 写入门的判别力受限于文本产物（已声明）；真正的鉴别力测试
  （生成的镜头里角色确实"湿了"吗）只能在真像素+真 VLM 下做——这正是 D7 最小真实链
  要验证的第一批东西。
- **Reviewer 攻击面**：①"门控写入会不会卡死更新（全拒）？"——修正条目（correction）
  是显式出口：渲染与提案矛盾时记录矛盾而非沉默丢弃；②"LLM 提案错误怎么办？"——
  提案不可信正是设计公理，门控+日志的全部意义。

---

## I4 · 层级自改进 HSI + 教训库（C2/C4/C5，v0.4 修正版）

### 来源依据

- VISTA (2510.15831)：测试时多 agent 闭环，但**只改 prompt、整段重生成、会话间失忆**；
- VideoRepair (2411.15115)：空间局部修复，无时间局部、单轮；
- MAViS (2508.08487)：逐阶段局部审查，**无跨阶段归因**；
- MemoGen (2606.03243，图像)：经验记忆写回——视频域无人做（gap G7）；
- M3 (2602.06166)：关键帧局部编辑。

### 实现原理（`pipeline/generate_loop.py`）

- Tier 0 关键帧局部修复（最便宜）→ **Tier 1 verdict 驱动的物理修复再生成**
  → Tier 2 导演级 spec 重写 → Tier 3 escape hatch（显式记账的妥协）。
  每层 Verifier 强制单调改进，接受后回落 Tier 0（成本摊销的自适应作用域）。
- **审查修复的关键语义错误**：旧 Tier-1 = 收紧验证门槛（strictness）——对失败镜头收紧
  门槛使接受**更难**，自败（审查代理用执行证明：同质量候选 p2 0.768<1.0 → Verifier 拒）。
  新 Tier-1 = 把最严重 verdict 的定位信息（实体/帧段/模式）转成修复指令在**不变门槛**下
  再生成；strictness 收紧改为可选的**接受后审计**（只记录不否决，
  `physics.post_accept_strictness`）——收紧失败者是自败，收紧通过者是质量哨兵。
- **escape-hatch 记账**（修复）：被跳过的缺陷以类型化 `skipped_modes` 全程携带——
  教训库不再把 hatch 掉的模式记成"已解决"，技能准入拒绝 hatch episode，
  实体写入门拒绝 hatch 渲染。`accepted` 的语义文档化为"我们交付的 clip"≠"无缺陷"。
- 教训库（C4）：从**真正被解决**的失效模式蒸馏，A-MEM 式双向链接，注入后续同类 prompt；
  注入的教训 id 现在真实耦合进蒸馏技能（修复：之前读不存在的字段，恒为空）。

#### Tier-1 修复是多动作、verdict 路由的（Review→Execute 桥）

UniVA 把"审查 verdict → 修复"交给每次重跑都临时重决策的 Act-LLM，且实际只会
re-prompt/重生成。Maestro 把同一座桥做成**确定性、training-free 的路由决策**
（`agents/repair_router.py` 的 `RepairRouter.choose`），从**最严重 verdict** 选出一个
工具支撑的修复动作，受后端能力 + 上传素材门控；每个动作仍走 `board.review` +
`verifier.is_better`，单调改进契约不变。

| 审查 verdict | 路由动作 | 工具（后端方法） | 门控条件 |
| --- | --- | --- | --- |
| 物理运动类（gravity_inertia / collision / conservation / penetration） | `edit_clip` | `video_gen.edit_video()`（runway gen4-aleph） | `"edit" ∈ capabilities`（更便宜、保留好的部分，免整段重 roll） |
| 语义"缺失元素"（失败的 semantic 清单项） | `retrieve_replace` | `retrieval.retrieve_source_shots()` → 真实上传素材 | `AssetMemory` 有 source shots |
| object_permanence / "incomplete" / "too short" | `extend_clip` | `video_gen.extend()`（末帧 i2v 续接） | `"extend" ∈ capabilities` |
| 关键帧级局部缺陷（Tier-0） | `keyframe_edit` | `image_edit.edit()` + first_frame 锚 | `image_edit` 后端可用 |
| 兜底（任何工具/素材不可用时） | `regenerate_hint` | `physics_planner.replan` + `generator.run`（verdict 提示） | 总是可用 |

工具脊柱因此**超过 UniVA**：edit / extend / retrieve / keyframe-edit / regenerate
五种动作，外加 UniVA 没有的**参考自由像素物理检测**（I2，正是 `edit_clip` /
`extend_clip` 的 verdict 来源）。诚实降级：任一被选动作的能力/素材缺失时，路由器
**绝不**返回它，而是回落到 `regenerate_hint`——所以 mock 管线（caps={t2v,i2v}、无 source
shots）`choose` 恒返回 `regenerate_hint`，既有行为逐字保持。注：真实部署可在同一
`choose` 签名后换上 LLM 修复规划器，路由形态不变。

#### Brain 编排的修复（真正的 agent 循环，v0.4 新增）

确定性 `RepairRouter` 只从【最严重 verdict】路由一个动作；它仍是 if-else 决策。
v0.4 在它之上放一个真正会 **function-calling 的 LLM 编排器（brain）**
（`agents/orchestrator.py::OrchestratorAgent`），把"读评审 → 决定调哪个工具"
交还给一个 LLM——但用两条硬轨把它**接地（grounded）**，这正是 UniVA / NEWTON
的 Act-LLM 所缺的：

- **接地于测量评审**：`decide` 把结构化评审（语义失败项的 question/fix_instruction/
  kind + 物理 verdict 的 mode/severity/frame_range/intervention/source + metric 分数）
  连同**工具菜单**和**动作历史**序列化进 prompt，brain 读的是**证据**不是口号。
  UniVA 的 Act-LLM 只看到一个文本目标，看不到测量评审。
- **接地于单调闸门**（"brain 提议，闸门裁决"）：brain 选的任何工具，结果都必须过
  `verifier.is_better` 才落地；被拒的动作回灌进 `history`，下一回合 brain 不许重复
  （prompt 明示）。UniVA 那边没有任何东西会拒绝一次回退。

工具注册表（`available_actions`，按真实能力 + 素材门控生成菜单）：
`regenerate` / `keyframe_edit` / `accept` 恒在；`edit_clip`、`extend_clip` 仅当后端
`capabilities()` 声明了对应能力时出现；`retrieve_replace` 仅当 `AssetMemory` 有
source shots 时出现。`decide` 用与真实 VLM critic 同一个 `_extract_json` 解析 brain
的严格 JSON 回复（provider 无关，零 SDK 依赖，走 `BaseLLMClient.complete`）。

**安全网降级**：brain 回复不可解析 / 越界工具 / 工具能力缺失时，`decide` 返回
`INVALID` 哨兵，`generate_shot_orchestrated` 回落到确定性 `RepairRouter` 执行**一个**
动作——所以环路永不卡死，且 brain 抽风时退化为既有的确定性修复。

可选：`compose.repair_mode: "hsi"`（默认，HSI 阶梯不变）|`"orchestrator"`
（brain 环路）。默认保持 `"hsi"`，既有全部测试逐字不变。诚实声明：这是 inference-time
的 function-calling 编排（training-free），brain 的"工具选择质量"取决于底层 LLM；
我们不主张它一定优于阶梯，只主张它把修复从**固定 if-else** 变成**接地的工具调用**，
且单调契约与降级安全网保证它不会比阶梯更差地卡死或回退。

### 可行性分析

- mock 验证：内容驱动收敛（修复指令未应用→不收敛）；tier-1 在不变门槛下接受修复
  （专项测试）；跨层 prompt 突变不再污染重标注（base_prompt 快照）。
- brain 环路验证（`tests/unit/test_orchestrator.py`，CPU、无 torch/网络）：StubBrainLLM
  返回 canned JSON——菜单门控正确、合法 JSON 被 `decide` 验证通过、垃圾回复→`INVALID`；
  `execute` 正确路由 edit_clip/retrieve_replace/accept；整环用 mock 评审收敛、决策轨迹
  被记录、单调闸门至少裁决一次；brain 抽风时回落 RepairRouter 仍终止不崩。
- 真实部署：层级结构与真实后端无耦合——Tier 0 需要 image_edit 后端（接口已留），
  Tier 1/2 即刻可用（prompt 级杠杆）。
- 边界：跨阶段归因（G8：坏成片该怪剧本还是生成器）只到"镜头内 tier"粒度，
  剧本级归因是计划中的扩展（INNOVATION_PLAN §3.1）。

---

## I5 · 配套工程（不主张为论文创新点，但支撑可行性）

- **真实链路**：WaveSpeedClient 完整实现（UniVA 同款云 API：提交→轮询→下载），
  `$WAVESPEED_API_KEY` 即出真像素无需本地 GPU——D7 最小真实链最快路径；
  OmniWeaving/Wan 本地骨架同契约。物理是验证出来的不是注入的，
  所以纯文本 API 后端也成立（这是 v0.4 重定位带来的工程红利）。
- **素材链路**：AssetMemory（identity anchors/style refs/music profile）+ RetrievalTool
  + UniVA 式 ToolRegistry/ActAgent；与父项目（LongVideoEditAgent）的检索+生成混合
  前提合并（D1/D2）。
- **音频**：按 D6 占位（audio_gen 工具接口 + mock）。实做路线已定
  （survey_skills_audio）：后置多模型路由（MMAudio 2412.15322 / ThinkSound 2506.21448
  的指令编辑级=评审环执行器 / V2M-Zero 2603.11042 管时序 + agent 写语义 brief），
  AV 一致性评审环与跨镜头声学连续性是无人占据的 agent 级贡献；
  注意 AV-Align 对音乐无效（节拍≠运动）。
- **评测计划**（D8）：EntityBench（长间隔实体一致性）、VideoPhy-2、Physics-IQ 协议、
  UniVA-Bench、AV-Align/JavisBench + 小规模人评；反循环协议（独立评测轴+反作弊正则）。

---

## 6. 审查-修复记录（本文声明可信的依据）

三个独立审查代理（物理数学、技能/记忆门控、流水线语义+信号溯源）共 42 项发现，
按严重度全部处置（29 项代码修复 + mock 信号内容化改造 + 文档化的诚实边界）：

- 致命语义类：Tier-1 自败反转、escape-hatch 缺陷冒充已解决、写入门同义反复、
  skill_id 进程加盐、回归门空声明绕过、PhysicsCritic 覆盖 verdict 列表的顺序脆弱性；
- 数学类：teleport 严重度退化恒 1.0、median 归一在最干净违例上失效、
  自由落体误判守恒违例、静-静双 tracker 比较除以 MIN_RANGE 爆炸；
- 元错误类（父项目教训的复发）：6 处 revision 计数驱动的 mock 信号全部改为
  产物内容驱动，并加信号诚实性回归卫兵；
- 文档类：所有"声明了但没实现"（降级路由、原子写、版本可回滚）或改实现或改声明。

每项修复都带回归测试；192 个测试全绿；mock 端到端收敛由内容驱动。
**仍开放的已声明边界**：tracker 漂移噪声的平滑性检验、剧本级跨阶段归因、
技能的素材参数化工作流形态、音频实做、真实像素下的全链验证（D7，等服务器）。
