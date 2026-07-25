# OpenMontage 深度解读与借鉴评估(2026-07-25)

> 用户令:精读 https://github.com/calesthio/OpenMontage,总结核心与创新点,
> 写详细文档,评估是否借鉴。方法:六路并行读者按子系统精读(架构/执行核心/
> 工具层/skill 层/知识库/渲染与测试)+ 创新鉴定 + Maestro 逐维对比,全部
> 结论带文件路径,营销口径逐条对过码。本地副本:
> `/Users/kevin/Desktop/Kevin/repositories/OpenMontage`。

## 〇、一句话定位

OpenMontage 是一个**把"AI 编码助手"本身当成视频制作总导演**的生产系统:
仓库里没有运行时编排器、不调任何 LLM API(`tools/` 全目录 grep anthropic
零命中)——你在 IDE 里跑的 Claude Code / Cursor / Codex 就是大脑,Python
代码只干两件事:提供工具、守住不变量。AGPLv3,42k 星,2026-03 创建,
活跃维护。**它和 Maestro 是两种物种**:它是宽而浅的人工把关"制片厂"
(IDE copilot 形态),Maestro 是窄而深的自主生成"机器"。

数字校准(营销 vs 实测):"12 条生产管线" = 13 个 YAML 减 1 个冒烟测试,
其中仅 6 个 `stability: production`,且没有任何一张文档表格与
`pipeline_defs/` 完全一致;"100+ 工具" = 144 个 .py 里 102 个声明了
capability(基本属实);"700+ 知识文件" = `skills/` 156 + `.agents/skills/`
567,属实。

## 一、核心架构:倒置的编排(它最值得看懂的一件事)

常规 agent 框架(包括 Maestro):**编排在代码里,判断交给 LLM**。
OpenMontage 完全反过来:**编排在 LLM 手里,代码只在关键缝隙上钉钉子**。

白话讲:它相信"大脑会走流程",但不相信"大脑不越界"。于是所有"不许
越界"的事都做成确定性代码,而且**失败是硬失败**:

1. **人工闸门在状态写入点强制执行**(`lib/checkpoint.py:378-405`):
   凡是清单标了 `human_approval_default: true` 的阶段,想把检查点写成
   "completed" 却没有 `human_approved=True` → 直接抛异常 GATE VIOLATION。
   闸门开关从清单本身读取(不由调用方传参,所以绕不过);pipeline 类型
   写错 → **fail closed**(注释原话:"a typo must not silently disable
   gate enforcement");漏传参数会从 project.json 回填。检查点原子写入
   (`os.replace`),被覆盖的状态归档到 `history/` 供回放。
2. **产物 schema 严格校验**:20 个 artifact JSON Schema 全部
   `additionalProperties: false`,读写检查点两头都校验
   (`lib/checkpoint.validate_checkpoint`)。
3. **每阶段一个"director skill"**:`skills/pipelines/<管线>/<阶段>-director.md`,
   agent 进阶段前必读;12 管线 × 7-10 阶段共 103 份。编排大脑本身也是
   一份 skill(`executive-producer.md`,423 行,自称 "You are the
   stateful brain; the directors are stateless workers",带 EP_STATE
   伪 schema、PASS/REVISE/SEND_BACK 三态门、修订上限 3 次后
   "PASS WITH WARNINGS (never block forever)")。

### 微模式:把报错文本当 prompt 写(便宜又高明)

它的异常信息是**写给将要读它的 LLM 看的**——GATE VIOLATION 的报错原文
直接把正确恢复流程写全:"write status='awaiting_human', present the
artifact summary to the user, END YOUR TURN, and only after the user
approves re-write with status='completed', human_approved=True."
工具失败同理(error 里拼上 `install_instructions`)。stderr 是一条
prompt 通道——这个微模式几乎零成本可移植。

## 二、三层知识体系(它的招牌,也是与我们 skill 路线的正面对照)

```
Layer 1  tools/ + pipeline_defs/     "存在什么"(注册表,机器可查)
Layer 2  skills/(156 份)            "OpenMontage 怎么用它们"(打法)
Layer 3  .agents/skills/(83 包 983 文件)"技术本身怎么回事"(供应商级知识)
```

- **桥是机器可读的**:每个工具类声明 `agent_skills: ["ai-video-gen"]`
  (`tools/base_tool.py:281`,93 个工具声明了)指向 Layer 3 知识包;
  AGENT_GUIDE 规定"调任何生成工具前必读其 Layer 3 skill,不可选"。
- **渐进披露是明文设计规则**:路由 skill 只带能力地图,内层
  rules-index("36 条原子配方,XML 式条目带标签")按需加载;
  "Do not read it speculatively";`media-use` 的设计原话:"The agent
  gets back one line. Candidates, scores, provenance stay on disk"
  ——上下文经济学被当作硬规则。
- **知识包不只是 md**:带 `references/` 深读文件(avatar-video 15 份)、
  可执行脚本(librosa 节拍分析、对比度报告)、子代理 prompt 文件
  (motion-graphics 的 director/builder/finalize 三角色)、75 个可运行
  Manim 场景、字体/贴图二进制、来源与许可证档案(PROVENANCE.md)。
- **知识密度实例**(全部原文可查):Seedance 提示词要以镜头结构声明
  开头 + "2.5 秒一镜的节奏最优";缓动即情绪("expo.out = 自信,
  sine.inOut = 梦幻;进场用 .out、退场用 .in——你总是搞反");
  音效 whoosh 提前 10-20ms("大脑处理声音更快");"cinematic/epic 这类
  情绪形容词被禁用",必须替换成像素级约束(2.39:1 黑边、24fps、8s+
  镜长);相机原语消歧表("zoom in ≠ dolly in")。
- **但装载是君子协定**:没有任何 Python 把 skill 文本注入 prompt,
  全靠 agent 自觉读;且已观察到漂移——`.claude/skills/` 是**过时的
  复制品**(文档声称 symlink,实测 48/83 且缺整个 HyperFrames/GSAP
  家族),`skills/INDEX.md` 声称 47 个包,实际 83。

## 三、工具层与路由

- **统一契约**:全部继承 `BaseTool`,`execute(dict) -> ToolResult
  {success, data, artifacts, error, cost_usd, seed, model}`;元数据极全
  (tier/capability/provider/runtime/稳定度/依赖/best_for/not_good_for/
  fallback_tools/重试策略/幂等键/副作用声明/人工核验清单)。
- **注册表自发现**:`pkgutil.walk_packages` 扫描实例化;加一个 TTS
  供应商 = 只写一个文件,选择器零改动。
- **可解释路由**:`lib/scoring.py` 七维加权(task_fit .30 / quality .20 /
  control .15 / reliability .15 / cost .10 / latency .05 / continuity .05),
  每个选择带 `.explain()`、`selection_reason`、`alternatives_considered`;
  preferred_provider 只在分差窗口内被尊重。
- **预算治理**(`tools/cost_tracker.py`):estimate → reserve →
  reconcile 三段账,observe/warn/cap 三模式,单动作超 $0.50 抛
  ApprovalRequiredError,付费工具首用要批准,10% 预留金,重试预留 1.3×。
- **诚实缺口**(实测):`retry_policy`/`input_schema`/`fallback_tools`/
  幂等键是**声明给 agent 看的元数据,运行时无消费者**;成功时的
  `cost_usd` 是把估价重报为实付。供应商覆盖面极广(22 个视频生成、
  13 图、8 TTS、16 个免费素材源适配器——其中几个是爬虫,法务面注意)。

## 四、渲染路线(它对我们最"降维"的一块)

三引擎:**Remotion**(React 数据驱动合成,默认)/ **HyperFrames**
(HTML/CSS/GSAP)/ **FFmpeg**(仅剪拼)。治理三件套:
1. `render_runtime` 在 proposal 阶段锁定,**静默换引擎被禁止**——引擎
   不可用就返回结构化 blocker 等人来,绝不偷偷替换;
2. `RENDERER_FAMILY_MAP` 按管线钉合成家族,注释原话:"防止所有管线
   坍缩进 Explainer 的视觉语法";
3. **atelier(手作)防火墙**:bespoke 模式下用正则**阻止 import 库存
   组件**(`video_compose.py:975`)——"bespoke means bespoke" 用机器
   执行,防模板陷阱。
另有 ink-theater:确定性手绘矢量动画引擎(种子 PRNG、无 Math.random、
闭式弹簧缓动、每帧可由时间重建),mocap 契约规定 "agent 只许选角色和
编排具名动作,**永远不许手调运动曲线**"——一条设计得很好的 LLM 授权
边界。字幕词级烧录、主题/playbook 系统带 WCAG 对比度与色盲混淆对校验
(`styles/playbook_loader.py`,829 行确定性设计智能)。

## 五、质量体系:没有 LLM 判官的质量工程

这是它最独特的一面:**全库测试 grep 不到任何 LLM-as-judge**。质量靠:

1. **确定性创意 lint**:`lib/slideshow_risk.py`(六维打分,verdict
   fail 阻止进入合成)、`delivery_promise.py`(承诺类型锁定——
   motion_led 要求真运动占比 ≥0.7,并明文 "Remotion 组件场景是'会动的
   幻灯片',不算真运动")、`variation_checker.py`(约 20 个空话短语
   黑名单)。**创意品味被转译成可回归测试的不变量**:
   `tests/eval/bench_runner.py` 拿已知好/坏计划过校验器断言判决。
2. **prompt 契约测试**(真创新):
   `tests/contracts/test_runtime_presentation_contract.py` 在 CI 里断言
   "每条管线的规划 skill 必须含运行时选择指引"——注释原话:新会话的
   agent 读不到指引就会静默默认 Remotion,"这正是本契约要防的失效模式"。
   **把 prompt 当接口做 CI 测试**。
3. **自审是建议制**:`skills/meta/reviewer.md`(CHAI 三律:批评必须
   Accurate/Complete/Constructive,最多两轮)——同一个 agent 给自己
   打分,这是它相对我们最弱的一环。

## 六、创新点鉴定(真创新 / 优秀工程 / 营销)

| # | 条目 | 判定 |
|---|---|---|
| 1 | 倒置编排:agent 即控制平面 + 代码只守不变量(fail-closed 闸门) | **真创新**(作为完整架构立场) |
| 2 | 报错文本写给 LLM 读(恢复协议进异常信息) | **真创新**(微模式,零成本可偷) |
| 3 | 确定性创意 lint + 无 LLM 判官的质量回归 | **真创新**(最可移植的思想) |
| 4 | prompt 契约测试(skill 文本当接口进 CI) | **真创新** |
| 5 | 三层知识 + agent_skills 机器桥 + 渐进披露 | 优秀工程(内容密度是真价值) |
| 6 | 七维可解释供应商路由 | 优秀工程 |
| 7 | 运行时锁定 + atelier import 防火墙 + ink-theater 授权边界 | 真创新/优秀工程之间 |
| 8 | 零配置观测(`__init_subclass__` 自动埋点)+ Backlot 只读回放板 | 优秀工程 |
| 9 | "12 管线 / 100+ 工具 / 首个开源" 等数字与定位 | 营销(逐条见上文校准) |

## 七、诚实弱点(借鉴前必须看清)

- **没有学习闭环**:`historical_success_rate`/`quality_score` 字段存在
  但无人写入;评审结果不回流路由;每次生产从零开始。这正是 Maestro
  的主场(episode memory / skill 蒸馏 / 正在做的 RL)。
- **执行不对称**:只有闸门和 schema 是代码执行;修订上限、重试、
  fallback、"必读 Layer 3"全是散文约束——弱模型或新会话可以静默跳过。
- **单次成本无界且不可测**:每支视频都要一只前沿编码 agent 吃完 713 行
  指南 + 各阶段 skill + Layer 3 包,过 5-8 道人工闸;无结果缓存。
- **文档漂移严重**:管线表两两不一致、`docs/stage-gates/` 是空目录、
  `.claude/skills/` 过时缺 35 包、检查点路径三处口径不一。
- **CI 薄**:lint = 4 个文件 py_compile;golden 回放架子 fixtures 为空;
  QA 脚本 01-03 文档提及但不存在。
- **AGPLv3**:抄代码有传染义务;抄思想自由。

## 八、与 Maestro 逐维对比

| 维度 | OpenMontage | Maestro | 谁强 |
|---|---|---|---|
| 编排 | agent 即编排器,人工闸门代码执行 | Python 控制平面,三层决策兜底(episode→LLM→确定性),stop_reason 全谱 | 各自问题域各强;自主生成场景 **Maestro** |
| skill 体系 | ~723 份,内容极富,装载靠自觉,已见漂移 | 11 份,代码注入 prompt(装载有证据),逐行审计过 | 绑定 **Maestro** / 内容 **OM** |
| 工具层 | 144 工具,统一契约+自发现+七维路由;契约多为声明 | 单后端,规范映射表+载荷硬闸,一出口全记录 | 广度 **OM** / 载荷正确性 **Maestro** |
| 评审验收 | 自审建议制 + 计划级确定性 lint;零 LLM 判官 | 评审/汇总/验收三权分立,原生视频审,盲测 A/B 硬闸 | **Maestro**(决定性) |
| 记忆与自提升 | decision_log 有痕无学习 | 台账 + episode replay/avoid + skill 蒸馏 + RL 管道 | **Maestro**(决定性) |
| 渲染合成 | 三引擎 + 治理 + 排版/图表/字幕/主题 | 仅 ffmpeg concat | **OM**(决定性) |
| 成本治理 | 三段账 + 三模式 + 单动作审批线 | 只记调用,无美元账 | **OM** |
| 跨镜像素连续性 | 无(独立片段拼装) | extend/pin 分类、免级联段修复、交接棒 | **Maestro**(它根本没有此问题域) |

## 九、借鉴清单(按 价值/成本 排序,均为机制级)

1. **skill↔代码契约测试**(仿 test_runtime_presentation_contract 思想):
   断言 `_CONDITION_PRIORITY` 的每个策略、orchestrator 菜单的每个动作
   都出现在对应 SKILL.md;退役名(tiv2v_window、multi_image_fusion、
   keyframe_edit…)全库 skill 零出现。落点
   `tests/unit/test_skill_contracts.py`。7-17 那次人工审计的 19 条
   HIGH/MEDIUM 漂移,以后 CI 自动抓。工作量:低(约一天)。
2. **预算三段账**(仿 cost_tracker):挂进 `WaveSpeedClient._run_task`
   这个既有单一出口;cost_snapshot 进 storyboard;预算耗尽成为新的
   stop_reason。修复循环的开销从此有帽。工作量:低-中。
3. **剧本级确定性 lint + 已知好坏基准**(仿 slideshow_risk/
   variation_checker/bench_runner):对 scene_write 产物做静态六维检查
   (空话短语、镜头密度、时长-动作密度……部分已有雏形:时长-密度法则
   目前只在 skill 里),**在花钱生成前杀掉坏剧本**;配一个
   known-good/known-bad 剧本基准回归。落点扩展 `agents/plan_validator.py`。
   工作量:中。
4. **CLIP 素材检索**(仿 lib/corpus.py + clip_embedder.py 的"纯基建、
   零判断"设计):我们缺口台账里明列的升级项,现在是词袋哈希。落点
   `tools/retrieval_tool.py`,标签链降级不变。工作量:中。
5. **只读观察板**(仿 Backlot:watchfiles+SSE、状态纯从磁盘推导、
   "observer never blocker"):我们三件套日志已齐,render_brain_log 是
   静态前身;做成 `Maestro/board/` 实时看长任务。工作量:中。
6. **可选人工闸门**(仿 checkpoint 的 awaiting_human 语义,fail-closed):
   在 scene_write 和 image-plan 后加**可选**闸(最高杠杆、花钱之前的
   两个点)。需要暂停/续跑缝(台账原子写已支持 resume)。工作量:中。
7. **提示词骨架增补**(仿 video-gen-prompting 的五槽骨架 Subject/
   Motion/Scene/Spatial/Camera + 相机原语消歧表 + 逐模型长度甜点):
   **只并入无锚路线**——attempt3 的瘦身法则证明锚定路线嫌话多,
   OM 骨架恰好适用于 prompt 是唯一载体的 t2v/换场。落点
   prompt_enhancer SKILL。工作量:极低。

另收两个微模式,随手可用:**报错文本按"给 LLM 的恢复协议"来写**
(我们的 GATE 类日志与异常可以照此改写);**上下文经济学明文化**
("工具返回一行,细节留磁盘"——我们的 media catalog/检索返回可对照)。

## 十、不该学的(反向 cargo-cult 警告)

- **自审模式**:同一个 agent 给自己打分,正是我们三权分立+盲测 A/B
  特意消灭的偏差;
- **散文编排**:修订上限靠 skill 记忆——我们的三层决策兜底和菜单闸是
  为了消灭这种"大脑跑丢没人接"的形态;
- **skill 装载靠自觉**:我们 7-15 的"百分百确认装载"(skill_chars
  逐调用留证)就是冲着这个失效模式建的;
- **BaseTool/注册表整套照搬**:我们单后端 + 规范映射表 + 载荷硬闸对
  载荷正确性更安全;供应商广度不是我们现阶段的问题;
- **Remotion 全家桶**:体量巨大,只有当产品需要排版/图表/字幕合成时
  再议(那将是一个独立立项,不是借鉴)。

## 附:关键证据路径速查

倒置编排与闸门:`AGENT_GUIDE.md`(713 行总纲)、`lib/checkpoint.py`
(378-405 闸门)、`pipeline_defs/*.yaml`;三层知识:`skills/INDEX.md`、
`tools/base_tool.py:281`(agent_skills)、`.agents/skills/*/SKILL.md`;
质量:`lib/{slideshow_risk,delivery_promise,variation_checker}.py`、
`tests/eval/bench_runner.py`、`tests/contracts/
test_runtime_presentation_contract.py`;路由与预算:`lib/scoring.py`、
`tools/cost_tracker.py`;渲染治理:`tools/video/video_compose.py`
(690 家族映射、975 atelier 防火墙)、`ink-theater/README.md`;
观测:`tools/base_tool.py:148`(自动埋点)、`backlot/state.py`。
