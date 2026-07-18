# 用强化学习提升 Brain 的 Tool-Calling 准确性 —— 调研报告与方案(2026-07-18,dev-rl)

> 用户令:调研 Crayotter 与 NEWTON 的 RL 用法;全面调研 agentic RL
> (long-horizon 等),重点 DPO(正负样本对)与 agentic GRPO 两个方向;
> 训练必须用开源 LLM;先报告、经同意后才改代码;一切操作在 dev-rl 分支。

## 一、任务定义:什么叫"tool calling 准"

Maestro 的 brain 不是自由聊天——每个决策点都是【菜单约束 + 严格 JSON】,
"准确性"可以拆成三层,每层的评判信号不一样:

| 层 | 什么算错 | 信号来源 | 成本 |
|---|---|---|---|
| **结构层** | JSON 解析失败 / 选了菜单外策略 / 参数越界(duration 不在 4-10)/ 引用了槽位清单外的 @ImageN / 契约标签(static:)漏进 prompt | 确定性校验器,**当场可算,零成本**(`_extract_json`、menu 校验、`validate_references`、`_scrub_*`) | 免费 |
| **决策层** | 策略选错(该 extend 却选 t2v,素材白传,修复轮数多、不收敛) | 台账:converged / stop_reason / 修复次数 / degraded_from / episode avoid | 延迟,一镜一条 |
| **生成层** | video_prompt / hint 写得差(首帧锁不住、循环、身份漂移) | VLM reviewer 缺陷报告 + verifier 盲测 A/B accept/reject | 延迟,最贵(要真生成) |

RL 的目标 = 把这三层的错误率一起压下去;三层信号密度和成本天差地别,
这直接决定训练算法怎么选(见方案节)。

## 二、Maestro 决策面盘点(RL 的"环境"长什么样)

六个决策点,全部已有 brain_calls.jsonl 落盘(stage/label/menu/context/
raw/parsed/usable/skill_chars):

| 决策点 | stage | 动作空间 | 当场校验 | 延迟信号 |
|---|---|---|---|---|
| 剧本 | window/scene_write | 结构化 JSON(shots/cast/setting/duration/end_state/variation/opening_frame) | 逐镜校验、词表校验 | 全片收敛率 |
| 图计划 | window/image-plan | 5 计划 × 逐张 {source, description} | 菜单成员、角色数匹配 | 图是否被消费、降级 |
| 条件策略 | window/generation-condition | 8 策略菜单 + video_prompt | 菜单成员、槽位引用闸门 | converged、修复轮数 |
| prompt 润色 | window/prompt_enhance | 重写文本 | 引用闸门(一次重试)、标签/建景句清洗 | reviewer/verifier |
| 修复决策 | repair/decide | 3 工具 + args(frame_range/hint) | 工具名、帧范围合法性 | **verifier 盲测 A/B accept/reject(天然正负对!)** |
| 基线锚 | window/baseline_anchor | 一次性 prompt | — | (用户人工看) |

环境特性(决定算法选型的三个硬事实):
1. **动作空间小而结构化**——菜单最多 8 项 + 受约束 JSON,不是开放文本。
   结构层奖励可以确定性计算(RLVR 式),这是最大的红利。
2. **环境步超贵**——一次完整 rollout = 真实视频生成 API + VLM 评审,
   一镜数十秒到分钟级、真金白银。在真环境上做大规模 on-policy 采样
   (裸 GRPO)不现实,必须分层:便宜的结构层大规模训,贵的生成层小
   规模/离线训。
3. **日志即数据集**——brain_calls.jsonl 已经记全了 (context, menu,
   decision);verifier 的 accept/reject 是现成的偏好标注器。缺的只是
   decision_id → outcome 的显式回填链接(现在靠 label+时序隐式对齐)。

## 三、参考工作:Crayotter 与 NEWTON 的 RL 用法

(调研代理产出,待回填)

### 3.1 Crayotter(GitHub API 逐文件精读,全文见附录 C)

**裁定:两个参考里唯一有真实可跑 RL 训练代码的**。`phase3_rl/` 是
基于 **verl 的 GRPO 管线**:训练开源 Qwen(0.5B/0.8B)做剪辑执行的
**多轮工具调用**,奖励**纯规则 RLVR**(工具执行结局算出,无奖励模型、
无偏好数据、无 SFT/LoRA);sglang 异步多轮 rollout(组大小 4,最多
8 轮),单卡 4090 冒烟规模(10 步、1 个 fixture),checkpoint 未回装
生产(生产 agent 仍用 API 模型)。

对我们最有价值的五个设计(细节见附录 C):
1. **奖励镜像 prompt 契约**:prompt 里的工作流纪律逐条翻译成顺序
   奖励(export 前必 inspect ±0.15、narrate 前必 validate +0.2/-0.4)
   —— 我们的 skill 法则(首帧 PIN、槽位引用、瘦身)同样可以逐条变成
   可验证奖励项,这是把"skill 教的"变成"模型学会的"的正道。
2. **一处 schema 三处消费**:工具 schema 单源导出到 运行时/prompt/
   verl 配置,训练分布 = 生产分布(RL 的 system prompt 直接复用生产
   prompt)。我们的 `_slot_manifest` + 菜单就是这个单源。
3. **fixture 隔离环境**:合成视频 + 金标轨迹 + 工具白名单 → rollout
   快、确定、免 API —— 我们的 S3 控本方案有了实证模板(mock 掉
   video_gen,决策正确性照样可判)。
4. **子进程工具运行时** + 机器可解析 JSON 返回(status/path/duration)
   → 免 LLM 判官的确定性成功判定。
5. 步级(±0.6 工具成败、-0.2 重复调用签名、顺序奖励)+ episode 级
   (export +1.0、时长差奖励、效率罚 -0.02/次)的**两级规则奖励结构**
   —— 与我们"结构层密集 + 结局稀疏"的两层设计互相印证。

局限(诚实记录):冒烟规模(单任务、10 步)未证明扩展性;vendored
verl 补丁未入库不可核验;训练果实未回流生产 —— 说明"训练能跑"和
"训练收益闭环"之间还有一段路,我们的方案必须把回装(vLLM 换端点)
设计进去。

### 3.2 NEWTON(本地精读,逐文件核实)

**裁定:仓库无任何训练代码**(全库 grep dpo/grpo/ppo/reward/trl/verl/
lora/finetune 零命中;git 全历史 6 commits 从未有过训练文件;依赖仅
genesis-world/openai/requests/numpy/mcp/imageio)。RL 只存在于 README
两句话:论文(arXiv:2605.18396)的 planner 是**唯一可训组件**,用
**Flow-GRPO 在真实多轮循环里 on-policy 优化**,VideoPhy-2 联合准确率
21.4→29.7(LTX-Video)/ 30.7→37.4(Veo-3.1),生成器不动;
"Release training code for open-sourced planner" 是**未勾选的 roadmap**。
放出的代码明确自称 training-free orchestrator。

对我们真正有用的是它**为 RL 准备数据的方式**(全部已实现在推理码里):
- 动作空间 = 工具调用(image_search/select_reference_image/img_create/
  make_keyframes/simulate/read_skill)+ 最终 video_prompt 文本;planner
  默认 gpt-5.5,但 test_function_call.py 给了 vLLM+Qwen3.5-9B 端点示例
  ——开源 planner 是预定训练对象(推断,仓库未明说)。
- **trace.json 逐轮全记**:每轮的工具调用(名/参/结果)、最终 prompt、
  条件预判 verdict(reasonable/reason/suggestions,含被拒历史)、盲测
  A/B 分([-10,+10] vs 固定 text-only 基线)、stop_reason、best_turn。
  A/B 分天然成对,接受/拒绝的条件提案天然成对 —— 结构即偏好数据,
  但仓库没有任何导出/聚合脚本(训练用途只是暗示)。
- 现成奖励信号清单(全部已算):盲测相对分、STOP 阈值(≥+5)、条件
  预判二值、到达成功的轮数、工具执行成败、函数调用离线回归器
  (test_function_call.py:expect_tool/expect_nonempty 等二值校验,
  CI 退出码)——第 7 项就是"结构层确定性奖励"的 NEWTON 版。
- 文档与代码脱节一处:tools/README 写的 VideoPhy2Client 绝对分
  (SA/PC 1-5,论文的天然 GRPO 奖励)**只有文档没有实现**。

**对 Maestro 的直接启示**:NEWTON 的公开形态 = "推理循环把奖励信号
全部算好并落盘,训练闭环在论文里"。我们的 brain_calls + verifier 与其
trace.json 同构,而且我们的信号更细(六决策点 vs 它一个 planner)。
它论文选择 on-policy Flow-GRPO 而非离线 DPO,佐证了"真环境在线训练
可行但必须小规模"(它的环境步同样是真生成 + VLM 评审)。

## 四、方向调研 A:agentic GRPO / long-horizon RL(全文见附录 A)

> 完整 40+ 篇带链接的调研在 `rl_appendix/A_agentic_grpo_landscape.md`;
> 这里只留决定我们方案的八个结论。

1. **ToolRL 的奖励分解与我们的校验器一一对应**(格式 {0,1} + 工具名/
   参数名/参数值分档给分):JSON 可解析↔格式、菜单成员↔工具名、槽位
   引用↔参数值。细分解显著优于二值;格式奖励权重过高有害。
2. **两阶段课程是贵环境的标准省钱法**(R1-Searcher):先只用免费的
   确定性校验器奖励训"接口正确"(甚至不用跑真管线),再花真 rollout
   训"决策好"。
3. **GiGPO 是与我们结构最合拍的信用分配**:episode 级组优势 + 按
   "同类决策点"分组的步级优势,零额外 rollout、免 critic —— 我们的
   决策点天然带类型标签(stage),按 (stage, 相近上下文) 分组即可移植。
4. **算法配置有共识**:GRPO 骨干 + Dr.GRPO 去归一化偏差(复合奖励跨
   阶段尺度不同)+ GSPO 序列级比率(动作单元=整个 JSON;Qwen3 MoE 必
   须)+ DAPO 动态采样(校验器饱和后组内零方差,不丢弃就白烧钱)+
   **观测 token(VLM 评审文本)从 loss 中 mask**(Search-R1/ReTool)。
5. **异步近乎必选**(AReaL):我们的 rollout 分钟级,同步 GRPO 会把
   trainer 饿死;历史日志无旧策略 logprob → 离线部分用 AWR(不需要
   logprob),不做天真重要性采样。
6. **Agent Lightning 的解耦是架构模板**:agent 原码不动,把每次 LLM
   调用采成 MDP 转移,训练是独立服务 —— brain_calls.jsonl 已是它的
   80%,补 decision_id + 回填即到位。
7. **修复控制器有近乎同构的先例**:Harness MDP 论文(2026)把
   verify/retry/branch 类结构动作当小离散策略,LLM 冻结,纯离线 AWR +
   终局奖励 —— 与我们 accept/segment/full 三选一完全同构,是**杠杆
   最高、成本最低的第一个 RL 靶子**。
8. **三个大警告**:(a) 模型判官可黑 —— VLM 分只能当有帽 shaping,
   盲测 A/B 因其盲性是唯一可靠锚;(b) 天真的逐轮密集奖励实测 **-14
   分**,过程/结局量级必须先在日志上校准;(c) Spurious Rewards:Qwen
   系上随机奖励也能召回大半"收益"(格式激发≠学习)—— 上真训练前
   必跑"仅格式奖励"消融当基线,防自欺。

## 五、方向调研 B:DPO 族偏好优化(全文见附录 B)

> 完整族谱与 30+ 篇依据在 `rl_appendix/B_dpo_for_agents_landscape.md`;
> 决定方案的七个结论:

1. **KTO 是我们的主力,不是 DPO**:KTO 不要成对数据,每条 (context,
   decision, 好/坏) 即可 —— 我们台账里最大量的信号(闸门通过/失败、
   converged 单边)正是这种形状;PLUM(确定性校验器→偏好标签的最近
   先例)实测 **KTO 常胜 DPO**。类不平衡用权重配平(1:1~4:3)。
2. **成对 DPO 留给真正成对的数据**:修复决策的 verifier accept/reject
   是天然同上下文对(近零噪声,Agent-Q 级数据免费产);跨 run 成败对
   次之。必须加 NLL 辅助项(["sigmoid","sft"])防 chosen 似然坍缩
   ——我们的决策是短结构化串,坍缩来得快。VLM 判官产的对一律
   label_smoothing≈0.1 / robust 损失。
3. **延迟信号的信用分配用 SDPO 段级配方**:找"首个分歧决策"造共享
   前缀的段级对 —— 论文要 MCTS 找分歧点,我们菜单约束下 **diff 解析
   后的 JSON 即得**,几乎免费。
4. **数据量级有底**:窄技能低千级干净 on-policy 对即可测得动
   (1.2K-1.9K 见效,10K 强,~20K 饱和)—— 我们几十个 attempt 的
   台账起步够 KTO,DPO 对要靠 verifier 台账攒。
5. **agent 先例齐全**:ETO(失败 vs 成功轨迹对,+22%)、DiaTool-DPO
   (工具对话控制,拒调 9.6→91%)、LoopTool(挖失败→合成难样本→
   迭代,8B 反超 32B 数据来源)、TL-Training(错误集中少数类别,
   关键 token 加权,1217 条追平闭源)。
6. **实操成本低得惊人**:Qwen3-8B QLoRA 单张 24G 可跑(TRL 原生支持
   工具调用数据格式;PEFT 下参考模型免费);按部署模式(非思考)训练。
7. **迭代不单训**:第一轮离线收益最大;之后 Maestro 正常运行就是采样
   循环(LoopTool 闭环,我们的闸门当判官);每轮重锚参考模型;评估用
   held-out 日志重放 + τ-bench 式 pass^k + BFCL 抽样防遗忘。

## 六、我的方案要点(定稿提案,已按调研修订)

### 一句话:**日志先行、KTO 主力、修复控制器第一靶、真 rollout 最后花**

调研改变了草案的四件事:(1) S1 主力从 DPO 改为 **KTO**(我们最大量的
信号是不成对二值,PLUM 实测 KTO 更稳);(2) 新增 **修复控制器离线
AWR** 为第一个 RL 靶(Harness-MDP 论文与我们同构,三动作离散、终局
奖励已落盘、LLM 冻结 —— 杠杆最高成本最低);(3) 架构定为 **Agent
Lightning 式解耦**(Maestro 原码不动,训练是独立服务,靠 decision_id
把日志变成 MDP 转移);(4) Crayotter 提供了 S2/S3 的**实证模板**:
verl+GRPO+规则奖励在"视频 agent 多轮工具调用"上真实跑通过(冒烟
规模),其"奖励镜像 prompt 契约"与"fixture 隔离环境"两个设计直接
写进我们的 S2/S3 —— 我们的 skill 法则(首帧 PIN/槽位引用/瘦身/顺序
纪律)逐条翻译成奖励项;S3 先用 mock video_gen 的 fixture 环境把决策
正确性训到位,真生成只做最后小规模校准。

### 总原则:三层信号 → 三段课程,先便宜后贵,日志先行

**S0 数据管道(纯代码,无训练,先行必做)**
- brain_calls 记录加 `decision_id`(uuid);台账/结果记录回写
  `decision_ids`;新增离线脚本把 (context, menu, decision, 结构层判定,
  延迟结局) 拼成标准训练样本 JSONL。
- 存量 attempt 日志用 label+时序做一次性回填(诚实标注 confidence)。
- 开源模型接入:OpenAICompatLLM 已支持任意 OpenAI 兼容端点 → vLLM 服务
  Qwen3(8B 起步)零代码接入,先跑 SFT 前的行为基线(结构层错误率)。

**S1 离线偏好训练:SFT 温启动 → KTO 主力 → DPO 精修(全部免新采样)**
- **SFT 温启动**(非可选,所有 agent 先例的共同起点):只用"闸门通过
  且最终收敛/verifier 接受"的决策做行为克隆 —— 即把现有 gpt-5.6-sol
  的好决策蒸馏进 Qwen3-8B(LoRA r16-32,非思考模式,与部署一致)。
- **KTO 全量台账**(最大量、最低噪声、零挖掘成本):每条 (context,
  decision, 好/坏) 直接进 KTOTrainer;好/坏 = 确定性闸门 ∧ 延迟结局;
  类不平衡用 desirable/undesirable_weight 配平到 1:1~4:3。
- **DPO 只吃真正成对的**,按价值排序:
  1. repair/decide 的 verifier accept/reject(同上下文天然对,近零噪声);
  2. enhancer 被引用闸门拒掉的 prompt vs 重试通过版(现成对);
  3. 同 label 跨 run 成败对(需上下文相似度过滤 + confidence 标注);
  损失 = ["sigmoid","sft"](NLL 防 chosen 坍缩);VLM 判官产的对加
  label_smoothing=0.1。延迟的 converged 信号不摊到全轨迹 —— SDPO 式
  找"首个分歧决策"(diff 解析后的 JSON,免费)造段级对。
- 实操:TRL(原生工具调用数据格式),QLoRA 单张 24G 起步;评估 =
  held-out 日志重放决策准确率 + pass^k 一致性 + BFCL 抽样防遗忘。

**S1.5 修复控制器离线 AWR(调研新增,第一个真 RL 靶)**
- accept / regenerate_segment / regenerate 三选一 + 终局奖励(verifier
  接受、收敛、修复轮数成本)已全部落盘 —— 与 Harness-MDP 论文
  (arXiv:2607.05458)同构:小离散动作空间、LLM 执行器冻结、纯离线
  advantage-weighted regression,**不需要旧策略 logprob**(历史日志
  没有,天真重要性采样不可用,AWR 正好绕开)。
- 这一步产出的还是同一个 LoRA 策略(repair/decide 上下文格式不变),
  与 S1 合并训练或顺序训练皆可。

**S2 结构层 RLVR-GRPO(单步、便宜、可大规模)**
- 把每个决策点当"单步环境":从日志重放 context,策略采 G=8-16 个候
  选,奖励 = ToolRL 式分档(格式 {0,1} 小权重 + 菜单成员 + 槽位引用
  逐项给分 + 瘦身合规 + 标签零泄漏),不碰真视频管线,毫秒级一步。
- 算法配置(调研共识):GRPO 骨干 + **Dr.GRPO 去归一化**(复合奖励
  跨阶段尺度不同)+ **GSPO 序列级比率**(动作=整个 JSON;Qwen3 MoE
  向上扩时必须)+ **DAPO 动态采样**(校验器饱和后组内零方差,不弃组
  就白烧)+ 观测 token 从 loss mask。
- ⚠️ 先跑 **"仅格式奖励"消融**(Spurious Rewards 教训):Qwen 系很可
  能光靠格式激发就把结构层错误压掉大半 —— 若如此,S2 可以缩水甚至
  跳过,钱留给 S3。
- 框架:TRL GRPOTrainer 起步(单步够用);要多轮/MoE 时迁 verl。

**S3 长程 agentic GRPO(小规模、真环境,最后才花这笔钱)**
- 前提:S1/S2 的 held-out 指标见顶,且残余错误集中在"要看真结局才能
  判"的决策层。
- 形态:窗口循环包成 gym 环境(Agent Lightning/OpenEnv 式,Maestro
  原码不动);**异步** rollout(AReaL 式,rollout 分钟级,同步会饿死
  trainer);信用分配 = GiGPO 按 (stage 类型, 相近上下文) 分组的步级
  优势;终局奖励 = 盲测 A/B(唯一难黑的锚)+ converged - 修复轮数
  惩罚;VLM 分只做有帽 shaping;regenerate 动作设预算帽(ToRL 式,
  防"无脑重生成"策略)。
- 控本:480p + 2-3 镜短任务 + ScalingInter 式横向课程(先单决策段,
  后整循环);RLEP 式把已验证成功轨迹回放进每批;若量不够,DreamGym
  式"评审仿真器"(用日志里的 VLM 评审蒸馏)先仿真后迁移。
- 预算公式先算后跑:成本 ≈ 组大小 G × 步数 × 每 rollout API 成本
  (≈镜数 × 单镜生成+评审)。G=8、200 步、3 镜任务就是 ~4800 次
  生成调用 —— 这笔钱必须先过目。

### 诚实边界(先亮出来)
- 生成层奖励含 VLM 判断 → 有 reward hacking 面(prompt 迎合 reviewer
  而不是迎合观众);缓解:结构层确定性奖励打底 + 人工抽检闸门。
- 离线对的上下文漂移(run 间配对)必须做相似度过滤并标 confidence。
- S3 的成本模型必须先算清楚再开跑(每 rollout 的 API 成本 × 组大小 ×
  步数),报告里给预算公式。

## 七、风险清单与开放问题

**风险(按杀伤力排序,每条带缓解)**

1. **奖励黑客 · VLM 判官被策略钻营**(文献实锤:模型判官显著可黑)。
   缓解:确定性闸门是终审;VLM 分只做有帽 shaping;盲测 A/B 因其盲性
   当唯一结局锚;上线前人工抽检闸门。
2. **假收益 · 格式激发冒充学习**(Spurious Rewards:Qwen 上随机奖励
   也"涨分")。缓解:任何真训练前先跑仅-SFT 与仅-格式-奖励两条消融
   基线;RL 收益必须超出这两条才算数。
3. **策略坍缩 · 菜单约束下的 Echo Trap**(RAGEN:全组选同一菜单项 →
   熵坍缩;修复策略坍缩到"永远 accept"或"永远 regenerate")。缓解:
   监控组奖励方差/熵;DAPO 动态采样弃零方差组;regenerate 预算帽;
   KTO/DPO 阶段盯 logps/chosen 防 chosen 似然坍缩(加 NLL 项)。
4. **离线分布漂移** · 日志来自 gpt-5.6-sol 的决策分布,Qwen 温启动后
   自采的分布会漂。缓解:第一轮离线后立刻切换到"Qwen 跑管线 → 挖
   自己的日志 → 重训"的 LoopTool 闭环;跨 run 成对样本带相似度过滤
   与 confidence。
5. **密集奖励校准错**(实测 -14 分教训)。缓解:过程/结局量级先在
   历史日志上离线校准;确定性校验器项(精确)才许密集,模型分不许。
6. **S3 成本失控**。缓解:预算公式先过目(G × 步数 × 每 rollout 成
   本);480p 短任务;RLEP 回放;评审仿真器(DreamGym 式)兜量。
7. **通用能力遗忘** · 窄域 LoRA 把 Qwen 的通用工具调用磨掉。缓解:
   保留 KL 锚(不用 ORPO/SimPO);BFCL 抽样当回归金丝雀。

**开放问题(留给用户裁决)**

- Q1 底座选型:Qwen3-8B 起步(单卡 24G 可训、便宜)vs 直接 14B
  (决策质量上限更高)?建议 8B 起步,plateau 再上。
- Q2 六个决策点是一个 LoRA 通训,还是 repair 控制器单独一个头?
  建议先通训(数据共享、部署简单),指标分stage 监控。
- Q3 训练触发时机:攒多少 attempt 的日志才开第一轮 SFT+KTO?
  (经验律:窄技能千级样本;一个 attempt ≈ 20+ 决策记录 →
  ~50 个 attempt 起步可训,配合公开工具调用数据混训防过拟。)
- Q4 S3 是否要做,以及预算上限是多少 —— S1/S2 结果出来再议。
- Q5 训练算力在哪跑(本机 Mac 不行,需要至少一张 24G CUDA 卡;
  或 Tinker 托管 LoRA 零基建)。
