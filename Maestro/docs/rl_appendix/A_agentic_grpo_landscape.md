# 附录 A:agentic GRPO / long-horizon RL 全景(调研代理原文,2026-07-18)

> 主报告 `../RL_TOOLCALLING_RESEARCH_2026_07_18.md` 第四节的完整依据。
> 检索与筛选口径:菜单约束 JSON 决策、确定性校验器、贵环境步、开源模型。

## 1. GRPO 基础与重要修正

- **GRPO**(DeepSeekMath/R1):每 prompt 采 G 条,组内归一化奖励当优势,
  免 critic。7-32B 单机可行。https://arxiv.org/abs/2402.03300
- **DAPO**(ByteDance):Clip-Higher(防熵坍缩)、**Dynamic Sampling**
  (全组同奖励 = 零梯度组,丢弃重采 —— 对我们至关重要:校验器饱和后
  组内零方差会白烧 rollouts)、token 级聚合、去 KL。
  https://arxiv.org/abs/2503.14476
- **Dr. GRPO**:去掉长度归一与组内 std 归一两个偏差(复合奖励跨阶段
  尺度不同时 std 归一会扭曲权重);Qwen2.5-Math-7B 8×A100 27 小时。
  https://arxiv.org/abs/2503.20783
- **GSPO**(Qwen 团队,Qwen3 后训练所用):序列级重要性比替代 token 级,
  MoE(Qwen3-30B-A3B 等)稳定训练必备;我们的动作单元本来就是整个
  JSON,概念上更对。https://arxiv.org/abs/2507.18071
- 补充:二值奖励下组均值中心化失效(gradient starvation,
  https://arxiv.org/html/2605.07689);tricks 消融
  https://arxiv.org/pdf/2508.08221

## 2. 工具调用 RL 专线

- **ToolRL**(最直接相关):奖励 = 格式 {0,1} + 分解正确性 [-3,3]
  (工具名/参数名/参数值分档给分);细分解 > 二值;格式权重过高有害;
  动态奖励调度(先格式后正确性)。与我们校验器一一对应:JSON 可解析↔
  格式、菜单成员↔工具名、槽位引用↔参数值。
  https://arxiv.org/abs/2504.13958
- **ToRL**:逐轨迹工具调用预算帽 C(防"无脑多调");可执行性小惩罚。
  https://arxiv.org/abs/2503.23383
- **ReTool**:SFT 冷启动 → RL;**环境观测 token 从 loss 中 mask**
  (我们的 VLM 评审文本不许吃梯度);32B 400 步 RL 见效。
  https://arxiv.org/abs/2504.11536
- **ARTIST**(Microsoft):纯 outcome GRPO 修多轮 function calling,
  τ-bench 翻倍。https://arxiv.org/abs/2505.01441
- **Search-R1** / **R1-Searcher**:观测 mask + 简单 outcome 奖励;
  两阶段课程(先只奖调用格式、再奖答案)= 贵环境的省钱课程模板。
  https://arxiv.org/abs/2503.09516 / https://arxiv.org/abs/2503.05592
- **Agent-R1 / StepPO**:每轮当步级 MDP 转移(不是不断拼长的单序列)
  —— 与我们逐决策 JSONL 记录同构。https://arxiv.org/abs/2511.14460
- 2025-26 新作:**EGPO**(熵增强优势,4B 上 BFCL 超 GPT-4o,
  https://arxiv.org/pdf/2508.05118);**RC-GRPO**(组内奖励方差坍缩 →
  奖励条件化 SFT 造组内多样性,直接复用混质量日志轨迹,
  https://arxiv.org/abs/2602.03025);**Iterative Reward Calibration**
  (⚠️ 天真的逐轮密集奖励最多 **-14 分**,量级须校准,
  https://arxiv.org/abs/2604.02869);**ARPO**(工具响应后熵尖峰处
  分叉采样,半预算,https://arxiv.org/abs/2507.19849);**FineStep**
  (可执行校验器出步级过程奖励,https://arxiv.org/pdf/2605.04719)

## 3. 多轮/长程信用分配与环境包装

- **RAGEN/StarPO**:Echo Trap(奖励方差坍缩→熵降→梯度尖峰);监控
  组奖励方差/熵/梯度范数;菜单约束 agent 特别容易坍缩到单一菜单项。
  https://arxiv.org/abs/2504.20073
- **GiGPO**(NeurIPS 2025,**与我们结构最合拍**):episode 级组优势 +
  锚状态组的步级优势(同状态不同动作比折扣回报),零额外 rollout;
  我们按 (stage 类型, 相近上下文) 分组即可移植。
  https://arxiv.org/abs/2505.10978
- **verl-agent**:步独立多轮 rollout(每步独立输入,非增长转录)——
  与我们每决策独立上下文的形态一致;支持 GiGPO/GRPO/DAPO/GSPO。
  https://github.com/langfengQ/verl-agent
- **AgentGym-RL / ScalingInter-RL**:训练中渐进加长允许交互轮数
  (横向课程)。https://arxiv.org/abs/2509.08755
- **SWEET-RL**:特权 critic(训练时看得到终局)给逐轮打分当密集奖励
  —— 我们训练时有 A/B 终判/收敛这类特权信息。
  https://arxiv.org/abs/2503.15478
- **LOOP**(Apple):RLOO 基线 + PPO 裁剪,单份 LLM 内存,32B 胜 o1
  9 分 —— 轨迹级信用足够论。https://arxiv.org/abs/2502.01600
- **ArCHer**:轮级 off-policy TD + 轮内 token 策略,比 PPO 高 ~100×
  样本效率 —— 贵环境下 off-policy 的经典论据。
  https://arxiv.org/abs/2402.19446
- **Agent Lightning**(Microsoft,**架构模板**):agent 原码零改动,
  sidecar 把每次 LLM 调用采成统一 MDP 转移,回报分摊到各调用后逐转移
  单轮 RL —— 我们的 brain_calls.jsonl 已经是它的 80%。
  https://arxiv.org/abs/2508.03680
- **AReaL**:全异步 + staleness 校正 —— rollout 分钟级到小时级时同步
  GRPO 会把 trainer 饿死,异步近乎必选。https://arxiv.org/abs/2505.24298
- **SkyRL**:真环境(SWE-Bench)异步派发参考架构。
  https://github.com/NovaSky-AI/SkyRL
- 横向发现:**horizon reduction**(短段训练、长程泛化,
  https://arxiv.org/html/2605.02572v1);长程配方系统研究(小模型要
  分阶段奖励,大模型简单密集奖励即可;~1K 平衡难度任务是甜点;环境
  不稳定会腐蚀策略,https://arxiv.org/abs/2603.21972)

## 4. RLVR 奖励设计

- 格式+正确性分解是事实标准;ToolRL 细分与时间调度更优。
- ⚠️ 奖励黑客:规则校验器难黑,**模型判官(我们的 VLM reviewer)显著
  可黑**(https://arxiv.org/pdf/2604.15149);盲测 A/B 因其盲性是最难
  黑的锚;“装饰性工具调用”需因果关联奖励(Proof-of-Use,
  https://arxiv.org/pdf/2510.10931);Weng 总览
  https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- ⚠️ **Spurious Rewards**:Qwen 系上随机/错误奖励也能召回大半收益
  (格式激发而非学习)—— 上训练前先跑"仅格式奖励"消融,省钱又防
  自欺。https://arxiv.org/abs/2506.10947
- 过程 vs 结局:确定性校验器可安全密集化;模型分只做有帽 shaping;
  量级先在日志上校准(-14 分教训)。AgentPRM(promise+progress)
  https://arxiv.org/abs/2511.08325

## 5. 训练栈与开源模型实操

- **verl**(主流,Agent Loop 多轮工具调用,SGLang/vLLM server,
  MoE 走 Megatron):https://github.com/volcengine/verl
- **TRL GRPOTrainer**(最易入门,单轮为主;多轮走 OpenEnv Gym 式规范
  https://huggingface.co/blog/openenv)
- **OpenRLHF**(Ray+vLLM+ZeRO3,TIS off-policy 校正,70B+)
- **Tinker**(托管 LoRA 训练 API,零基建,Qwen3 MoE 可用)
- **LoRA Without Regret**:RL 场景 LoRA(全层、~10× 学习率)≈ 全参
  —— 32B 单机 8 卡可行,权重同步到 vLLM 近即时。
  https://thinkingmachines.ai/blog/lora/
- 16 框架横评:https://huggingface.co/blog/async-rl-training-landscape
- 算力锚点:7B 接口级 RL = 单机数天;32B 真环境 agentic RL = 多机数周
  (除非 rollout 便宜或复用)。

## 6. 贵环境专题

- **Harness MDP + AWR 离线 RL**(2026,**与 Maestro 修复控制器近乎
  同构**):harness 结构动作(verify/retry/branch)当小离散策略,LLM
  执行器冻结,纯离线 AWR + 终局 rubric 奖励。
  https://arxiv.org/abs/2607.05458
- **OREO**(离线 soft-Bellman,从失败也能学,无需成对偏好):
  https://aclanthology.org/2025.findings-acl.464.pdf
- **RLEP / ExGRPO / 新鲜度 PER**:已验证成功轨迹回放进每个 GRPO 批,
  贵成功样本反复榨取。https://arxiv.org/abs/2507.07451
- **DreamGym**(Meta):把环境动态蒸馏成 LLM 经验模型(离线日志初始
  化),合成环境里训 + 少量真环境迁移,>30% 提升 —— "rollout 太贵"
  的最强已发表答案。https://arxiv.org/abs/2511.03773
- 历史日志无旧策略 logprob → 用 AWR(不需要)或 PPO-EWMA 近似,
  不做天真重要性采样。https://arxiv.org/pdf/2605.12070
