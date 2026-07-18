# 附录 C:Crayotter phase3_rl 精读(调研代理原文,2026-07-18)

> 主报告 3.1 节的完整依据。读取方式:GitHub API 逐文件(网络原因整库
> clone 不可行);fork xk0720/Crayotter@8d79e6c,phase3_rl/ 树与上游
> idwts/Crayotter 逐字节一致。

## 裁定

**真实可跑的(冒烟规模)RL 训练代码存在**,在 `phase3_rl/`:基于
**verl 的 GRPO 管线**,训练开源 Qwen(默认 Qwen2.5-0.5B-Instruct,
实测复现用 Qwen3.5-0.8B)做第三阶段(剪辑执行)**多轮工具调用**,
奖励**纯规则(RLVR 式)**,从工具执行结局计算 —— 无奖励模型网络、
无人类偏好数据、无 DPO/SFT/LoRA。明确标注为单卡冒烟验证
(RTX 4090 24G,total_training_steps=10,一个 fixture 任务);
**训练出的 checkpoint 没有回装进主应用**(生产 agent 用 API 模型
qwen-plus)。论文页自述:"RLVR setup with verifiable rewards over
editing artifacts"、"RLVR-ready traces"。

## 主应用架构(与 RL 相关的要点)

三阶段 LangGraph:planner(JSON 依赖 DAG,tool_hint 限 4 个工具名)
→ 资源调度器确定性执行 → 剪辑执行(受控 DAG 编辑器 → 短片结构化
编辑器 → ReAct 兜底,recursion_limit=100)。设计铁律(AGENTS.md):
"Agents/planners 决定任务形状与依赖;**只有确定性执行器调用工具**并
登记 artifact。" 24 个 LangChain @tool,全部返回机器可解析 JSON
(status/path/duration)。

## phase3_rl 技术细节(全部有实现)

- **算法/框架**:verl main_ppo @ GRPO 模式(adv_estimator=grpo),
  lr 1e-6,use_kl_loss(coef 0.001, low_var_kl),entropy_coeff=0,
  FSDP bf16,**sglang 异步多轮 rollout**(max_assistant_turns=8,
  rollout.n=4 即组大小),vendored verl 本地补丁(Qwen3.5 mRoPE 等,
  未入库,不可核验)。
- **动作空间**:真实剪辑工具的 OpenAI function calls;
  `convert_to_openai_tool(ALL_TOOLS)` 一处 schema 三处消费
  (运行时函数调用 / prompt 工具目录 / verl 工具配置 YAML)——
  训练、提示、执行永不漂移。
- **rollout 工具执行**:每次调用一个**新子进程**(JSON stdin →
  哨兵行结果),隔离的逐 episode 工作区;成功判定 = 解析 JSON 读
  status=="success" + 错误标记启发式 + 输出文件存在性。
- **奖励(reward.py,纯规则)**:
  - 步级:tool_success +0.6/-0.8;有产物 +0.1;returncode -0.1;
    重复同签名调用 -0.2;**顺序奖励把 prompt 里的工作流纪律翻译成
    可验证信号**(export 前必 inspect +0.15/-0.15;narrate 前必
    validate +0.2/-0.4;merge 前 ≥2 次 cut +0.1/-0.1)。
  - episode 级:步和 + 成功 export +1.0 + 时长奖励
    max(-0.5, 0.8-|实际-目标|/max(目标,1)) + 完成消息 +0.3 +
    超 6 次调用每次 -0.02 效率罚。
- **数据**:fixture 制(隔离、可再现、免 API):合成视频(moviepy
  ColorClip)+ 金标脚本轨迹 + 允许工具白名单;导出 verl JSONL
  (reward_model: {style: "rule"});**RL 的 system prompt 直接复用
  生产的 REACT_EDITOR_PROMPT** —— 训练分布 = 生产分布。
- **verl 之外的本地 rollout 硬件无关**:Phase3RolloutEnv +
  ScriptedPolicy/OpenAIToolPolicy,逐 episode 落
  trace/tool_events.jsonl/messages.json。

## 不存在的东西

无 DPO/偏好对、无 SFT、无 LoRA/TRL、无学习型奖励模型、无 critic
(GRPO 免 critic)、planner/Phase1/2 不训、checkpoint 无回装路径。

## 生产应用里可当训练数据的信号(已落盘未收割)

结构化工具轨迹事件(tool_started/completed/failed + 时长/错误)、
phase3_evaluation JSON(decision: finish|fallback,时长差、修复)、
验证工具的 pass/fail/score JSON、计划评审人类改稿(approved vs
revised 潜在偏好对,未收割)。

## 值得 Maestro 借的六件(代理归纳)

1. 一处 schema 三处消费(训练/提示/执行零漂移);
2. **奖励镜像 prompt 契约**(prompt 里的纪律逐条变成 order bonus);
3. 工具返回机器可解析 JSON → 免 LLM 判官的确定性成功判定;
4. 受控结构化计划(Pydantic Literal/Field 界)走幸福路径,ReAct 只
   兜底;parallel_tool_calls=False 保可再现;
5. **fixture 隔离环境**:合成视频 + 金标轨迹 → rollout 快、确定、
   免 API;子进程隔离 + 哨兵行协议;
6. 重复调用按"工具名:规范化 JSON 参数"签名判定。
