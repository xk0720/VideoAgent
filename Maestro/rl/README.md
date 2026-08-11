# rl/ — brain(generation-condition)semi-online GRPO

**一行命令**:
```
zsh rl/run_grpo.sh          # GPU 机:vLLM(Qwen3+LoRA 热载)+ rollout
                            # 农场(--rl-group,review 开)+ 收集 + 训练
zsh rl/run_grpo.sh --smoke  # 本机自检:mock rollout→收集→分组→advantage
```

- DESIGN.md                 设计全文(v1 + v2 增补)
- collect/build_step_dataset.py  S1 冷启动:历史轨迹 → SFT 蒸馏语料
- collect/watch_online.py   在线收集:rl_steps → groups_online.jsonl
- reward/reward_fn.py       离线 reward(与生产闸门同构)
- train/train_online.py     在线 trainer(组相对优势 PG + LoRA + 热载)
- train/grpo_condition.py   离线干跑/trl 骨架(管道自检用)
- configs/online.yaml       编排参数参考

要点:rollout 必须开 review(reward 来源);enhancer 关(brain 学写
终稿);max_turns=1(无修复,reward=纯候选质量);非训练 agent 全部
冻结在 models.llm。

## 训练服务器约束(2026-08-11 用户令)
- 服务器 GPT/Google 全不可达 → RL 链全体 API 角色走【百炼】:
  冻结 agent = qwen-max,评审/验收 VLM = qwen-vl-max(抽帧路),
  视频 = 可灵,t2i/编辑 = wavespeed;只需 DASHSCOPE_API_KEY(或
  QWEN_API_KEY,脚本自动互补)+ WAVESPEED_API_KEY。
- 隔离纪律:RL 专属配置 = rl/configs/server_bailian_qwen.yaml,
  只被 rl/run_grpo.sh 引用;configs/ 与既有跑法零改动,主管线代码
  零改动(管线里唯一的 RL 挂点 rl_group 缺省 0 = 行为与旧版逐字节
  一致,有回归测试锁死)。
- 服务器部署:git clone 本分支 → pip install vllm torch transformers
  peft → 写 .env(两把 key)→ `zsh rl/run_grpo.sh`。
