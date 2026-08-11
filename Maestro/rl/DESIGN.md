# Maestro Brain RL(Agent-R1 式 step-level 在线训练)

目标:训练 brain(Qwen3)"按上下文产出 tool call + prompt"的能力。
episode = 一镜闭环;step = brain 一次决策(image_plan / condition+prompt /
repair decide / junction 类)。state = 长期记忆(技能+episode_guidance)
+ 当前 rollout 实况(台账视图/junction/槽位清单)—— 即现有 brain prompt。

## 数据层(零管线侵入)
管线已落 (state, action, outcome):brain_calls.jsonl(context/raw/parsed/
usable/decision_id)× repair-outcome/shot_outcome(decision_id → new_total/
verifier_score)。collect/build_step_dataset.py 做 join → steps.jsonl。

## Reward
r = 0.2·format + 0.8·task
- format:JSON 可解析 .4 + 工具/策略在菜单 .2 + 引用⊆清单 .2 + 语言 .1
  + 必填字段 .1;unusable → 0(复用闸门判据)
- task(按 step 类型归因):
  condition/prompt → weighted_total + 0.3·(verifier/10)
  repair decide    → Δ = clip(new_total − prev_total, −1, 1);
                     accept → ±(final − bar)
  image_plan       → 0.1·无降级 + 0.5·weighted_total
  episode 加成:verified → 全 step +0.3;−0.05/修复轮
- GRPO 组 = (script, shot_idx, junction_kind) 的多 rollout,adv = r − 组均值

## Online 架构
vLLM 服 Qwen3(OpenAI-compat)← checkpoint 热载 ← GRPO trainer(verl/trl)
N× rollout worker(现有管线,models.crew.video_brain 指 vllm 端点)
→ StepRecorder join → rl/data/steps.jsonl(带 policy_version;训练侧
过滤陈旧度 ≤K 版)。reward 异步补记(decision_id 串联)。

## 路线图
S1 冷启动:build_step_dataset 扫全部 outputs/movie_* → SFT 蒸馏
   (gpt-5.6-sol 轨迹,usable 且高分为正样本)
S2 离线 RL 试跑:existing 数据 + reward_fn → GRPO 干跑(梯度管道验证)
S3 在线:vllm 换脑 + rollout 农场 + 异步 join + 热载


## v2 增补(2026-08-10 深夜,用户连环裁决)
- 只训 generation-condition;junction_stitcher 独立 crew 槽位(裁决 a:
  RL 换脑时非训练角色冻结在 models.llm)。
- semi-online GRPO:组必须 on-policy —— 主干+单步分支(rollout 内每镜
  同 state 采 K 个决策各生成一候选,评审择优推进主干;分支永不活过
  一镜 → 组内 state 逐字节相同,资产天然共享)。历史数据伪组仅用于
  梯度管道自检,不作正式训练(诚实定位:那是 off-policy 加权 BC)。
- 组记录自包含:state(context+menu)+action(completion 规范 JSON)
  +reward 原料全在 rl_steps.jsonl,收集/训练不依赖旁路文件。
- 一键命令:`zsh rl/run_grpo.sh`(GPU 机)/ `--smoke`(本机全链自检:
  mock rollout → 收集 → 分组 → advantage)。
- trainer 诚实定位:组相对优势 policy gradient(LoRA),陈旧度 ≤K 版
  过滤,无 ratio clip;vLLM --enable-lora 热载 adapter,policy_version
  经 MAESTRO_POLICY_VERSION 注入 rollout 记录。
