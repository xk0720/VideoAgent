# rl/ — brain(generation-condition)在线 RL

- DESIGN.md          框架设计(MDP/上下文/reward/在线架构/路线图)
- collect/build_step_dataset.py  S1:扫 outputs/movie_* → steps.jsonl
                     (245 样本已建,gpt-5.6-sol 轨迹,可先 SFT 蒸馏)
- reward/reward_fn.py  r = 0.2·format + 0.8·task(与生产闸门同构)
- train/grpo_condition.py  GRPO 骨架(trl;--dry-run 无 GPU 自检)
- configs/online.yaml  S3 三进程编排(vLLM policy / rollout 农场 / trainer)

重要:task reward 依赖评审 —— rollout 必须开 review(--no-review 的
run 只有 format 分,本批 245 样本中位 0.2 即此故)。
