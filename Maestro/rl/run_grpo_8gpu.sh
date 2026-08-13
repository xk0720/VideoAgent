#!/usr/bin/env bash
# 单机八卡预设(2026-08-12 用户令):策略推理 2 卡 + 训练 4 卡;
# 6,7 两卡预留给【本地评审 VLM】(方案已评估待裁决,启用前闲置)。
# 用法与 run_grpo.sh 完全一致:
#   bash rl/run_grpo_8gpu.sh            # 正式
#   bash rl/run_grpo_8gpu.sh --smoke    # 自检
export VLLM_GPUS=${VLLM_GPUS:-0,1}
export TRAIN_GPUS=${TRAIN_GPUS:-2,3,4,5}
# export VLM_GPUS=6,7                  # 本地评审 VLM(待裁决后启用)
exec bash "$(dirname "$0")/run_grpo.sh" "$@"
