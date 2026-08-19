#!/usr/bin/env bash
# 单机八卡预设(2026-08-19 用户令:32B 底座 —— 策略推理 TP4 + LoRA
# 训练 4 卡,r=64)。评审 VLM 走百炼 API,不占卡。
# 用法与 run_grpo.sh 完全一致:
#   bash rl/run_grpo_8gpu.sh --fresh    # 全新训练
#   bash rl/run_grpo_8gpu.sh --stop     # 一键收摊
# BASE_MODEL 在 .env 里指 32B 本地权重目录,如:
#   BASE_MODEL=/data/models/Qwen3-32B
export VLLM_GPUS=${VLLM_GPUS:-0,1,2,3}
export TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7}
exec bash "$(dirname "$0")/run_grpo.sh" "$@"
