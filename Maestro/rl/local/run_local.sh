#!/usr/bin/env bash
# 本地推理 GRPO —— 唯一的运行文件(2026-08-21 用户裁决)
#
#   bash rl/local/run_local.sh              # 1 训练器 + N 条流
#   bash rl/local/run_local.sh --fresh      # 清零后再起
#   bash rl/local/run_local.sh --stop       # 一键收摊
#   TRAIN_GPU=0,1 STREAM_GPUS=2,3,4,5,6,7 bash rl/local/run_local.sh
#                                           # ↑ 双卡拆分训练 + 6 条流
#
# 与旧的 vLLM 路线(rl/run_grpo.sh)完全并存 —— 那条路一行未动,
# 因为 rl/env/loop.py 的三个钩子默认关闭。
#
# 进程只有两类(旧架构是四类:vLLM/收集器/训练器/农场):
#   ① 训练器 ×1  认领组 → 比值+裁剪+KL → 每 N 步广播 LoRA
#   ② 流 ×N      本地采样 → 视频 → 判官 → 组入队列 → 镜间换脑
# 两条文件通道:rl/data/queue/(组) 与 rl/state/live_adapter/(LoRA)
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

TRAIN_GPU=${TRAIN_GPU:-0}          # 训练器占哪些卡;"0,1" = 双卡拆分训练
                                   # (权重按层切开,显存翻倍,吞吐不变)
STREAM_GPUS=${STREAM_GPUS:-1}      # 流占哪些卡(逗号分隔),流数由此推出
LOGS=${LOGS:-rl/logs}

# ── 收摊 ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  echo "== 停止本地 GRPO"
  for pid in $(pgrep -f "rl.local.main"); do
    [[ "$pid" != "$$" && "$pid" != "$PPID" ]] && kill -9 "$pid" 2>/dev/null
  done
  pkill -9 -f "rl.local.main" 2>/dev/null && echo "  进程 ✓"
  sleep 2
  echo "== GPU 残留:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
    2>/dev/null || echo "  (无 nvidia-smi)"
  exit 0
fi

# ── 清零 ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--fresh" ]]; then
  shift
  echo "== FRESH:清空组队列与 adapter 广播"
  rm -rf rl/data/queue rl/state/live_adapter
  mkdir -p rl/data/queue/claimed rl/state/live_adapter
  echo "== FRESH 完成"
fi

# ── key 只认 .env(先 unset 壳残留,不互补)────────────────────────────
unset DASHSCOPE_API_KEY QWEN_API_KEY WAVESPEED_API_KEY IDEALAB_API_KEY
if [[ -f .env ]]; then set -a; source .env; set +a; fi

fail() { echo "❌ $1"; exit 2; }
[[ -n "${BASE_MODEL:-}" ]]        || fail "BASE_MODEL 缺失 —— 写进 Maestro/.env(本地权重目录)"
[[ -n "${DASHSCOPE_API_KEY:-}" ]] || fail "DASHSCOPE_API_KEY 缺失(可灵视频 + MaaS 文本)"
[[ -n "${WAVESPEED_API_KEY:-}" ]] || fail "WAVESPEED_API_KEY 缺失(t2i / 图像编辑)"
[[ -n "${IDEALAB_API_KEY:-}" ]]   || fail "IDEALAB_API_KEY 缺失(视频判官网关)"
python -c "import torch, peft, transformers, accelerate" 2>/dev/null \
  || fail "训练依赖缺失:pip install torch transformers peft accelerate"

mkdir -p "$LOGS" rl/data/queue/claimed rl/state/live_adapter

# 抗显存碎片:长短序列交替会把显存打成筛子,可分配总量够却拿不出连续块。
# expandable_segments 让分配器按需伸缩段,实测能省下几个 GB 的"虚胖"。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

IFS=',' read -ra GPUS <<< "$STREAM_GPUS"
N=${#GPUS[@]}

# 训练卡与流卡不许重叠 —— 重叠不会报错,只会两边一起 OOM,查半天
for tg in ${TRAIN_GPU//,/ }; do
  for sg in "${GPUS[@]}"; do
    [[ "$tg" == "$sg" ]] && fail "GPU$tg 同时出现在 TRAIN_GPU 和 STREAM_GPUS"
  done
done

echo "== 本地 GRPO:训练器 GPU[$TRAIN_GPU] + $N 条流 (GPU $STREAM_GPUS)"
echo "   底座 $BASE_MODEL"

PIDS=()
cleanup() {
  echo "== 收尾,停止子进程"
  for p in ${PIDS[@]+"${PIDS[@]}"}; do kill "$p" 2>/dev/null; done
}
trap cleanup EXIT INT TERM

# ① 训练器(先起:它要发布 v0,流才有脑可用)
CUDA_VISIBLE_DEVICES=$TRAIN_GPU \
python -m rl.local.main --role trainer --wandb "$@" \
  > "$LOGS/local_trainer.log" 2>&1 &
PIDS+=($!)
echo "   训练器 PID ${PIDS[-1]} → $LOGS/local_trainer.log"

# 等 v0 落盘
for _ in $(seq 1 120); do
  [[ -f rl/state/live_adapter/VERSION ]] && break
  sleep 5
done
[[ -f rl/state/live_adapter/VERSION ]] \
  || fail "训练器 10 分钟未发布 v0,见 $LOGS/local_trainer.log"
echo "   起点 adapter v$(cat rl/state/live_adapter/VERSION) 已就绪"

# ② N 条流
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} \
  python -m rl.local.main --role stream --worker "$i" --workers "$N" "$@" \
    > "$LOGS/local_stream$i.log" 2>&1 &
  PIDS+=($!)
  echo "   流 $i (GPU ${GPUS[$i]}) PID ${PIDS[-1]} → $LOGS/local_stream$i.log"
done

echo "== 全部就绪。跟日志:tail -f $LOGS/local_trainer.log"
wait
