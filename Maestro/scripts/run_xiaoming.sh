#!/bin/zsh
# xiaoming 例子实跑(2026-08-05 用户令):百炼可灵老链路,新钉/切提前路由。
# 用法: zsh scripts/run_xiaoming.sh
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a
exec python scripts/test_window_movie.py \
  --config configs/bailian.yaml \
  --screenplay /Users/kevin/Desktop/script-xiaoming/script.json \
  --prompt "海边黄昏,小明与海鸥阿浪的对话短片" \
  --prompt-enhancer \
  --n-candidates 1 \
  --max-turns 3
