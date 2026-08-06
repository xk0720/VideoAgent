#!/bin/zsh
# rainnight 雨夜黑帮 VO 短片(2026-08-06 用户令):可灵链路。
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a
exec python scripts/test_window_movie.py \
  --config configs/bailian.yaml \
  --screenplay /Users/kevin/Desktop/script-rainnight/script.json \
  --prompt "雨夜黑帮:命运的金色子弹" \
  --prompt-enhancer \
  --audio \
  --n-candidates 1 \
  --max-turns 3
