#!/bin/zsh
# 空间圣经验证对(2026-08-10):面包店 + 天台,串行。
cd "$(dirname "$0")/.."
set -a; source .env; set +a
set +e
T=/Users/kevin/.claude/jobs/5bd83ba8/tmp
python scripts/test_window_movie.py --config configs/bailian.yaml \
  --screenplay scripts/sim_scripts/s3_bakery.json --prompt "晨光面包店" \
  --prompt-enhancer --audio --n-candidates 1 --no-review \
  > $T/sb_bakery.log 2>&1
echo "bakery exit=$?"
python scripts/test_window_movie.py --config configs/bailian.yaml \
  --screenplay scripts/sim_scripts/s2_rooftop.json --prompt "天台魔术师" \
  --prompt-enhancer --audio --n-candidates 1 --no-review \
  > $T/sb_rooftop.log 2>&1
echo "rooftop exit=$?"
echo "SPACE BIBLE PAIR FINISHED"
