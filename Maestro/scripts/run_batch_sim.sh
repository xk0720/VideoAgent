#!/bin/zsh
# 批量直跑(2026-08-08 用户令):雨夜打头,再跑五个自拟剧本。
# --no-review = 关评审关修复(首掷即收,台账如实记 review_disabled)。
cd "$(dirname "$0")/.."
set -a; source .env; set +a
set +e
run_one() {
  local name="$1" script="$2" prompt="$3"
  echo "===== RUN $name ====="
  python scripts/test_window_movie.py \
    --config configs/bailian.yaml \
    --screenplay "$script" \
    --prompt "$prompt" \
    --prompt-enhancer --audio --n-candidates 1 \
    --no-review \
    > "/Users/kevin/.claude/jobs/5bd83ba8/tmp/batch_${name}.log" 2>&1
  echo "===== $name exit=$? ====="
}
run_one rainnight /Users/kevin/Desktop/script-rainnight/script.json "雨夜黑帮:命运的金色子弹"
run_one s1 scripts/sim_scripts/s1_snowbus.json "雪夜末班车"
run_one s2 scripts/sim_scripts/s2_rooftop.json "天台魔术师"
run_one s3 scripts/sim_scripts/s3_bakery.json "晨光面包店"
run_one s4 scripts/sim_scripts/s4_seaside.json "海边旧信"
run_one s5 scripts/sim_scripts/s5_rainstop.json "雨停之前"
echo "ALL BATCH RUNS FINISHED"
