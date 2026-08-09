#!/bin/zsh
# 对比实验(2026-08-09 用户令):s4 海边旧信 + s2 天台魔术师
# ① 素朴基线(复用旧台账,纯文字 t2v 直拼)
# ② 新栈重跑(缝合师+同景派生+视区背景法)
# 前置:等当前面包店验证跑完再排队,避免可灵并发限流。
cd "$(dirname "$0")/.."
set -a; source .env; set +a
set +e
while pgrep -f "test_window_movie.py" > /dev/null; do sleep 60; done

T=/Users/kevin/.claude/jobs/5bd83ba8/tmp
echo "== naive s4 =="
python scripts/naive_baseline.py --src outputs/movie_20260808_055242 \
  --tag s4 > $T/naive_s4.log 2>&1
echo "naive s4 exit=$?"
echo "== naive s2 =="
python scripts/naive_baseline.py --src outputs/movie_20260808_042620 \
  --tag s2 > $T/naive_s2.log 2>&1
echo "naive s2 exit=$?"
echo "== newstack s4 =="
python scripts/test_window_movie.py --config configs/bailian.yaml \
  --screenplay scripts/sim_scripts/s4_seaside.json --prompt "海边旧信" \
  --prompt-enhancer --audio --n-candidates 1 --no-review \
  > $T/newstack_s4.log 2>&1
echo "newstack s4 exit=$?"
echo "== newstack s2 =="
python scripts/test_window_movie.py --config configs/bailian.yaml \
  --screenplay scripts/sim_scripts/s2_rooftop.json --prompt "天台魔术师" \
  --prompt-enhancer --audio --n-candidates 1 --no-review \
  > $T/newstack_s2.log 2>&1
echo "newstack s2 exit=$?"
echo "COMPARE PAIR FINISHED"
