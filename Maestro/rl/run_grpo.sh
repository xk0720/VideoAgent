#!/bin/zsh
# ══════════════════════════════════════════════════════════════════════
#  Maestro brain GRPO 一键训练(semi-online;2026-08-10 用户令)
#
#    zsh rl/run_grpo.sh            # 全链:vLLM 策略 + rollout 农场 +
#                                  # 收集器 + trainer(需 GPU 机)
#    zsh rl/run_grpo.sh --smoke    # 无 GPU 自检:数据流/分组/advantage
#
#  四进程:①vLLM 服 Qwen3(--enable-lora,adapter 热载)
#          ②rollout 农场(现有管线,--rl-group K,review 开、enhancer
#            关、max_turns=1;video_brain 指 vLLM,其余 agent 冻结)
#          ③收集器(rl_steps → groups_online.jsonl,补 reward)
#          ④trainer(组相对优势 PG + LoRA,热载回 ①)
# ══════════════════════════════════════════════════════════════════════
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
RL=$REPO/rl
LOGS=$RL/logs; mkdir -p $LOGS $RL/state $RL/data
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
VLLM_PORT=${VLLM_PORT:-8000}
RL_GROUP=${RL_GROUP:-4}
PIDS=()
cleanup() { for p in ${PIDS[@]:-}; do kill $p 2>/dev/null; done }
trap cleanup EXIT INT TERM

# ── smoke:无 GPU 自检 ────────────────────────────────────────────────
if [[ "${1:-}" == "--smoke" ]]; then
  echo "== smoke: mock rollout → collector → trainer 分组(全链合成)"
  SBX=$(mktemp -d)
  python - "$SBX" <<'PY' || exit 1
import sys, json, pathlib, logging
sys.path.insert(0, "src"); sys.path.insert(0, "tests/unit")
logging.disable(logging.CRITICAL)
sbx = pathlib.Path(sys.argv[1]); run = sbx / "movie_rlsmoke"
from test_rl_group import _SamplingLLM, _VG, _components
import maestro.pipeline.window_loop as wl
wl._last_frame = lambda v, o: None
import os; os.environ["MAESTRO_POLICY_VERSION"] = "1"
from maestro.pipeline.window_loop import generate_movie_windowed
generate_movie_windowed("a cat walks", cache_dir=run,
                        llm=_SamplingLLM(), max_turns=1, n_candidates=1,
                        rl_group=3, rl_temperature=0.8,
                        **_components(_VG()))
n = len((run / "rl_steps.jsonl").read_text().splitlines())
assert n >= 1, "mock rollout 未产出组记录"
print(f"[smoke] mock rollout OK: {n} groups in {run}")
PY
  python rl/collect/watch_online.py --once --outputs "$SBX"     --out "$SBX/groups.jsonl" --state "$SBX/state.json" || exit 1
  python rl/train/train_online.py --dry-run --data "$SBX/groups.jsonl"     | tee /tmp/rl_smoke_trainer.txt || exit 1
  grep -qE "usable groups=[1-9]" /tmp/rl_smoke_trainer.txt || { echo "❌ trainer 分组为 0"; exit 1 }
  python -m pytest tests/unit/test_rl_group.py -q || exit 1
  rm -rf "$SBX"
  echo "== SMOKE OK(采样→落盘→收集→分组→advantage 全链通;"
  echo "   真训练需 GPU 机跑本脚本无参形态)"
  exit 0
fi

# ── 预检(缺什么直说什么)──────────────────────────────────────────
set -a; source .env 2>/dev/null; set +a
fail() { echo "❌ $1"; exit 2 }
command -v vllm >/dev/null || fail "vllm 未安装:pip install vllm"
python -c "import torch, peft, transformers" 2>/dev/null \
  || fail "训练依赖缺失:pip install torch transformers peft"
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || fail "无 CUDA GPU(vLLM+LoRA 训练需要;Mac 请在 GPU 机上跑)"
[[ -n "${GEMINI_API_KEY:-}" ]] || fail "GEMINI_API_KEY 缺失(评审=reward 来源)"
[[ -n "${DASHSCOPE_API_KEY:-}${WAVESPEED_API_KEY:-}" ]] \
  || fail "视频后端 key 缺失(DASHSCOPE/WAVESPEED)"
[[ -n "${OPENAI_API_KEY:-}" ]] || fail "OPENAI_API_KEY 缺失(冻结 agent 用)"

# ── ① vLLM 策略服务 ──────────────────────────────────────────────────
if ! curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null; then
  echo "== 启动 vLLM($BASE_MODEL)"
  vllm serve "$BASE_MODEL" --served-model-name brain \
    --enable-lora --max-loras 4 --port $VLLM_PORT \
    > $LOGS/vllm.log 2>&1 &
  PIDS+=($!)
  for i in $(seq 1 120); do
    curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null && break
    sleep 5
  done
  curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null \
    || fail "vLLM 120×5s 未就绪,见 $LOGS/vllm.log"
fi
echo "== vLLM 就绪"

# ── ③ 收集器 & ④ trainer ────────────────────────────────────────────
python rl/collect/watch_online.py > $LOGS/collector.log 2>&1 &
PIDS+=($!)
python rl/train/train_online.py --model "$BASE_MODEL" \
  --vllm-url "http://localhost:$VLLM_PORT" \
  > $LOGS/trainer.log 2>&1 &
PIDS+=($!)
echo "== 收集器/trainer 已起(logs: $LOGS)"

# ── ② rollout 农场(前台循环 = 本脚本的生命线)────────────────────
SCRIPTS=(scripts/sim_scripts/s3_bakery.json
         scripts/sim_scripts/s2_rooftop.json
         scripts/sim_scripts/s5_rainstop.json)
PROMPTS=("晨光面包店" "天台魔术师" "雨停之前")
i=0
while true; do
  idx=$(( i % ${#SCRIPTS[@]} + 1 ))
  ADAPTER=$(cat $RL/state/active_adapter.txt 2>/dev/null || true)
  MODEL_NAME=${ADAPTER:-brain}
  export MAESTRO_POLICY_VERSION=$(cat $RL/state/policy_version.txt \
                                  2>/dev/null || echo 0)
  # 动态生成本轮配置:video_brain → vLLM 当前 adapter;其余 agent
  # 无 crew 条目 = 冻结在 models.llm(gpt-5.6-sol)—— 裁决 a。
  python - "$MODEL_NAME" <<'PY'
import sys, yaml, pathlib
name = sys.argv[1]
cfg = yaml.safe_load(open("configs/bailian.yaml"))
cfg.setdefault("models", {}).setdefault("crew", {})["video_brain"] = {
    "name": "vllm", "base_url": "http://localhost:8000/v1",
    "model": name, "max_tokens": 4096, "api_key": "dummy"}
pathlib.Path("rl/configs/_bailian_rl.generated.yaml").write_text(
    yaml.safe_dump(cfg, allow_unicode=True))
PY
  echo "== rollout #$i script=${SCRIPTS[$idx]} policy=$MODEL_NAME" \
       "(v$MAESTRO_POLICY_VERSION)"
  python scripts/test_window_movie.py \
    --config rl/configs/_bailian_rl.generated.yaml \
    --screenplay "${SCRIPTS[$idx]}" --prompt "${PROMPTS[$idx]}" \
    --rl-group $RL_GROUP --n-candidates 1 --max-turns 1 \
    >> $LOGS/rollout.log 2>&1
  echo "== rollout #$i exit=$?"
  i=$(( i + 1 ))
done
