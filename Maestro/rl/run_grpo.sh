#!/usr/bin/env bash
# (bash 兼容;zsh 也能跑 —— 服务器常无 zsh,2026-08-12 实报修复)
# ══════════════════════════════════════════════════════════════════════
#  Maestro brain GRPO 一键训练(semi-online;2026-08-10 用户令)
#
#    bash rl/run_grpo.sh            # 全链:vLLM 策略 + rollout 农场 +
#                                  # 收集器 + trainer(需 GPU 机)
#    bash rl/run_grpo.sh --smoke    # 无 GPU 自检:数据流/分组/advantage
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

# ══ 密钥纪律(2026-08-13 用户令 + 排障实锤)═══════════════════════
# key 【只】从仓库根 Maestro/.env 读取 —— 不在脚本里写默认、不 export
# 补齐、不互补。先清掉壳里的残留(bashrc 里 export 过的旧 key 曾顶掉
# .env,可灵报"未开通"排查半天),再载 .env,来源唯一、可审计。
# .env 需含三行(同一把百炼 key 写两遍也行):
#   DASHSCOPE_API_KEY=sk-xxx     # 可灵视频
#   QWEN_API_KEY=sk-xxx          # qwen-max 冻结 agent + omni 评审
#   WAVESPEED_API_KEY=ws-xxx     # t2i/图像编辑
unset DASHSCOPE_API_KEY QWEN_API_KEY WAVESPEED_API_KEY
set -a; source .env 2>/dev/null; set +a

# 本地策略权重 —— vLLM 与 trainer 共用一个值;服务器写本地目录
#   (如 /data/models/Qwen3-8B,自动 HF 离线+完整性预检),联网机可写
#   HF hub id
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}

# 运行旋钮
VLLM_PORT=${VLLM_PORT:-8000}          # 策略服务端口
RL_GROUP=${RL_GROUP:-4}               # 每镜组采样数 K(GRPO 组大小)
# GPU 划分(2026-08-12,单机八卡):推理/训练物理隔离 —— vLLM 会
# 预占它可见卡约 90% 显存,不隔离必撞 trainer。8B 底座默认 1+1 卡
# (瓶颈在可灵 API 不在 GPU);换大底座改这两个变量,vLLM 张量并行
# 数自动 = 卡数。
VLLM_GPUS=${VLLM_GPUS:-0}             # 推理卡:0 或 0,1
TRAIN_GPUS=${TRAIN_GPUS:-1}           # 训练卡:1 或 1,2,3
RL_BASE_CONFIG=${RL_BASE_CONFIG:-rl/configs/server_bailian_qwen.yaml}
                                      # RL 专属基配置(全百炼,与
                                      # configs/ 隔离,勿指回主配置)
# ═══════════════════════════════════════════════════════════════════
PIDS=()
# ${arr[@]+...} = set -u 下的空数组安全展开(bash 全版本;绝不 kill 0)
cleanup() { for p in ${PIDS[@]+"${PIDS[@]}"}; do kill "$p" 2>/dev/null; done; }
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
  # mock 各候选 m1/p1 全同 → 组内零优势被 trainer 正确弃组(这正是
  # reward v2 焊死结构代理刷分的证明)。补一个带真差异的合成组,
  # 专验 trainer 的分组/优势数学:
  python - "$SBX/groups.jsonl" <<'PY'
import json, sys
g = {"kind": "condition_group", "run": "synthetic", "shot_idx": 0,
     "label": "synthetic shot", "junction_kind": None,
     "policy_version": "1", "group_size": 3,
     "menu": [{"name": "t2v"}],
     "context": {"shot": {"label": "synthetic"}},
     "samples": [
       {"decision_id": f"d{i}", "via": "llm", "chosen": i == 0,
        "completion": '{"strategy": "t2v"}', "raw": '{"strategy": "t2v"}',
        "weighted_total": w, "reward": 0.2 + 0.8 * w,
        "metrics": {"m1_semantic": w, "p1_physics": w}}
       for i, w in enumerate([0.9, 0.5, 0.7])]}
open(sys.argv[1], "a").write(json.dumps(g) + "\n")
PY
  python rl/train/train_online.py --dry-run --data "$SBX/groups.jsonl"     | tee /tmp/rl_smoke_trainer.txt || exit 1
  grep -qE "usable groups=[1-9]" /tmp/rl_smoke_trainer.txt || { echo "❌ trainer 分组为 0"; exit 1; }
  python -m pytest tests/unit/test_rl_group.py -q || exit 1
  rm -rf "$SBX"
  echo "== SMOKE OK(采样→落盘→收集→分组→advantage 全链通;"
  echo "   真训练需 GPU 机跑本脚本无参形态)"
  exit 0
fi

# ── 预检(缺什么直说什么)──────────────────────────────────────────
fail() { echo "❌ $1"; exit 2; }
command -v vllm >/dev/null || fail "vllm 未安装:pip install vllm"
python -c "import torch, peft, transformers" 2>/dev/null \
  || fail "训练依赖缺失:pip install torch transformers peft"
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || fail "无 CUDA GPU(vLLM+LoRA 训练需要;Mac 请在 GPU 机上跑)"
# key 只认 .env(顶部已 unset 壳残留后 source):三把各自点名检查,
# 不做任何互补/export —— 少哪行就去 Maestro/.env 补哪行
[[ -n "${DASHSCOPE_API_KEY:-}" ]] \
  || fail "DASHSCOPE_API_KEY 缺失 —— 写进 Maestro/.env(可灵视频)"
[[ -n "${QWEN_API_KEY:-}" ]] \
  || fail "QWEN_API_KEY 缺失 —— 写进 Maestro/.env(qwen 冻结 agent+omni 评审;与 DASHSCOPE 同一把 key 也要写这一行)"
[[ -n "${WAVESPEED_API_KEY:-}" ]] \
  || fail "WAVESPEED_API_KEY 缺失 —— 写进 Maestro/.env(t2i/图像编辑)"
[[ -f "$RL_BASE_CONFIG" ]] || fail "RL 基配置缺失:$RL_BASE_CONFIG"
# 权重路径预检:BASE_MODEL 含 "/" 开头或 "./" = 本地目录 → 必须存在,
# 且切 HF 离线(服务器拉不到 hub;权重不齐当场报,不半夜才崩)
case "$BASE_MODEL" in
  /*|./*)
    [[ -d "$BASE_MODEL" ]] || fail "本地权重目录不存在:$BASE_MODEL"
    [[ -f "$BASE_MODEL/config.json" ]] \
      || fail "$BASE_MODEL 缺 config.json(不是完整 HF 权重目录)"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    echo "== 本地权重:$BASE_MODEL(HF 离线模式)"
    ;;
esac

# ── ① vLLM 策略服务 ──────────────────────────────────────────────────
if ! curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null; then
  echo "== 启动 vLLM($BASE_MODEL)"
  VLLM_TP=$(( $(echo "$VLLM_GPUS" | tr -cd "," | wc -c) + 1 ))
  CUDA_VISIBLE_DEVICES=$VLLM_GPUS \
  vllm serve "$BASE_MODEL" --served-model-name brain \
    --enable-lora --max-loras 4 --port $VLLM_PORT \
    --tensor-parallel-size $VLLM_TP \
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
python rl/collect/watch_online.py --judge > $LOGS/collector.log 2>&1 &
PIDS+=($!)
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
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
  idx=$(( i % ${#SCRIPTS[@]} ))
  ADAPTER=$(cat $RL/state/active_adapter.txt 2>/dev/null || true)
  MODEL_NAME=${ADAPTER:-brain}
  export MAESTRO_POLICY_VERSION=$(cat $RL/state/policy_version.txt \
                                  2>/dev/null || echo 0)
  # 动态生成本轮配置:video_brain → vLLM 当前 adapter;其余 agent
  # 无 crew 条目 = 冻结在 models.llm(gpt-5.6-sol)—— 裁决 a。
  python - "$MODEL_NAME" "$RL_BASE_CONFIG" <<'PY'
import sys, yaml, pathlib
name = sys.argv[1]
cfg = yaml.safe_load(open(sys.argv[2]))
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
