#!/usr/bin/env bash
# (bash 兼容;zsh 也能跑 —— 服务器常无 zsh,2026-08-12 实报修复)
# ══════════════════════════════════════════════════════════════════════
#  Maestro brain GRPO 一键训练(semi-online;2026-08-10 用户令)
#
#    bash rl/run_grpo.sh            # 全链:vLLM 策略 + rollout 农场 +
#                                  # 收集器 + trainer(需 GPU 机)
#    bash rl/run_grpo.sh --smoke    # 无 GPU 自检:数据流/分组/advantage
#    bash rl/run_grpo.sh --fresh    # 全新训练:清零状态+开跑标记后继续起链
#    bash rl/run_grpo.sh --stop     # 一键收摊:清杀四类进程(含孤儿 vLLM)
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
#   WANDB_API_KEY=...            # wandb 密钥(2026-08-14 用户提供,
#   WANDB_BASE_URL=https://api.wandb.ai      # 写在 .env,不进 git)
#   WANDB_ENTITY=1120230293-nankai-university
#   WANDB_PROJECT=VideoAgent
#   WANDB_MODE=online            # 服务器连不上 api.wandb.ai 就改
#                                # offline,之后 wandb sync 补传
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

# ── --stop:一键收摊(2026-08-18 实报:vLLM 是上一次启动留下的
#    "孤儿"时不在本次 trap 的名单里,Ctrl-C 杀不到它)────────────────
if [[ "${1:-}" == "--stop" ]]; then
  echo "== STOP:清杀全部训练进程"
  pkill -f "test_window_movie.py"  2>/dev/null && echo "  rollout ✓"
  pkill -f "watch_online.py"       2>/dev/null && echo "  collector ✓"
  pkill -f "train_online.py"       2>/dev/null && echo "  trainer ✓"
  pkill -f "vllm serve"            2>/dev/null && echo "  vllm ✓"
  sleep 3
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -f "vllm.entrypoints" 2>/dev/null
  # 兜底:按【显卡占用 PID】清(2026-08-18 实报:vLLM 的 worker 子
  # 进程名不含 "vllm serve",按名清杀漏网,显存不放)
  if command -v nvidia-smi >/dev/null; then
    GPIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
            2>/dev/null | tr -d " ")
    if [[ -n "$GPIDS" ]]; then
      echo "== 显卡残留 PID:$GPIDS(kill -9)"
      echo "$GPIDS" | xargs -r kill -9 2>/dev/null
      sleep 2
    fi
    nvidia-smi --query-compute-apps=pid,used_memory \
      --format=csv,noheader 2>/dev/null | grep . \
      && echo "⚠️ 仍有占卡进程,手动 nvidia-smi 核查" \
      || echo "== 显存已清空"
  fi
  echo "== 进程残留:"; ps aux | grep -E "vllm|train_online|watch_online|test_window_movie" | grep -v grep || echo "  (干净)"
  exit 0
fi

# ── --fresh:全新训练总开关(2026-08-18 用户令:系统默认断点续跑,
#    这个开关一键归零)——清收集台账/书签/版本计数/旧 checkpoint,
#    杀掉带旧 adapter 的 vLLM,落"开跑标记"(收集器只收标记之后的
#    rollout,旧片彻底隔离)─────────────────────────────────────────
if [[ "${1:-}" == "--fresh" ]]; then
  shift
  echo "== FRESH:清零训练状态"
  rm -f rl/data/groups_online.jsonl rl/data/.watch_state.json
  rm -rf rl/state rl/ckpt
  mkdir -p rl/state
  date "+movie_%Y%m%d_%H%M%S" > rl/state/session_start
  pkill -f "vllm serve" 2>/dev/null && sleep 5
  echo "== FRESH 完成,标记 $(cat rl/state/session_start)"
fi

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
  # VLLM_ALLOW_RUNTIME_LORA_UPDATING:不开它 /v1/load_lora_adapter
  # 直接 4xx,热载永败(2026-08-18 实报根因)
  CUDA_VISIBLE_DEVICES=$VLLM_GPUS \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
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
  --vllm-url "http://localhost:$VLLM_PORT" --wandb \
  > $LOGS/trainer.log 2>&1 &
PIDS+=($!)
echo "== 收集器/trainer 已起(logs: $LOGS)"

# ── ② rollout 农场(前台循环 = 本脚本的生命线)────────────────────
# 任务来源 = rl/configs/task_pool.yaml(2026-08-14:剧本/idea 双制式
# 加权轮转;确定性调度 = 按迭代序号取模,断点续跑不乱序)
TASK_POOL=${TASK_POOL:-rl/configs/task_pool.yaml}
i=0
while true; do
  ADAPTER=$(cat $RL/state/active_adapter.txt 2>/dev/null || true)
  MODEL_NAME=${ADAPTER:-brain}
  export MAESTRO_POLICY_VERSION=$(cat $RL/state/policy_version.txt \
                                  2>/dev/null || echo 0)
  # 动态生成本轮配置 + 从任务池取本轮任务(加权轮转)
  TASK_JSON=$(python - "$MODEL_NAME" "$RL_BASE_CONFIG" "$TASK_POOL" "$i" <<'PY'
import sys, yaml, pathlib, json
name, cfg_p, pool_p, it = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = yaml.safe_load(open(cfg_p))
cfg.setdefault("models", {}).setdefault("crew", {})["video_brain"] = {
    "name": "vllm", "base_url": "http://localhost:8000/v1",
    "model": name, "max_tokens": 4096, "api_key": "dummy"}
pathlib.Path("rl/configs/_bailian_rl.generated.yaml").write_text(
    yaml.safe_dump(cfg, allow_unicode=True))
pool = yaml.safe_load(open(pool_p))
mix = pool.get("mix", {})
sw, iw = int(mix.get("screenplay_weight", 3)), int(mix.get("idea_weight", 2))
sps, ideas = pool.get("screenplays", []), pool.get("ideas", [])
# 确定性调度表:每周期 sw 个剧本位 + iw 个 idea 位,组内各自轮转
cycle = ["s"] * sw + ["i"] * iw
pos = it % len(cycle)
kind = cycle[pos] if (sps or ideas) else "s"
if kind == "i" and not ideas: kind = "s"
if kind == "s" and not sps: kind = "i"
# 该制式的【累计第几次】= 完整周期数×每周期名额 + 本周期内已过名额
# (修:旧版按周期号取游标,同周期内连抽同一任务)
n_cyc = it // len(cycle)
if kind == "s":
    k = n_cyc * sw + cycle[:pos].count("s")
    e = sps[k % len(sps)]
    print(json.dumps({"mode": "screenplay", "file": e["file"],
                      "prompt": e["prompt"]}, ensure_ascii=False))
else:
    k = n_cyc * iw + cycle[:pos].count("i")
    print(json.dumps({"mode": "idea", "prompt": ideas[k % len(ideas)]},
                     ensure_ascii=False))
PY
)
  TASK_MODE=$(echo "$TASK_JSON" | python -c "import sys,json;print(json.load(sys.stdin)['mode'])")
  TASK_PROMPT=$(echo "$TASK_JSON" | python -c "import sys,json;print(json.load(sys.stdin)['prompt'])")
  echo "== rollout #$i mode=$TASK_MODE prompt=$TASK_PROMPT" \
       "policy=$MODEL_NAME (v$MAESTRO_POLICY_VERSION)"
  if [[ "$TASK_MODE" == "screenplay" ]]; then
    TASK_FILE=$(echo "$TASK_JSON" | python -c "import sys,json;print(json.load(sys.stdin)['file'])")
    python scripts/test_window_movie.py \
      --config rl/configs/_bailian_rl.generated.yaml \
      --screenplay "$TASK_FILE" --prompt "$TASK_PROMPT" \
      --rl-group $RL_GROUP --n-candidates 1 --max-turns 1 \
      >> $LOGS/rollout.log 2>&1
  else
    python scripts/test_window_movie.py \
      --config rl/configs/_bailian_rl.generated.yaml \
      --prompt "$TASK_PROMPT" \
      --rl-group $RL_GROUP --n-candidates 1 --max-turns 1 \
      >> $LOGS/rollout.log 2>&1
  fi
  echo "== rollout #$i exit=$?"
  i=$(( i + 1 ))
done
