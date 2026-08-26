"""rl/local 的超参与装配(2026-08-21 用户裁决:本地推理 + 多流并行)。

与 rl/env 的关系:本目录【只新增】,不改 rl/env 的既有行为 ——
rl/env/loop.py 的三个钩子默认 None,不给就走原来的 vLLM 路径。

路径约定(两条文件通道,见 DESIGN 注释):
  rl/data/queue/*.pt          流 → 训练器的组队列(消费即删)
  rl/data/queue/claimed/      训练器认领中(崩溃可恢复)
  rl/state/live_adapter/      训练器 → 流的 LoRA 广播(VERSION + vN/)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RL = REPO / "rl"
QUEUE_DIR = RL / "data/queue"
CLAIMED_DIR = QUEUE_DIR / "claimed"
LIVE_ADAPTER = RL / "state/live_adapter"
VERSION_FILE = LIVE_ADAPTER / "VERSION"


@dataclass
class HParams:
    # ── 底座与 LoRA ───────────────────────────────────────────────
    base_model: str = ""
    rank: int = 64
    alpha: int = 128
    # 先收窄到注意力四件套:vLLM/PEFT 双侧支持最扎实,故障面最小;
    # 跑通后再考虑放开到 all-linear(2026-08-21 用户裁决)
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # ── 采样 ─────────────────────────────────────────────────────
    group: int = 4                 # K:每镜的候选数 = 一个组
    temp_main: float = 0.7         # 主干候选(v0)
    temp_branch: float = 0.9       # 分支候选(v1..K-1)
    max_new_tokens: int = 1024
    # 核采样必须关:否则真实行为分布 ≠ softmax(logits/T),
    # 记下来的 old_logprob 就是错的,重要性比值跟着错(用户裁决①)
    top_p: float = 1.0
    top_k: int = 0
    enable_thinking: bool = False  # 用户裁决②:与冻结 agent 一致

    # ── 训练 ─────────────────────────────────────────────────────
    lr: float = 1e-5               # verl 官方:LoRA 应比全参高一个数量级
    clip_low: float = 0.2
    clip_high: float = 0.3         # 非对称裁剪(DAPO clip-higher)
    kl_coef: float = 0.001         # k3 估计量,作损失项、不进奖励
    grad_clip: float = 1.0
    # 14B/40 层在 7000 token 上的反向激活约 50GB;开检查点后只存层边界
    # (~4GB),反向多算约 30%。训练器本来就在空等视频,这笔交换近乎白送。
    grad_checkpoint: bool = True
    broadcast_every: int = 4       # 用户裁决④:每 N 次 optimizer step 广播
    staleness_max: int = 5
    keep_adapters: int = 3         # live 目录保留几代(滚动删除)
    # 峰值回滚的物质基础(2026-08-25 用户裁决):每 N 版把 adapter 永久
    # 归档到 archive/,不受滚动删除影响 —— 否则奖励峰值那一版早被删了,
    # "按峰值回滚"无版可回。LoRA 一版约 170MB,每 20 版归档成本可忽略。
    archive_every: int = 20        # 0 = 关闭归档
    poll_s: float = 5.0

    # ── 流 ───────────────────────────────────────────────────────
    workers: int = 1               # 总流数(任务池按 worker 错开的除数)
    worker_id: int = 0
    trunk_select: str = "greedy"   # 用户裁决③(备选 "softmax:0.1")
    task_pool: str = str(RL / "configs/task_pool_train100.yaml")
    env_config: str = str(RL / "configs/server_bailian_qwen.yaml")
    out_root: str = field(default_factory=lambda: os.environ.get(
        "MAESTRO_OUTPUT_ROOT", str(REPO / "outputs")))


def add_args(ap: argparse.ArgumentParser) -> None:
    d = HParams()
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", ""))
    ap.add_argument("--rank", type=int, default=d.rank)
    ap.add_argument("--alpha", type=int, default=d.alpha)
    ap.add_argument("--target-modules", default=",".join(d.target_modules))
    ap.add_argument("--group", type=int, default=d.group)
    ap.add_argument("--temp-main", type=float, default=d.temp_main)
    ap.add_argument("--temp-branch", type=float, default=d.temp_branch)
    ap.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    ap.add_argument("--thinking", action="store_true",
                    help="给策略开思考模式(默认关)")
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--clip-low", type=float, default=d.clip_low)
    ap.add_argument("--clip-high", type=float, default=d.clip_high)
    ap.add_argument("--kl-coef", type=float, default=d.kl_coef)
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="关掉梯度检查点(显存换速度;14B 单卡请勿关)")
    ap.add_argument("--broadcast-every", type=int, default=d.broadcast_every)
    ap.add_argument("--archive-every", type=int, default=d.archive_every,
                    help="每 N 版永久归档一份 adapter(0 关闭)")
    ap.add_argument("--staleness-max", type=int, default=d.staleness_max)
    ap.add_argument("--workers", type=int, default=d.workers)
    ap.add_argument("--worker", type=int, default=d.worker_id)
    ap.add_argument("--trunk-select", default=d.trunk_select)
    ap.add_argument("--task-pool", default=d.task_pool)
    ap.add_argument("--config", dest="env_config", default=d.env_config)
    ap.add_argument("--out-root", default=d.out_root)
    ap.add_argument("--wandb", action="store_true")


def from_args(a) -> HParams:
    return HParams(
        base_model=a.base_model, rank=a.rank, alpha=a.alpha,
        target_modules=tuple(x.strip() for x in a.target_modules.split(",")
                             if x.strip()),
        group=a.group, temp_main=a.temp_main, temp_branch=a.temp_branch,
        max_new_tokens=a.max_new_tokens, enable_thinking=bool(a.thinking),
        lr=a.lr, clip_low=a.clip_low, clip_high=a.clip_high,
        kl_coef=a.kl_coef, grad_checkpoint=not a.no_grad_checkpoint,
        broadcast_every=a.broadcast_every, archive_every=a.archive_every,
        staleness_max=a.staleness_max, workers=a.workers, worker_id=a.worker,
        trunk_select=a.trunk_select, task_pool=a.task_pool,
        env_config=a.env_config, out_root=a.out_root)


# ── 外部客户端装配(冻结 agent / 可灵 / t2i / 图像编辑 / VLM)──────────
# 与 rl/env/rollout.py 同源的选择,只是这里【不建 vLLM 策略客户端】。
_MAAS = ("https://ws-ox5q19lbmn2u1drg.cn-beijing.maas.aliyuncs.com"
         "/compatible-mode/v1")
_DAS = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_IDE = "https://idealab-external.alibaba-inc.com/api/openai/v1"
_LLM_BASES = {"qwen-maas": (_MAAS, "DASHSCOPE_API_KEY"),
              "qwen": (_DAS, "QWEN_API_KEY")}
_VLM_BASES = {"idealab": (_IDE, "IDEALAB_API_KEY"),
              "idealab-gemini": (_IDE, "IDEALAB_API_KEY"),
              "qwen-vl": (_DAS, "QWEN_API_KEY")}


def build_externals(env_config: str, call_log):
    """→ (frozen_llm, video_gen, image_edit, mllm, models_cfg)。
    缺 key 直接抛,不静默降级。"""
    import sys
    sys.path.insert(0, str(RL))
    from env.clients import (EnvVLM, KlingClient, TextLLM,
                             WaveSpeedImageEdit, WaveSpeedT2I)
    from env.config import load_yaml

    cfg = load_yaml(Path(env_config))
    models = cfg.get("models", {})
    llm_spec = models.get("llm") or {}
    mllm_spec = models.get("mllm") or {}
    vg_spec = models.get("video_gen") or {}

    base, ev = _LLM_BASES.get(str(llm_spec.get("name", "qwen")),
                              _LLM_BASES["qwen"])
    key = (llm_spec.get("api_key") or os.getenv(ev)
           or os.getenv("DASHSCOPE_API_KEY") or "")
    if not key:
        raise RuntimeError(f"冻结 LLM 缺 key:在 .env 里设 {ev}")
    frozen = TextLLM(llm_spec.get("base_url") or base,
                     llm_spec.get("model", "qwen-max"), key,
                     timeout=float(llm_spec.get("timeout", 600)),
                     max_tokens=int(llm_spec.get("max_tokens", 16384)),
                     extra_body=llm_spec.get("extra_body"),
                     log=call_log, name="frozen")

    ds_key = os.getenv("DASHSCOPE_API_KEY", "")
    ws_key = os.getenv("WAVESPEED_API_KEY", "")
    if not ds_key or not ws_key:
        raise RuntimeError("缺 DASHSCOPE_API_KEY / WAVESPEED_API_KEY")
    t2i = WaveSpeedT2I(ws_key, log=call_log)
    video_gen = KlingClient(ds_key, mode=str(vg_spec.get("mode", "std")),
                            aspect_ratio=str(vg_spec.get("aspect_ratio",
                                                         "16:9")),
                            log=call_log, t2i=t2i)
    image_edit = WaveSpeedImageEdit(ws_key, log=call_log)
    vbase, vev = _VLM_BASES.get(str(mllm_spec.get("name", "idealab-gemini")),
                                _VLM_BASES["idealab-gemini"])
    mllm = EnvVLM(mllm_spec.get("base_url") or vbase,
                  mllm_spec.get("model", "gemini-3.1-pro-preview"),
                  mllm_spec.get("api_key") or os.getenv(vev) or ds_key,
                  log=call_log)
    return frozen, video_gen, image_edit, mllm, models
