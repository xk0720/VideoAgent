#!/usr/bin/env python
"""RL rollout 入口(2026-08-19 用户令:agent loop 全部在 rl/ 下,
训练=生产完全同构)。

一次调用 = 一条轨迹:任务(剧本文件或 idea)→ 生产同构窗口管线
(分镜/资产/空间圣经/三叉分诊/image plan/出门链/条件执行)+ 每镜
K 组采样 + skill 判官择主干 → outputs/movie_*/rl_steps.jsonl
(reward 已内联,收集器只聚合)。

用法(run_grpo.sh 的农场就是这么调的):
  python rl/env/rollout.py --config rl/configs/server_bailian_qwen.yaml \
      --screenplay rl/trainset/screenplays/xx.screenplay.json \
      --prompt "雪夜末班车" --group 4 \
      --policy-base http://localhost:8000/v1 --policy-model <name>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parent.parent
REPO = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT))

from env.config import load_dotenv, load_yaml                # noqa: E402

load_dotenv(REPO / ".env")

from env.clients import (CallLog, EnvVLM, KlingClient, TextLLM,  # noqa: E402
                         WaveSpeedImageEdit, WaveSpeedT2I)
from env.loop import build_judges, run_episode               # noqa: E402

_MAAS = ("https://ws-ox5q19lbmn2u1drg.cn-beijing.maas.aliyuncs.com"
         "/compatible-mode/v1")
_DAS = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_IDE = "https://idealab-external.alibaba-inc.com/api/openai/v1"
_LLM_BASES = {"qwen-maas": (_MAAS, "DASHSCOPE_API_KEY"),
              "qwen": (_DAS, "QWEN_API_KEY")}
_VLM_BASES = {"idealab": (_IDE, "IDEALAB_API_KEY"),
              "idealab-gemini": (_IDE, "IDEALAB_API_KEY"),
              "qwen-vl": (_DAS, "QWEN_API_KEY")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "rl/configs/server_bailian_qwen"
                                       ".yaml"))
    ap.add_argument("--screenplay", default="",
                    help="剧本契约 JSON({'content': …})或纯文本;"
                         "空 = 用 --prompt 当 idea(§A0 现写剧本)")
    ap.add_argument("--prompt", default="", help="idea / 任务一句话")
    ap.add_argument("--group", type=int,
                    default=int(os.environ.get("RL_GROUP", "4")))
    ap.add_argument("--rl-temperature", type=float, default=0.9)
    ap.add_argument("--audio", action="store_true",
                    help="对白原生音频(生产 RL 农场默认关,与旧跑法同)")
    ap.add_argument("--out-root",
                    default=os.environ.get("MAESTRO_OUTPUT_ROOT",
                                           str(REPO / "outputs")))
    ap.add_argument("--policy-base",
                    default=os.environ.get("RL_POLICY_BASE",
                                           "http://localhost:8000/v1"))
    ap.add_argument("--policy-model",
                    default=os.environ.get("RL_POLICY_MODEL", ""))
    args = ap.parse_args()

    screenplay = None
    if args.screenplay:
        p = Path(args.screenplay)
        if p.suffix == ".json":
            screenplay = str(json.loads(p.read_text()).get("content",
                                                           "")).strip()
        else:
            screenplay = p.read_text().strip()
        screenplay = screenplay or None
    task_text = args.prompt.strip()
    if not (task_text or screenplay):
        print("empty task: give --screenplay or --prompt", flush=True)
        return 2
    if not args.policy_model:
        print("missing --policy-model (or $RL_POLICY_MODEL)", flush=True)
        return 2

    cfg = load_yaml(Path(args.config))
    models = cfg.get("models", {})
    llm_spec = models.get("llm") or {}
    mllm_spec = models.get("mllm") or {}
    vg_spec = models.get("video_gen") or {}
    max_shots = int((cfg.get("plan") or {}).get("max_shots", 12))

    run_dir = Path(args.out_root) / time.strftime("movie_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    call_log = CallLog(run_dir / "env_calls.jsonl")

    base, ev = _LLM_BASES.get(str(llm_spec.get("name", "qwen")),
                              _LLM_BASES["qwen"])
    frozen_key = (llm_spec.get("api_key") or os.getenv(ev)
                  or os.getenv("DASHSCOPE_API_KEY") or "")
    if not frozen_key:
        print(f"missing frozen-LLM key: set {ev} in .env", flush=True)
        return 2
    frozen_llm = TextLLM(llm_spec.get("base_url") or base,
                         llm_spec.get("model", "qwen-max"), frozen_key,
                         timeout=float(llm_spec.get("timeout", 600)),
                         max_tokens=int(llm_spec.get("max_tokens",
                                                     16384)),
                         extra_body=llm_spec.get("extra_body"),
                         log=call_log, name="frozen")
    policy = TextLLM(args.policy_base, args.policy_model, "dummy",
                     timeout=600, max_tokens=8192, log=call_log,
                     name="policy")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    wavespeed_key = os.getenv("WAVESPEED_API_KEY", "")
    if not dashscope_key or not wavespeed_key:
        print("missing DASHSCOPE_API_KEY / WAVESPEED_API_KEY in .env",
              flush=True)
        return 2
    t2i = WaveSpeedT2I(wavespeed_key, log=call_log)
    video_gen = KlingClient(dashscope_key,
                            mode=str(vg_spec.get("mode", "std")),
                            aspect_ratio=str(vg_spec.get("aspect_ratio",
                                                         "16:9")),
                            log=call_log, t2i=t2i)
    image_edit = WaveSpeedImageEdit(wavespeed_key, log=call_log)
    vbase, vev = _VLM_BASES.get(str(mllm_spec.get("name",
                                                  "idealab-gemini")),
                                _VLM_BASES["idealab-gemini"])
    vlm_key = (mllm_spec.get("api_key") or os.getenv(vev)
               or os.getenv("DASHSCOPE_API_KEY") or "")
    mllm = EnvVLM(mllm_spec.get("base_url") or vbase,
                  mllm_spec.get("model", "gemini-3.1-pro-preview"),
                  vlm_key, log=call_log)
    judges = build_judges(models, RL_ROOT / "logs/judge_calls.jsonl")

    print(f"[env] run={run_dir.name} "
          f"task={(screenplay or task_text)[:40]!r} "
          f"policy={args.policy_model} "
          f"(v{os.environ.get('MAESTRO_POLICY_VERSION', '0')}) "
          f"K={args.group}", flush=True)
    res = run_episode(task_text=task_text, screenplay=screenplay,
                      run_dir=run_dir, frozen_llm=frozen_llm,
                      policy=policy, video_gen=video_gen,
                      image_edit=image_edit, mllm=mllm, judges=judges,
                      group=args.group,
                      rl_temperature=args.rl_temperature,
                      max_shots=max_shots, enable_audio=args.audio)
    print(f"[env] episode done: {res}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
