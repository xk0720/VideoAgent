"""GRPO 训练骨架(v1:只训 generation-condition action;Qwen3 底座)。

依赖(训练机安装,不进主仓依赖):
  pip install trl transformers accelerate peft datasets

流程:
  steps.jsonl → 组装 (prompt, response, reward) → 组内归一 advantage
  → GRPO(trl.GRPOTrainer;无 critic)。
在线模式:--watch 让本脚本轮询 steps.jsonl 增量(rollout 农场持续追加),
每攒够 batch 触发一步更新;每 M 步存 checkpoint,vLLM 端热载
(vllm serve --served-model-name brain --model <ckpt> 重启或热切)。

prompt 还原:condition 决策的完整上下文 = 技能全文 + task JSON ——
与管线 `_decide` 的拼装严格一致(复用同一函数,保证训练分布 = 推理
分布;见 build_prompt)。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "rl"))


def build_prompt(step: dict) -> str:
    """训练用 prompt = 采样用 prompt(单一事实源:rl/env/skills.py;
    2026-08-19 用户令 rl/ 自包含,不再 import 主管线)。"""
    from env.skills import skill_body
    skill = skill_body("window_generation")
    task = dict(step.get("context") or {})
    task["menu"] = step.get("menu")
    return (skill + "\n\nTHIS TASK (JSON):\n"
            + json.dumps(task, ensure_ascii=False))


def group_key(step: dict) -> str:
    """GRPO 组:同 (剧本 run 系列, shot_idx, junction kind) —— 组内
    advantage = r − 组均值,消题目难度方差。"""
    ctx = step.get("context") or {}
    jk = ""
    j = ctx.get("junction")
    if isinstance(j, dict):
        jk = str(j.get("junction_kind") or "")
    label = ""
    if isinstance(ctx.get("shot"), dict):
        label = str(ctx["shot"].get("label") or "")
    return f"{label}|{jk}"


def load_groups(path: Path, min_group: int = 2) -> list:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("kind") == "generation-condition"]
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    out = []
    for k, g in groups.items():
        if len(g) < min_group:
            continue
        mean = sum(x["reward"] for x in g) / len(g)
        for x in g:
            x["advantage"] = round(x["reward"] - mean, 4)
        out.append(g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "rl/data/steps.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default=str(REPO / "rl/ckpt"))
    ap.add_argument("--dry-run", action="store_true",
                    help="只做分组与统计,不起训练(无 GPU 环境自检)")
    args = ap.parse_args()

    groups = load_groups(Path(args.data))
    n = sum(len(g) for g in groups)
    print(f"groups={len(groups)} usable steps={n}")
    if args.dry_run or not groups:
        for g in groups[:3]:
            print("--", group_key(g[0]), [x["advantage"] for x in g])
        return 0

    # ── 真训练(需 GPU + trl)──
    from datasets import Dataset          # noqa: deferred heavy imports
    from trl import GRPOConfig, GRPOTrainer

    flat = [x for g in groups for x in g]
    ds = Dataset.from_list([{
        "prompt": build_prompt(x),
        "completion": x.get("raw") or "",
        "reward": x["reward"],
    } for x in flat])

    def reward_fn(completions, **kw):
        # 在线采样模式:对新采样的 completion 即时打 format 分;
        # task 分需 rollout 回填 —— 纯离线复放时直接用记录 reward。
        sys.path.insert(0, str(REPO / "rl"))
        from reward.reward_fn import format_reward
        return [format_reward({"raw": c, "usable": None,
                               "parsed": None})[0] for c in completions]

    cfg = GRPOConfig(output_dir=args.out, per_device_train_batch_size=2,
                     num_generations=4, max_prompt_length=8192,
                     max_completion_length=1024, logging_steps=5,
                     save_steps=50)
    trainer = GRPOTrainer(model=args.model, args=cfg,
                          reward_funcs=reward_fn, train_dataset=ds)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
