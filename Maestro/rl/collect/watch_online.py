"""在线收集器(2026-08-19 瘦身:评审整体移进 rl/env 采样端 —— skill
判官在 rollout 里择主干并把 reward 内联进 rl_steps.jsonl;这里只做
【聚合】:增量扫 outputs/movie_* 的组记录 → 追加 rl/data/
groups_online.jsonl,断点书签 + 开跑标记隔离旧片)。

用法:python rl/collect/watch_online.py [--once] [--poll 120]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
STATE = REPO / "rl/data/.watch_state.json"
OUT = REPO / "rl/data/groups_online.jsonl"


def collect_run(run_dir: Path, seen: set) -> list:
    p = run_dir / "rl_steps.jsonl"
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text(errors="replace").splitlines()):
        key = f"{run_dir.name}:{i}"
        if key in seen:
            continue
        try:
            g = json.loads(line)
        except Exception:
            continue
        if g.get("kind") != "condition_group":
            continue
        samples = g.get("samples") or []
        # 采样端必须已判分(reward 内联);缺 reward 的组是旧格式/半截
        # 记录 —— 响亮跳过,绝不编分
        if len(samples) < 2 or any(s.get("reward") is None
                                   for s in samples):
            print(f"[collector] skip {key}: missing rewards "
                  f"(old-format or unjudged group)", flush=True)
            seen.add(key)
            continue
        out.append({**g, "_key": key})
        seen.add(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--since-marker",
                    default=str(REPO / "rl/state/session_start"),
                    help="开跑标记文件(--fresh 写入;存在则只收目录名"
                         "晚于标记值的 run —— 旧 rollout 彻底隔离)")
    args = ap.parse_args()
    cutoff = ""
    mk = Path(args.since_marker)
    if mk.exists():
        cutoff = mk.read_text().strip()
        print(f"[collector] fresh marker: only runs after {cutoff}",
              flush=True)
    out_path = Path(args.out)
    state_path = Path(args.state)
    seen = set()
    if state_path.exists():
        try:
            seen = set(json.loads(state_path.read_text()))
        except Exception:
            seen = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        new = 0
        for run_dir in sorted(Path(args.outputs).glob("movie_*")):
            if cutoff and run_dir.name <= cutoff:
                continue                    # 开跑标记之前的旧 rollout
            groups = collect_run(run_dir, seen)
            if groups:
                with open(out_path, "a") as f:
                    for g in groups:
                        f.write(json.dumps(g, ensure_ascii=False,
                                           default=str) + "\n")
                new += len(groups)
        if new:
            state_path.write_text(json.dumps(sorted(seen)))
            print(f"[collector] +{new} groups → {out_path}", flush=True)
        if args.once:
            break
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
