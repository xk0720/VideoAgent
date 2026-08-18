"""在线收集器:增量扫 outputs/movie_* 的 rl_steps.jsonl(RL 组记录)
→ 补 reward → 追加 rl/data/groups_online.jsonl。

reward(组内成员,semi-online 版):
  r = 0.2·format + 0.8·weighted_total
  format 按 brain_calls 里该 decision 的 usable/via 判(fallback = brain
  回复不可用 = 0 分;llm = 过闸 = 1 —— 管线闸门就是判分器);
  组 advantage 由 trainer 计算,这里只记原始 r。

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

W_FORMAT, W_TASK = 0.2, 0.8


def _task_score(s: dict) -> float:
    """reward v2(2026-08-13 用户裁决):只取【看片维】——
    r_task = 0.5·m1_semantic + 0.5·p1_physics。
    剔除:id1/m2(结构代理 —— 只看挂没挂图/钉没钉帧,可被策略选择
    白刷分,组内对比会放大该偏差)、p2(用户令:去掉)、m5/aesthetic
    (常量/读计划)。老记录无分维数据 → 退回 weighted_total。"""
    m = s.get("metrics") or {}
    if "m1_semantic" in m and "p1_physics" in m:
        return 0.5 * float(m["m1_semantic"]) +             0.5 * float(m["p1_physics"])
    return float(s.get("weighted_total") or 0.0)


def _brain_index(run_dir: Path) -> dict:
    idx = {}
    p = run_dir / "brain_calls.jsonl"
    if not p.exists():
        return idx
    for line in p.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        did = d.get("decision_id")
        if did and d.get("stage") == "window/generation-condition":
            idx[did] = d
    return idx


def collect_run(run_dir: Path, seen: set) -> list:
    p = run_dir / "rl_steps.jsonl"
    if not p.exists():
        return []
    bidx = None
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
        # 组记录自包含(context/menu/completion 都在记录里);
        # brain_calls 仅作 raw 补充(有则富化,无则不缺)。
        if bidx is None:
            bidx = _brain_index(run_dir)
        samples = []
        for s_ in g.get("samples") or []:
            call = bidx.get(s_.get("decision_id")) or {}
            r_fmt = 1.0 if s_.get("via") == "llm" else 0.0
            wt = _task_score(s_)
            r = round(W_FORMAT * r_fmt + W_TASK * wt, 4)
            samples.append({**s_, "r_format": r_fmt,
                            "reward": r,
                            "raw": call.get("raw")
                                   or s_.get("completion"),
                            "usable": call.get("usable")})
        out.append({**g, "samples": samples, "_key": key})
        seen.add(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--state", default=str(STATE))
    args = ap.parse_args()
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
