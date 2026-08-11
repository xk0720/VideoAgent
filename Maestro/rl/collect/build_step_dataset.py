"""S1 数据构建器:扫 outputs/movie_* → join → rl/data/steps.jsonl。

join 键 = decision_id:
  brain_calls.jsonl 里的决策记录(context/raw/parsed/usable)
  × repair/outcome(decision_id → tool/new_total/verifier_score)
  × shot_outcome(condition_decision_id → converged/repair_turns)
  × storyboard.json(weighted_total 终分/prompt_language/junction_meta)

产物一行 = 一个 step:
  {run, kind, decision_id, shot_idx, context, raw, parsed, usable,
   menu, slots, prompt_language, outcome{...}, reward, r_format, r_task,
   policy_model, ts}

用法:
  python rl/collect/build_step_dataset.py            # 扫全部 outputs
  python rl/collect/build_step_dataset.py --runs outputs/movie_X ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "rl"))

from reward.reward_fn import step_reward  # noqa: E402

# 2026-08-10 用户裁决:v1 只训 generation-condition 一个 action ——
# 其余 stage 不出样本,它们的产物本来就在 condition 调用的 context 里
# (junction/槽位清单/台账视图),即"长期记忆 + 当前 rollout 结果"。
_KIND_BY_STAGE = {
    "window/generation-condition": "generation-condition",
}


def _load_jsonl(p: Path) -> list:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def build_run(run_dir: Path) -> list:
    calls = _load_jsonl(run_dir / "brain_calls.jsonl")
    sb_path = run_dir / "storyboard.json"
    sb = {}
    if sb_path.exists():
        try:
            sb = json.loads(sb_path.read_text())
        except Exception:
            sb = {}
    entries = {e.get("shot_idx"): e for e in (sb.get("entries") or [])}

    # 结局索引:decision_id → repair outcome;condition_decision_id → 镜结局
    repair_out: dict = {}
    shot_out: dict = {}
    prev_total: dict = {}                 # shot_idx → 修复前分(Δ分用)
    for c in calls:
        st = c.get("stage")
        d = c if isinstance(c, dict) else {}
        body = d.get("parsed") if isinstance(d.get("parsed"), dict) else {}
        if st == "repair/outcome" or (
                "decision_id" in d and "outcome" in d and "tool" in d):
            rec = {k: d.get(k) for k in
                   ("tool", "outcome", "new_total", "verifier_score",
                    "shot_idx", "turn")}
            si = d.get("shot_idx")
            rec["prev_total"] = prev_total.get(si)
            if d.get("new_total") is not None:
                prev_total[si] = d.get("new_total")
            repair_out[d.get("decision_id")] = rec
        elif st == "window/shot_outcome" or "condition_decision_id" in d:
            shot_out[d.get("condition_decision_id")] = {
                k: d.get(k) for k in
                ("converged", "repair_turns", "stop_reason", "shot_idx")}

    steps = []
    for c in calls:
        stage = c.get("stage") or ""
        kind = _KIND_BY_STAGE.get(stage)
        if kind is None:
            continue
        ctx = c.get("context") or {}
        did = c.get("decision_id")
        si = (ctx.get("shot_idx") if isinstance(ctx, dict) else None)
        if si is None and isinstance(ctx.get("shot"), dict):
            lbl = str(ctx["shot"].get("label") or "")
            import re as _re
            m = _re.search(r"shot (\d+)", lbl)
            si = int(m.group(1)) - 1 if m else None
        entry = entries.get(si) or {}
        last_rev = (entry.get("reviews") or [{}])[-1]
        outcome = {
            "weighted_total": last_rev.get("weighted_total"),
            "status": entry.get("status"),
        }
        if kind == "repair" and did in repair_out:
            outcome.update(repair_out[did])
            outcome["verifier_score"] = repair_out[did].get(
                "verifier_score")
        so = shot_out.get(did)
        if so:
            outcome.update(so)
        elif entry:
            outcome["converged"] = entry.get("status") == "verified"
            outcome["repair_turns"] = len(entry.get("repair_actions")
                                          or [])
        if kind == "image-plan":
            outcome["degraded_from"] = entry.get("plan_degraded_from")
        step = {
            "run": run_dir.name,
            "kind": kind,
            "decision_id": did,
            "shot_idx": si,
            "context": ctx,
            "raw": c.get("raw"),
            "parsed": c.get("parsed"),
            "usable": c.get("usable"),
            "menu": (ctx.get("menu") if isinstance(ctx, dict) else None)
                    or c.get("menu"),
            "slots": (ctx.get("slots_by_strategy") or {}).get(
                (c.get("parsed") or {}).get("strategy"), None)
                if isinstance(ctx, dict)
                and isinstance(c.get("parsed"), dict) else None,
            "prompt_language": (ctx.get("prompt_language")
                                if isinstance(ctx, dict) else None),
            "outcome": outcome,
            "policy_model": c.get("model") or "unknown",
            "ts": c.get("ts"),
        }
        step.update(step_reward(step))
        steps.append(step)
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=str(REPO / "rl/data/steps.jsonl"))
    args = ap.parse_args()
    runs = ([Path(r) for r in args.runs] if args.runs else
            sorted((REPO / "outputs").glob("movie_*")))
    total = 0
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in runs:
            if not (r / "brain_calls.jsonl").exists():
                continue
            rows = build_run(r)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False,
                                   default=str) + "\n")
            total += len(rows)
            print(f"{r.name}: {len(rows)} steps")
    print(f"TOTAL {total} steps → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
