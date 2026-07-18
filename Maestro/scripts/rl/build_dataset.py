"""S0 · RL 训练数据构建器:把 attempt 运行日志变成训练样本(2026-07-18)。

用法:
    python scripts/rl/build_dataset.py outputs/attempt2 outputs/attempt3 \
        --out data/rl [--holdout attempt3]

输入(每个 run 目录):brain_calls.jsonl(决策 + 结局记录)+ storyboard.json
输出(--out 目录):
    sft.jsonl        — label=true 的样本(SFT 温启动用)
    kto.jsonl        — 全量带 label 样本(TRL KTOTrainer 格式)
    dpo_pairs.jsonl  — 真正成对的样本(TRL DPOTrainer 格式)
    eval_holdout.jsonl — --holdout 指定 run 的样本(整 run 划走,防泄漏)
    excluded.jsonl   — 被排除的记录 + 排除原因(诚实审计:每条不进训练
                       集的决策都写明为什么)

标签规则(v1,保守;"不怪它的失败不进它的坏样本"):
    generation-condition / image-plan:
        via != llm            → 排除(不是模型说的话)
        usable=False          → 坏(结构层错误,确定性)
        镜零修复收敛           → 好
        镜修复后才收敛 / 未收敛 → 排除(归因不清,v1 不强判)
        图计划被降级执行       → 坏(计划本身不可执行)
    prompt_enhance:
        引用闸门拒(ref_audit 不 ok)→ 坏;解析失败 → 坏
        可用 ∧ 镜零修复收敛          → 好;其余排除
    repair/decide(靠 decision_id 连 repair/outcome):
        verifier accepted → 好;rejected → 坏
        accept 且镜收敛 → 好;via=skill/fallback 或无判决 → 排除
    scene_write:v1 整体排除(全片级归因另做)。

prompt 重建:generation-condition / image-plan 走生产同源函数
`window_loop.decision_prompt`(逐字符一致);repair / enhancer 按各自
生产格式重拼,skill 文本用当前仓库版本 —— skill_chars 与日志不符时
fidelity 降为 "approx"(skill 演化过,样本仍可用但标注置信下降)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from maestro.pipeline.window_loop import decision_prompt  # noqa: E402
from maestro.skills.loader import load_skill              # noqa: E402

_DECISION_STAGES = {
    "window/generation-condition": "condition",
    "window/image-plan": "image_plan",
    "window/prompt_enhance": "enhance",
    "repair/decide": "repair",
}
_STAGE_SKILL = {"condition": "window_generation", "image_plan": "image_plan",
                "enhance": "prompt_enhancer", "repair": "orchestrator"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _skill_text(stage: str) -> tuple[str, int]:
    try:
        sk = load_skill(_STAGE_SKILL[stage]) or {}
        body = str(sk.get("body") or "")
        return body, len(body)
    except Exception:
        return "", 0


def _rebuild_prompt(stage: str, rec: dict) -> tuple[str, str]:
    """(prompt 文本, fidelity)。fidelity: exact | approx | missing。"""
    ctx = rec.get("context")
    if not isinstance(ctx, dict):
        return "", "missing"
    skill, n = _skill_text(stage)
    if not skill:
        return "", "missing"
    fid = "exact" if n == rec.get("skill_chars") else "approx"
    if stage in ("condition", "image_plan"):
        menu = rec.get("menu") or []
        menu_dicts = [{"name": m} if isinstance(m, str) else m for m in menu]
        return decision_prompt(skill, menu_dicts, ctx), fid
    if stage == "repair":
        user_ctx = dict(ctx)
        if rec.get("tools_menu"):
            user_ctx["tools"] = rec["tools_menu"]
        else:
            fid = "approx"          # 旧日志没记完整工具菜单
        return (skill + "\n\nTHIS TURN (JSON):\n"
                + json.dumps(user_ctx, ensure_ascii=False, indent=2)
                + '\n\nRespond with STRICT JSON only: {"tool": ..., '
                  '"args": {...}, "reason": ...}'), "approx"
    if stage == "enhance":
        return (skill + "\n\nTHIS TURN (JSON):\n"
                + json.dumps(ctx, ensure_ascii=False)
                + '\n\nSTRICT JSON only: {"video_prompt": "<the final '
                  'polished prompt, English, 30-100 words>"}'), "approx"
    return "", "missing"


def _completion(rec: dict) -> str:
    """训练目标文本:可用决策 → 规范化 JSON(教干净输出);不可用 →
    模型原话(那才是要抑制的行为)。"""
    if rec.get("usable") and isinstance(rec.get("parsed"), dict):
        clean = {k: v for k, v in rec["parsed"].items()
                 if k not in ("decision_id", "via")}
        return json.dumps(clean, ensure_ascii=False)
    return str(rec.get("raw") or "")


def _label(stage: str, rec: dict, shot_out: dict, repair_out: dict,
           sb_entry: dict) -> tuple:
    """→ (label True/False/None=排除, why, confidence)。"""
    via = rec.get("via") or (rec.get("parsed") or {}).get("via") or "llm"
    if via != "llm":
        return None, f"via={via} (not a model decision)", 0.0
    if stage == "repair":
        if not rec.get("usable"):
            return False, "unparseable / out-of-menu reply", 1.0
        oc = (repair_out or {}).get("outcome")
        if oc == "accepted":
            return True, "verifier accepted", 1.0
        if oc == "rejected":
            return False, "verifier rejected", 1.0
        if oc == "stop":
            if (shot_out or {}).get("converged"):
                return True, "accept on a converged shot", 0.8
            return None, "accept without convergence — ambiguous", 0.0
        return None, "no recorded verdict", 0.0
    if not rec.get("usable"):
        why = "unparseable / out-of-menu reply"
        if stage == "enhance" and (rec.get("ref_audit") or {}).get("unknown"):
            why = f"referenced unknown slots {rec['ref_audit']['unknown']}"
        return False, why, 1.0
    if stage == "image_plan" and (sb_entry or {}).get("plan_degraded_from"):
        return False, "plan degraded at execution (not executable as decided)", 0.8
    so = shot_out or {}
    if so.get("converged") and so.get("repair_turns", 1) == 0:
        return True, "shot converged with zero repairs", 1.0
    if so.get("converged"):
        return None, "converged only after repairs — credit ambiguous", 0.0
    if so:
        return None, "shot unconverged — attribution ambiguous in v1", 0.0
    return None, "no shot outcome record (legacy run?)", 0.0


def build_run(run_dir: Path) -> dict:
    """一个 run → {"samples": [...], "pairs": [...], "excluded": [...]}"""
    recs = _load_jsonl(run_dir / "brain_calls.jsonl")
    sb = {}
    sb_path = run_dir / "storyboard.json"
    if sb_path.exists():
        try:
            data = json.loads(sb_path.read_text(encoding="utf-8"))
            sb = {e.get("label"): e for e in data.get("entries", [])}
        except (json.JSONDecodeError, OSError):
            pass
    shot_outs = {r.get("label"): r for r in recs
                 if r.get("stage") == "window/shot_outcome"}
    shot_outs_idx = {r.get("shot_idx"): r for r in recs
                     if r.get("stage") == "window/shot_outcome"}
    repair_outs = {r.get("decision_id"): r for r in recs
                   if r.get("stage") == "repair/outcome"
                   and r.get("decision_id")}
    samples, pairs, excluded = [], [], []

    decisions = [r for r in recs if r.get("stage") in _DECISION_STAGES]
    for rec in decisions:
        stage = _DECISION_STAGES[rec["stage"]]
        label_key = rec.get("label")
        shot_out = (shot_outs.get(label_key)
                    or shot_outs_idx.get(rec.get("shot_idx")) or {})
        did = (rec.get("decision_id")
               or (rec.get("parsed") or {}).get("decision_id"))
        repair_out = repair_outs.get(did, {})
        sbe = sb.get(label_key, {})
        lab, why, conf = _label(stage, rec, shot_out, repair_out, sbe)
        meta = {"run": run_dir.name, "stage": stage,
                "decision_id": did, "shot": label_key
                or rec.get("shot_idx"), "why": why, "confidence": conf}
        if lab is None:
            excluded.append({**meta, "usable": rec.get("usable")})
            continue
        prompt, fid = _rebuild_prompt(stage, rec)
        if not prompt:
            excluded.append({**meta, "why": f"{why}; prompt not rebuildable"})
            continue
        samples.append({
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant",
                            "content": _completion(rec)}],
            "label": bool(lab),
            "meta": {**meta, "prompt_fidelity": fid},
        })

    # ── 成对样本 ① enhancer 拒/过重试对(同镜 attempt 0→1)──
    enh = [r for r in decisions if r["stage"] == "window/prompt_enhance"]
    by_label: dict = {}
    for r in enh:
        by_label.setdefault(r.get("label"), []).append(r)
    for label_key, group in by_label.items():
        a0 = next((r for r in group if r.get("attempt") == 0
                   and not r.get("usable")
                   and (r.get("ref_audit") or {}).get("unknown")), None)
        a1 = next((r for r in group if r.get("attempt") == 1
                   and r.get("usable")), None)
        if a0 and a1:
            prompt, fid = _rebuild_prompt("enhance", a0)
            if prompt:
                pairs.append({
                    "prompt": [{"role": "user", "content": prompt}],
                    "chosen": [{"role": "assistant",
                                "content": str(a1.get("raw") or "")}],
                    "rejected": [{"role": "assistant",
                                  "content": str(a0.get("raw") or "")}],
                    "meta": {"run": run_dir.name, "kind": "enhancer_retry",
                             "shot": label_key, "confidence": 1.0,
                             "prompt_fidelity": fid},
                })

    # ── 成对样本 ② 修复:被拒的 t 轮 vs 被收的 t+1 轮(状态近似不变:
    # 拒收后 best 未变,仅 history 多一条 —— confidence 0.7 如实标注)──
    reps = [r for r in decisions if r["stage"] == "repair/decide"
            and r.get("usable")]
    by_shot: dict = {}
    for r in reps:
        by_shot.setdefault(r.get("shot_idx"), []).append(r)
    for shot_idx, group in by_shot.items():
        for i in range(len(group) - 1):
            did_a = (group[i].get("decision_id")
                     or (group[i].get("parsed") or {}).get("decision_id"))
            did_b = (group[i + 1].get("decision_id")
                     or (group[i + 1].get("parsed") or {}).get("decision_id"))
            oc_a = (repair_outs.get(did_a) or {}).get("outcome")
            oc_b = (repair_outs.get(did_b) or {}).get("outcome")
            if oc_a == "rejected" and oc_b == "accepted":
                prompt, fid = _rebuild_prompt("repair", group[i])
                if prompt:
                    pairs.append({
                        "prompt": [{"role": "user", "content": prompt}],
                        "chosen": [{"role": "assistant",
                                    "content": str(group[i + 1].get("raw") or "")}],
                        "rejected": [{"role": "assistant",
                                      "content": str(group[i].get("raw") or "")}],
                        "meta": {"run": run_dir.name,
                                 "kind": "repair_reject_then_accept",
                                 "shot": shot_idx, "confidence": 0.7,
                                 "prompt_fidelity": fid},
                    })
    return {"samples": samples, "pairs": pairs, "excluded": excluded}


def _dump(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories (outputs/attemptN)")
    ap.add_argument("--out", default="data/rl")
    ap.add_argument("--holdout", nargs="*", default=[],
                    help="run dir NAMES routed to eval_holdout.jsonl")
    args = ap.parse_args(argv)

    train_s, train_p, holdout, excluded = [], [], [], []
    for rd in args.runs:
        run_dir = Path(rd)
        got = build_run(run_dir)
        excluded.extend(got["excluded"])
        if run_dir.name in args.holdout:
            holdout.extend(got["samples"])
        else:
            train_s.extend(got["samples"])
            train_p.extend(got["pairs"])

    out = Path(args.out)
    _dump(out / "kto.jsonl", train_s)
    _dump(out / "sft.jsonl", [s for s in train_s if s["label"]])
    _dump(out / "dpo_pairs.jsonl", train_p)
    _dump(out / "eval_holdout.jsonl", holdout)
    _dump(out / "excluded.jsonl", excluded)
    n_good = sum(1 for s in train_s if s["label"])
    print(f"kto={len(train_s)} (good={n_good} bad={len(train_s) - n_good}) "
          f"sft={n_good} dpo_pairs={len(train_p)} "
          f"holdout={len(holdout)} excluded={len(excluded)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
