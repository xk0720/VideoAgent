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
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "rl"))
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


def _load_sb(run_dir: Path) -> dict:
    p = run_dir / "storyboard.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _judge_group(g: dict, samples: list, run_dir: Path, judges) -> None:
    """reward v3(2026-08-14 用户设计):文本判官逐候选 + 三路视频排名
    (一组一调用)+ 一致性对照直判 → 合成覆写 reward。任一分量失败
    → 剔除该分量、其余归一化并留痕;全失败 → 保留 v2 分。就地修改
    samples。"""
    from reward.judges import compose_rewards
    ctx = g.get("context") or {}
    sb = _load_sb(run_dir)
    entry = next((e for e in sb.get("entries", [])
                  if e.get("shot_idx") == g.get("shot_idx")), {})
    jm = entry.get("junction_meta") or {}
    kind = jm.get("kind")
    continuity = bool(kind == "continue"
                      or (kind == "derive" and jm.get("space_view")))
    shot = ctx.get("shot") or {}
    videos = [s.get("video") for s in samples]
    have_videos = all(v and Path(v).exists() for v in videos)

    # 文本判官(逐候选;动作 = brain 原始 video_prompt)
    text_scores: list = []
    for s in samples:
        try:
            comp = json.loads(s.get("completion") or "{}")
            case = {
                "shot_script": shot.get("description", ""),
                "cast_canon": ctx.get("cast") or {},
                "story_so_far": [
                    {"label": r.get("label"),
                     "description": r.get("description")}
                    for r in (ctx.get("storyboard") or [])],
                "prev_end_state": (ctx.get("prev_shot")
                                   or {}).get("end_state", ""),
                "junction": {"kind": kind,
                             "continuity_applicable": continuity,
                             "handoff_required": kind == "derive"},
                "slots": (ctx.get("slots_by_strategy")
                          or {}).get(comp.get("strategy"), []),
                "candidate_prompt": comp.get("video_prompt", ""),
            }
            score, detail = judges["text"].score(case)
            s["judge_text"] = detail
            text_scores.append(score)
        except Exception as exc:
            print(f"[judge] text failed ({exc})", flush=True)
            s["judge_text"] = {"error": str(exc)[:120]}
            text_scores.append(None)

    # 视频侧(排名三路 + 一致性直判;无视频文件则整块跳过)
    video_parts: dict = {"action": None, "physics": None,
                         "camera": None, "consistency": None}
    if have_videos and len(videos) >= 2:
        rank_ctx = {"shot_script": shot.get("description", ""),
                    "camera_facing": entry.get("camera_facing", ""),
                    "cast_canon": ctx.get("cast") or {}}
        for dim in ("action", "physics", "camera"):
            try:
                res = judges["ranker"].rank(dim, rank_ctx, videos)
                video_parts[dim] = res["points"]
                g.setdefault("judge_video", {})[dim] = {
                    "evidence": res.get("evidence"),
                    "order": res.get("order")}
            except Exception as exc:
                print(f"[judge] rank {dim} failed ({exc})", flush=True)
                g.setdefault("judge_video", {})[dim] = {
                    "error": str(exc)[:120]}
        refs = []
        portraits = sb.get("portraits") or {}
        for name in (ctx.get("cast_in_shot") or []):
            pth = portraits.get(name)
            if pth and Path(pth).exists():
                refs.append({"kind": f"portrait:{name}", "path": pth,
                             "note": (ctx.get("cast")
                                      or {}).get(name, "")[:120]})
        sv = jm.get("space_view") or {}
        if isinstance(sv, dict) and sv.get("path")                 and Path(sv["path"]).exists():
            refs.append({"kind": "space_view", "path": sv["path"],
                         "note": (sv.get("caption") or "")[:200]})
        if refs:
            cons: dict = {}
            for i, v in enumerate(videos):
                try:
                    sc, detail = judges["consistency"].score(
                        v, refs, {"shot_script":
                                  shot.get("description", "")})
                    cons[i] = sc
                    samples[i]["judge_consistency"] = detail
                except Exception as exc:
                    print(f"[judge] consistency failed ({exc})",
                          flush=True)
                    samples[i]["judge_consistency"] = {
                        "error": str(exc)[:120]}
            video_parts["consistency"] = cons or None

    composed = compose_rewards([s["r_format"] for s in samples],
                               text_scores, video_parts, len(samples))
    if all(c["r_text"] is None and c["r_video"] is None
           for c in composed):
        return                       # 全失败:保留 v2 分,诚实不覆写
    for s, c in zip(samples, composed):
        s.update(c)


def build_judges(judge_cfg_path: str):
    """从 RL 配置造判官三件套(文本=models.llm;视频=models.mllm 的
    omni 原生视频通道)。key:QWEN_API_KEY / DASHSCOPE_API_KEY。"""
    import os
    import yaml
    from reward.judges import (ConsistencyChecker, OpenAICompatChat,
                               TextJudge, VideoRanker)
    cfg = yaml.safe_load(open(judge_cfg_path))
    models = cfg.get("models", {})
    key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("judge 需要 QWEN_API_KEY/DASHSCOPE_API_KEY")
    base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    txt = OpenAICompatChat(base, (models.get("llm")
                                  or {}).get("model", "qwen-max"), key)
    vlm = OpenAICompatChat(base, (models.get("mllm")
                                  or {}).get("model",
                                             "qwen3.5-omni-plus"), key)
    return {"text": TextJudge(txt), "ranker": VideoRanker(vlm),
            "consistency": ConsistencyChecker(vlm)}


def collect_run(run_dir: Path, seen: set, judges=None) -> list:
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
        if judges is not None and len(samples) >= 2:
            _judge_group(g, samples, run_dir, judges)
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
    ap.add_argument("--judge", action="store_true",
                    help="开 reward v3 双重评审(文本+视频判官;需百炼 key)")
    ap.add_argument("--judge-config",
                    default=str(REPO / "rl/configs/server_bailian_qwen.yaml"))
    args = ap.parse_args()
    judges = build_judges(args.judge_config) if args.judge else None
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
            groups = collect_run(run_dir, seen, judges=judges)
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
