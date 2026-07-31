#!/usr/bin/env python
"""S1 预备 · 合成训练日志生成器:按 S0 真实格式伪造若干 run 的
brain_calls.jsonl(+storyboard.json),供 build_dataset → TRL 训练管道
【冒烟】。⚠️ 合成样本只用于验证训练链路能跑通,不用于换取真实收益 ——
真训练数据来自实际跑片(Crayotter fixture 哲学:先零成本把管道走通)。

每个 run 内含(全部 S0 新格式,decision_id/结局记录齐全):
  · 条件决策:可用(零修复收敛→好 / 修后收敛→排除 / 不收敛→排除)、
    解析失败(→坏)、episode 重放(→排除);
  · 润色决策:干净通过(好)、引用未知槽位被拒+重试通过(坏+DPO 对);
  · 修复决策:t 轮被拒 + t+1 轮被收(KTO 一坏一好 + DPO 对);
  · 每镜 window/shot_outcome。

用法:
    python scripts/rl/make_synthetic_runs.py --runs 6 --shots 4 --seed 7
    python scripts/rl/build_dataset.py data/rl/synthetic_runs/run_* \\
        --out data/rl --holdout run_05
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SUBJECTS = [
    ("the orange-and-white cat", "trots across the sunlit living room "
                                 "toward the ceramic food bowl"),
    ("the young baker", "carries a tray of golden croissants to the "
                        "glass counter"),
    ("the painter", "adds a bridge outline to the canvas by the canal"),
    ("the skater girl", "glides along the riverside path at dusk"),
]
_STRATS = ["extend_prev", "ti2v_prev_last", "ti2v_prev_plus_keyframe",
           "i2v_keyframe", "t2v"]


def _slots(strategy: str) -> list[dict]:
    if strategy == "ti2v_prev_plus_keyframe":
        return [{"slot": "@Image1", "referenceable": True,
                 "content": "the previous shot's final frame"},
                {"slot": "@Image2", "referenceable": True,
                 "content": "user asset: character photo"}]
    if strategy in ("ti2v_prev_last", "i2v_keyframe"):
        return [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": "the opening frame"}]
    return []


def _ctx(label: str, subject: str, action: str, menu: list[str]) -> dict:
    return {
        "shot": {"label": label,
                 "description": f"{label}: <{subject}> {action}",
                 "end_state": f"{subject} mid-action, moving right; "
                              "camera: tracking right",
                 "dialogue": ""},
        "junction": {"prev_last_frame_actual":
                     f"{subject} is mid-stride, camera tracking right",
                     "required_end_state": "arrives at the target"},
        "cast": {subject: f"static: {subject} with distinctive look; "
                          "dynamic: pose"},
        "menu": menu,
        "slots_by_strategy": {m: _slots(m) for m in menu},
    }


def _rec(stage: str, **kw) -> dict:
    kw.setdefault("decision_id", uuid.uuid4().hex[:16])
    return {"ts": 0.0, "stage": stage, **kw}


def make_run(run_dir: Path, rng: random.Random, n_shots: int) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = []
    entries = []
    for i in range(n_shots):
        label = f"scene 1 shot {i + 1}"
        subject, action = rng.choice(_SUBJECTS)
        menu = rng.sample(_STRATS, k=3)
        strat = menu[0]
        ctx = _ctx(label, subject, action, menu)
        skill_chars = rng.randint(6000, 9000)
        # 每 run 第 0 镜:episode 重放(排除);第 1 镜:解析失败(坏);
        # 其余:可用决策(结局见 shot_outcome)。
        if i == 0:
            recs.append(_rec("window/generation-condition", label=label,
                             usable=True, via="episode", menu=menu,
                             parsed={"strategy": strat, "via": "episode"},
                             context=ctx))
        elif i == 1:
            recs.append(_rec("window/generation-condition", label=label,
                             usable=False, menu=menu, context=ctx,
                             raw="I think maybe we should... (not json)",
                             parsed=None, skill_chars=skill_chars))
        else:
            vp = (f"The shot opens EXACTLY on @Image1 — the final moment "
                  f"of the previous shot. {subject} {action}, camera "
                  f"tracking right. Preserve the established scene, "
                  f"lighting and camera."
                  if strat == "ti2v_prev_plus_keyframe" else
                  f"{subject} {action}, camera tracking right at walking "
                  f"pace, warm light.")
            parsed = {"strategy": strat, "reason": "continuity first",
                      "video_prompt": vp, "via": "llm"}
            recs.append(_rec("window/generation-condition", label=label,
                             usable=True, via="llm", menu=menu,
                             raw=json.dumps(parsed), parsed=parsed,
                             context=ctx, skill_chars=skill_chars))

        # 润色:每 run 第 2 镜做"拒+重试"对,其余干净通过
        enh_ctx = {"shot_description": f"{subject} {action}",
                   "strategy": strat, "model_family": "seedance_t2v",
                   "conditions": [{"kind": "image", "slot": "@Image1",
                                   "referenceable": True,
                                   "description": "prev final frame"}]}
        if i == 2:
            recs.append(_rec("window/prompt_enhance", label=label,
                             usable=False, attempt=0,
                             ref_audit={"ok": False, "unknown": ["@Image9"],
                                        "allowed": ["@Image1"]},
                             raw=f"{subject} {action}, continue @Image9",
                             parsed=None, context=enh_ctx,
                             skill_chars=skill_chars))
            recs.append(_rec("window/prompt_enhance", label=label,
                             usable=True, attempt=1,
                             ref_audit={"ok": True, "unknown": [],
                                        "appended": []},
                             raw=f"{subject} {action}, opening exactly on "
                                 f"@Image1, camera tracking right.",
                             parsed={"video_prompt": f"{subject} {action}"},
                             context=enh_ctx, skill_chars=skill_chars))
        else:
            recs.append(_rec("window/prompt_enhance", label=label,
                             usable=True, attempt=0,
                             ref_audit={"ok": True, "unknown": [],
                                        "appended": []},
                             raw=f"{subject} {action}, camera tracking.",
                             parsed={"video_prompt": f"{subject} {action}"},
                             context=enh_ctx, skill_chars=skill_chars))

        # 修复:偶数镜做一段"拒→收"序列(KTO 一坏一好 + DPO 对)
        n_repairs = 0
        if i % 2 == 0:
            n_repairs = 2
            tools_menu = [{"name": "regenerate_segment",
                           "description": "frame-precise repair"},
                          {"name": "regenerate",
                           "description": "full re-generation"}]
            d1 = _rec("repair/decide", shot_idx=i, usable=True, via="llm",
                      menu=["regenerate", "regenerate_segment"],
                      tools_menu=tools_menu,
                      raw=json.dumps({"tool": "regenerate_segment",
                                      "args": {"frame_start": 0,
                                               "frame_end": 30,
                                               "hint": "fix the opening"}}),
                      parsed={"tool": "regenerate_segment",
                              "args": {"frame_start": 0, "frame_end": 30,
                                       "hint": "fix the opening"},
                              "via": "llm"},
                      context={"defects": "opening does not continue",
                               "history": []}, skill_chars=skill_chars)
            recs.append(d1)
            recs.append(_rec("repair/outcome",
                             decision_id=d1["decision_id"], shot_idx=i,
                             turn=1, tool="regenerate_segment",
                             outcome="rejected"))
            hint = (f"{subject} {action} from the opening moment; preserve "
                    f"scene, lighting and camera.")
            d2 = _rec("repair/decide", shot_idx=i, usable=True, via="llm",
                      menu=["regenerate", "regenerate_segment"],
                      tools_menu=tools_menu,
                      raw=json.dumps({"tool": "regenerate",
                                      "args": {"hint": hint}}),
                      parsed={"tool": "regenerate", "args": {"hint": hint},
                              "via": "llm"},
                      context={"defects": "opening does not continue",
                               "history": [["regenerate_segment",
                                            "rejected"]]},
                      skill_chars=skill_chars)
            recs.append(d2)
            recs.append(_rec("repair/outcome",
                             decision_id=d2["decision_id"], shot_idx=i,
                             turn=2, tool="regenerate", outcome="accepted"))

        # 每镜结局:第 3+ 镜按概率收敛;修过的镜 repair_turns>0
        converged = (i >= 2 and rng.random() < 0.7) or (i == 0)
        recs.append(_rec("window/shot_outcome", label=label, shot_idx=i,
                         converged=bool(converged),
                         stop_reason="converged" if converged
                         else "turns_exhausted",
                         repair_turns=n_repairs, gen_calls=1 + n_repairs))
        entries.append({"label": label, "scene_idx": 1, "shot_idx": i,
                        "description": f"{label}: {subject} {action}",
                        "status": "verified" if converged
                        else "generated_with_defects"})

    with (run_dir / "brain_calls.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "storyboard.json").write_text(
        json.dumps({"entries": entries}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return len(recs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "rl"
                                         / "synthetic_runs"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    total = 0
    for i in range(args.runs):
        total += make_run(out / f"run_{i:02d}", rng, args.shots)
    print(f"synthetic: {args.runs} runs × {args.shots} shots → "
          f"{total} records under {out}")
    print("next: python scripts/rl/build_dataset.py "
          f"{out}/run_* --out data/rl --holdout run_{args.runs - 1:02d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
