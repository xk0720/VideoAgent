#!/usr/bin/env python3
"""viral_studio 主入口 —— 一条命令跑完 Planner + Act(暂不执行生成)。

  商品信息 + 人物图  →  ① Planner 挑卡填空  →  ② Act 编译调用计划  →  落盘 + 打印

产物(outputs/run_<ts>/):
  storyboard.json     Planner 输出(skill_id + slots, 极简)
  validation.json     分镜校验报告
  call_plan.json      Act 编译出的工具调用计划(可直接执行的形态)
  decoded_script.txt  人读版剧本(每段真正会送进模型的 prompt)
  calls.txt           人读版调用清单(逐调用的工具名 + 全部参数 + 完整 prompt)
  prompts/            每段 prompt 单独成文件, 方便直接复制去试
  summary.json        一页纸: 时间轴 + 计费预估 + 校验结论

用法:
  python3 run_studio_v2.py --product examples/product_pink_tee.yaml
  python3 run_studio_v2.py --product ... --hooks 2 --bgm /path/to/bgm.wav
  python3 run_studio_v2.py --storyboard outputs/run_xxx/storyboard.json ...  # 复用已审剧本
"""
import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from studio.agents.act_agent import ActAgent
from studio.agents.storyboard_planner import StoryboardPlanner
from studio.config import OUTPUT_DIR, load_dotenv
from studio.skill_store import SkillStore
from studio.storyboard import Storyboard

log = logging.getLogger("viral_studio")
LINE = "=" * 78


def _short(v, n=70):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def write_calls_txt(plan: dict, dst: Path) -> None:
    """人读版调用清单 —— 每个调用的工具、参数、完整 prompt 一字不省。"""
    out = [LINE, f"调用计划  {plan['product_name']}  |  总时长 {plan['total_duration_s']}s"
                 f"  |  {len(plan['segments'])} 段", LINE, ""]
    for seg in plan["segments"]:
        out += ["─" * 78,
                f"▶ {seg['seg_id']}  [{seg['part']}]  {seg['t0']}–{seg['t1']}s"
                f"  skill={seg['skill_id']}"
                + (f"  variant={seg['variant']}" if seg.get("variant") else "")
                + (f"  (+{seg['tail_s']}s 尾部余量)" if seg.get("tail_s") else ""), ""]
        for i, c in enumerate(seg["calls"], 1):
            where = "本地" if c.get("local") else "远程·计费"
            out.append(f"  ({i}) [{c['id']}] {c['tool']}   〔{where}〕")
            for k, v in c["params"].items():
                if k == "prompt" and isinstance(v, str) and len(v) > 80:
                    out.append(f"        {k}:")
                    out += [f"          {ln}" for ln in v.splitlines()]
                else:
                    out.append(f"        {k}: {_short(v, 200)}")
            out.append("")
    c = plan["cost_estimate"]
    out += [LINE,
            f"计费预估: 视频 {c['video_s']:.0f}s | 音乐 {c['music_s']:.0f}s | "
            f"TTS {c['tts_chars']} 字 | 图像 {c['image_calls']} 张", LINE]
    dst.write_text("\n".join(out), encoding="utf-8")


def write_prompts(plan: dict, dst_dir: Path) -> int:
    """每段 prompt 单独成文件, 方便直接复制到模型 playground 里试。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for seg in plan["segments"]:
        for c in seg["calls"]:
            for key in ("prompt", "text"):
                v = c["params"].get(key)
                if isinstance(v, str) and v.strip():
                    (dst_dir / f"{seg['seg_id']}_{c['id']}_{key}.txt").write_text(
                        v.strip() + "\n", encoding="utf-8")
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, help="商品 brief YAML")
    ap.add_argument("--storyboard", default=None, help="复用已有分镜脚本, 跳过 Planner")
    ap.add_argument("--hooks", type=int, default=None, help="覆盖人物图数量")
    ap.add_argument("--bgm", default=None,
                    help="整片共享音轨(卡里 bgm_source=shared 的段用它切片; "
                         "策略层的默认来源, 此参数仅作覆盖)")
    ap.add_argument("--act", choices=["compiler", "agent", "both"], default="compiler",
                    help="Act 通路: compiler=确定性编译(默认) | agent=LLM 编排 | both=两者并跑并 diff")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    load_dotenv()

    brief = yaml.safe_load(Path(args.product).read_text(encoding="utf-8"))
    if args.hooks is not None:
        brief["person_hooks"] = (brief.get("person_hooks") or [])[:args.hooks]
        brief["hook_colors"] = (brief.get("hook_colors") or [])[:args.hooks]
    out = Path(args.out) if args.out else OUTPUT_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("输出目录: %s", out)

    store = SkillStore()

    # ── ① Planner ────────────────────────────────────────
    if args.storyboard:
        sb = Storyboard.model_validate(
            json.loads(Path(args.storyboard).read_text(encoding="utf-8")))
        rep = StoryboardPlanner(store).validate(sb, brief)
        log.info("① 复用已有分镜: %s", args.storyboard)
    else:
        log.info("① Planner 挑卡填空: %s (%d 张人物图)",
                 brief.get("name"), len(brief.get("person_hooks") or []))
        sb, rep = StoryboardPlanner(store).plan(brief, bgm_source=args.bgm)
    (out / "storyboard.json").write_text(
        json.dumps(sb.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(
        json.dumps(rep.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{LINE}\n① 分镜脚本  {sb.product_name} | {sb.category} | "
          f"{sb.person_count} 人\n{LINE}")
    for s in sb.segments:
        print(f"  {s.seg_id} [{s.part:7s}] {s.skill_id:20s}"
              + (f" v{s.variant}" if s.variant else "")
              + (f" hook#{s.hook_index}" if s.hook_index else ""))
        print(f"        {s.t0}–{s.t1}s  " + " → ".join(c["tool"] for c in s.pipeline))
        for k, v in s.texts.items():
            print(f"        {k:10s} = {_short(v, 84)}")
    print(f"  校验: {'通过' if rep.ok else '未通过'} "
          f"(阻断 {len(rep.errors)} / 警告 {len(rep.warnings)})")
    for i in rep.errors:
        print(f"    ✗ [{i.seg_id}] {i.field}: {i.msg}")
    for i in rep.warnings:
        print(f"    ⚠ [{i.seg_id}] {i.field}: {i.msg}")
    if not rep.ok:
        log.error("分镜校验未通过, 停止(产物已落盘供检查)")
        return 1

    # ── ② Act ────────────────────────────────────────────
    log.info("② Act 编译调用计划")
    plan, errs = ActAgent(store, brief, bgm_source=args.bgm).plan(sb, mode=args.act)
    agent_segs = plan.pop("_agent_segments", None)
    (out / "call_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_calls_txt(plan, out / "calls.txt")
    n_prompt = write_prompts(plan, out / "prompts")

    # 人读版剧本(复用解码器)
    import subprocess
    dec = subprocess.run(
        ["python3", "tools/decode_storyboard.py", str(out / "storyboard.json")],
        capture_output=True, text=True)
    (out / "decoded_script.txt").write_text(dec.stdout, encoding="utf-8")

    print(f"\n{LINE}\n② 调用计划  总时长 {plan['total_duration_s']}s\n{LINE}")
    n_remote = 0
    for seg in plan["segments"]:
        print(f"\n  ▶ {seg['seg_id']} {seg['t0']}–{seg['t1']}s  {seg['skill_id']}")
        for c in seg["calls"]:
            remote = not c.get("local")
            n_remote += remote
            keys = ", ".join(f"{k}={_short(v, 42)}" for k, v in c["params"].items()
                             if k not in ("prompt", "text"))
            print(f"     [{c['id']:6s}] {'远程' if remote else '本地'} "
                  f"{c['tool']:22s} {keys[:110]}")
            for k in ("prompt", "text"):
                if isinstance(c["params"].get(k), str) and c["params"][k].strip():
                    print(f"              {k} → prompts/{seg['seg_id']}_{c['id']}_{k}.txt")

    # both 模式: 对照 compiler 与 agent 的差异
    if agent_segs:
        diffs = []
        for ref, ag in zip(plan["segments"], agent_segs):
            a = [(c["tool"], tuple(sorted(c["params"]))) for c in ref["calls"]]
            b = [(c["tool"], tuple(sorted(c["params"]))) for c in ag["calls"]]
            if a != b:
                diffs.append((ref["seg_id"],
                              [c["tool"] for c in ref["calls"]],
                              [c["tool"] for c in ag["calls"]]))
        print(f"\n{LINE}\n③ compiler ↔ agent 对照\n{LINE}")
        if not diffs:
            print("  两条通路结果一致 ✓")
        for sid, x, y in diffs:
            print(f"  ⚠ {sid}\n      compiler: {' → '.join(x)}\n      agent   : {' → '.join(y)}")
        (out / "call_plan_agent.json").write_text(
            json.dumps({"segments": agent_segs}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    cost = plan["cost_estimate"]
    summary = {"product": plan["product_name"], "person_count": plan["person_count"],
               "total_duration_s": plan["total_duration_s"],
               "segments": [{"seg_id": s["seg_id"], "part": s["part"],
                             "skill_id": s["skill_id"], "variant": s.get("variant"),
                             "t0": s["t0"], "t1": s["t1"],
                             "tools": [c["tool"] for c in s["calls"]]}
                            for s in plan["segments"]],
               "remote_calls": n_remote, "cost_estimate": cost,
               "storyboard_ok": rep.ok, "call_plan_issues": errs}
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{LINE}\n计费预估: 视频 {cost['video_s']:.0f}s | 音乐 {cost['music_s']:.0f}s | "
          f"TTS {cost['tts_chars']} 字 | 图像 {cost['image_calls']} 张"
          f"  (远程调用 {n_remote} 次)")
    print(f"计划校验: {'通过' if not errs else f'{len(errs)} 项问题'}")
    for e in errs:
        print(f"  ✗ {e}")
    print(f"\n产物 → {out}")
    for f, desc in (("storyboard.json", "分镜(skill_id + slots)"),
                    ("call_plan.json", "调用计划(可执行形态)"),
                    ("calls.txt", "调用清单(人读, 参数与 prompt 全文)"),
                    ("decoded_script.txt", "剧本(人读, 每段实际 prompt)"),
                    (f"prompts/ ({n_prompt} 个)", "每段 prompt 单文件"),
                    ("summary.json", "一页纸摘要")):
        print(f"  · {f:26s} {desc}")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
