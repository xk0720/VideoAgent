#!/usr/bin/env python3
"""viral_studio 端到端: 商品信息 → 创意方向 → 分镜脚本 → 生成 → 带音乐成片。

  python run_studio.py --product examples/product_pink_tee.yaml --dry-run
  python run_studio.py --product examples/product_pink_tee.yaml            # 真跑(会花钱)
  python run_studio.py --script outputs/plan_xxx/shot_script.json          # 复用已审脚本

安全默认: 未加 --yes 时会打印计费预估并要求确认。校验未过的脚本默认拒绝执行
(--force 可越过, 但你得知道自己在干嘛)。
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import yaml

from studio.agents import Director, Planner
from studio.agents.executor_agent import ExecutorAgent
from studio.config import OUTPUT_DIR, load_dotenv
from studio.executor import build_tool_plan
from studio.memory_store import MemoryStore
from studio.schemas import ProductBrief, ShotScript
from studio.validate import validate_script


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", help="商品 brief YAML(与 --script 二选一)")
    ap.add_argument("--script", help="复用已有 shot_script.json, 跳过策划/导演")
    ap.add_argument("--out", default=None, help="输出目录(默认 outputs/studio_<ts>)")
    ap.add_argument("--dry-run", action="store_true", help="只出计划, 不调用生成模型")
    ap.add_argument("--yes", action="store_true", help="跳过费用确认")
    ap.add_argument("--force", action="store_true", help="校验未过也执行")
    ap.add_argument("--mode", choices=["wan-std", "wan-pro"], default="wan-std")
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bgm", default=None, help="全片 BGM(可选, 与段内音轨混合)")
    ap.add_argument("--no-audio", action="store_true",
                    help="关闭 seedance 原生音画同出(口播段会没声音)")
    args = ap.parse_args()
    if not args.product and not args.script:
        ap.error("需要 --product 或 --script")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    load_dotenv()
    mem = MemoryStore()
    out = Path(args.out) if args.out else OUTPUT_DIR / f"studio_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    logging.info("输出目录: %s", out)

    # ── 规划(P2) ─────────────────────────────────────────
    if args.script:
        script = ShotScript.model_validate(
            json.loads(Path(args.script).read_text(encoding="utf-8")))
        brief = ProductBrief(name=script.product_name, description="(复用脚本)",
                             duration_target_s=script.total_duration_s)
        report = validate_script(script, brief, mem)
        logging.info("复用脚本 %s (%d 段)", args.script, len(script.segments))
    else:
        brief = ProductBrief.model_validate(
            yaml.safe_load(Path(args.product).read_text(encoding="utf-8")))
        logging.info("① 策划中: %s", brief.name)
        direction = Planner(mem).plan(brief)
        _save(out / "creative_direction.json", direction.model_dump())
        for s in direction.structure:
            logging.info("  [%s] %.0fs %s", s.role, s.duration_s, s.idea[:44])
        logging.info("② 导演中")
        script, report = Director(mem).direct(brief, direction)
        _save(out / "shot_script.json", script.model_dump())

    _save(out / "validation.json", report.model_dump())
    for w in report.warnings:
        logging.info("  ⚠ %s", w)
    for e in report.errors:
        logging.error("  ✗ %s", e)

    plan = build_tool_plan(script, mem)
    _save(out / "tool_plan.json", plan)
    billed = sum(i["billed_estimate_s"] for i in plan)
    logging.info("③ 计划: %d 段, 预计计费 ≈ %.0f 秒生成视频", len(plan), billed)

    if args.dry_run:
        logging.info("dry-run 结束(未调用生成模型): %s", out)
        return 0 if report.ok else 1
    if not report.ok and not args.force:
        logging.error("校验未通过, 拒绝执行(--force 可越过)")
        return 1
    if not args.yes:
        ans = input(f"将真实调用生成模型 {len(plan)} 次, 预计计费约 {billed:.0f} 秒, 继续? [y/N] ")
        if ans.strip().lower() != "y":
            logging.info("已取消")
            return 0

    # ── 执行(P3) ─────────────────────────────────────────
    logging.info("④ 生成中")
    ex = ExecutorAgent(mem, out,
                       dashscope_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                       wavespeed_key=os.environ.get("WAVESPEED_API_KEY", ""),
                       animate_mode=args.mode, resolution=args.resolution,
                       generate_audio=not args.no_audio, workers=args.workers)
    summary = ex.execute(script, bgm=args.bgm)
    if summary["dropped"]:
        logging.info("剔除段落: %s", ", ".join(summary["dropped"]))
    return 0 if summary["final"] else 1


def _save(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
