#!/usr/bin/env python3
"""viral_studio P2 入口: 商品信息 → 创意方向 → 分镜脚本(校验) → 工具调用计划。

只花少量 LLM tokens, 不调用任何视频生成模型——产物供人工审阅:
  outputs/plan_<ts>/creative_direction.json   策划产出
  outputs/plan_<ts>/shot_script.json          导演产出(机器可执行)
  outputs/plan_<ts>/validation.json           安检门报告
  outputs/plan_<ts>/tool_plan.json            逐段工具调用计划(dry)

用法: python run_plan.py --product examples/product_pink_tee.yaml
"""
import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from studio.config import OUTPUT_DIR, load_dotenv
from studio.agents import Director, Planner
from studio.executor import build_tool_plan
from studio.memory_store import MemoryStore
from studio.schemas import ProductBrief


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, help="商品 brief YAML")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    load_dotenv()

    brief = ProductBrief.model_validate(
        yaml.safe_load(Path(args.product).read_text(encoding="utf-8")))
    mem = MemoryStore()
    out = OUTPUT_DIR / f"plan_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    logging.info("① 策划中: %s", brief.name)
    direction = Planner(mem).plan(brief)
    _save(out / "creative_direction.json", direction.model_dump())
    for seg in direction.structure:
        logging.info("  [%s] %.1fs %s (pattern=%s asset=%s)",
                     seg.role, seg.duration_s, seg.idea[:40],
                     seg.pattern_ref, seg.asset_ref)

    logging.info("② 导演中")
    script, report = Director(mem).direct(brief, direction)
    _save(out / "shot_script.json", script.model_dump())
    _save(out / "validation.json", report.model_dump())
    logging.info("  校验: %s | 阻断 %d | 警告 %d",
                 "通过" if report.ok else "未通过",
                 len(report.errors), len(report.warnings))
    for w in report.warnings:
        logging.info("  ⚠ %s", w)
    for e in report.errors:
        logging.error("  ✗ %s", e)

    plan = build_tool_plan(script, mem)
    _save(out / "tool_plan.json", plan)
    billed = sum(i["billed_estimate_s"] for i in plan)
    logging.info("③ 工具计划: %d 段, 预计计费 ≈ %.0f 秒生成视频 | 产物: %s",
                 len(plan), billed, out)
    return 0 if report.ok else 1


def _save(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
