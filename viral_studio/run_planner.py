#!/usr/bin/env python3
"""Planner 入口: 商品信息 + 人物图 → 分镜脚本(挑卡+填空)。不调用任何生成模型。

用法:
  python run_planner.py --product examples/product_pink_tee.yaml
  python run_planner.py --product ... --hooks 2      # 覆盖人物图数量(试 2 人/1 人路径)
"""
import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from studio.agents.storyboard_planner import StoryboardPlanner
from studio.config import OUTPUT_DIR, load_dotenv
from studio.skill_store import SkillStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--hooks", type=int, default=None, help="覆盖人物图数量")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    load_dotenv()

    brief = yaml.safe_load(Path(args.product).read_text(encoding="utf-8"))
    if args.hooks is not None:
        brief["person_hooks"] = brief.get("person_hooks", [])[:args.hooks]
    store = SkillStore()
    sb, rep = StoryboardPlanner(store).plan(brief)

    out = Path(args.out) if args.out else OUTPUT_DIR / f"sb_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "storyboard.json").write_text(
        json.dumps(sb.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(
        json.dumps(rep.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*74}\n分镜脚本: {sb.product_name} | {sb.category} | "
          f"{sb.person_count}人 | 预计 {sb.duration_hint(store):.0f}s\n{'='*74}")
    for s in sb.segments:
        card = store.get(s.skill_id) or {}
        p = card.get("produces", {})
        dur = p.get("duration_s") or (p.get("variants", {}).get(s.variant, {}).get("duration_s"))
        print(f"\n[{s.seg_id}] {s.part:8s} {s.skill_id:20s} {dur}s"
              f"{'  variant=' + s.variant if s.variant else ''}"
              f"{'  hook#' + str(s.hook_index) if s.hook_index else ''}")
        for k, v in s.slots.items():
            print(f"    {k:10s} = {v}")
        print(f"    理由: {s.reason}")
    print(f"\n整体: {sb.overall_reason}")
    print(f"\n校验: {'通过' if rep.ok else '未通过'} | 阻断 {len(rep.errors)} | 警告 {len(rep.warnings)}")
    for i in rep.errors:
        print(f"  ✗ [{i.seg_id}] {i.field}: {i.msg}")
    for i in rep.warnings:
        print(f"  ⚠ [{i.seg_id}] {i.field}: {i.msg}")
    print(f"\n产物: {out}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
