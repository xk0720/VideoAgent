#!/usr/bin/env python3
"""Act: 分镜脚本 → 工具调用计划(call plan)。确定性编译, 不调用任何模型。

用法:
  python3 run_act.py --storyboard outputs/sb_xxx/storyboard.json \
                     --product examples/product_pink_tee.yaml [--bgm path.wav]
"""
import argparse
import json
from pathlib import Path

import yaml

from studio.act_compiler import ActCompiler          # noqa: F401 (供 --debug 用)
from studio.agents.act_agent import ActAgent
from studio.config import load_dotenv
from studio.skill_store import SkillStore
from studio.storyboard import Storyboard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboard", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--bgm", default=None, help="整片源 BGM(供 $bgm_slice 切片)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_dotenv()

    sb = Storyboard.model_validate(
        json.loads(Path(args.storyboard).read_text(encoding="utf-8")))
    brief = yaml.safe_load(Path(args.product).read_text(encoding="utf-8"))
    agent = ActAgent(SkillStore(), brief, bgm_source=args.bgm)
    plan, errs = agent.plan(sb)

    out = Path(args.out) if args.out else Path(args.storyboard).parent / "call_plan.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*78}\n调用计划: {plan['product_name']} | "
          f"{plan['total_duration_s']}s | {len(plan['segments'])} 段\n{'='*78}")
    for seg in plan["segments"]:
        print(f"\n▶ {seg['seg_id']} [{seg['part']}] {seg['t0']}–{seg['t1']}s  "
              f"{seg['skill_id']}" + (f" v{seg['variant']}" if seg.get("variant") else ""))
        for c in seg["calls"]:
            tag = "本地" if c.get("local") else "远程"
            ps = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v)
                  for k, v in c["params"].items()}
            print(f"   [{c['id']:6s}] {tag} {c['tool']:22s} {json.dumps(ps, ensure_ascii=False)[:150]}")
    c = plan["cost_estimate"]
    print(f"\n计费预估: 视频 {c['video_s']:.0f}s | 音乐 {c['music_s']:.0f}s | "
          f"TTS {c['tts_chars']} 字 | 图像 {c['image_calls']} 张")
    print(f"\n计划校验: {'通过' if not errs else f'{len(errs)} 项问题'}")
    for e in errs:
        print(f"  ✗ {e}")
    print(f"\n产物: {out}")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
