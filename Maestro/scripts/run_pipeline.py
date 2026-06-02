#!/usr/bin/env python3
"""End-to-end Maestro run (v0.1 mock; CPU-only).

Example:
  python scripts/run_pipeline.py \
      --prompt "a ball is thrown and bounces; a person runs through a city" \
      --music data/track.mp3 --image data/hero.png \
      --output outputs/demo.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maestro.config import load_config  # noqa: E402
from maestro.pipeline.run import run_maestro  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--source", nargs="*", default=[], help="source video paths")
    ap.add_argument("--image", nargs="*", default=[], help="reference image paths")
    ap.add_argument("--music", default=None)
    ap.add_argument("--output", default="outputs/demo.mp4")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    out = Path(args.output)
    result = run_maestro(
        user_prompt=args.prompt,
        output_path=out,
        source_videos=[Path(p) for p in args.source],
        images=[Path(p) for p in args.image],
        music=Path(args.music) if args.music else None,
        config=config,
        cache_dir=out.parent / "cache",
        trajectory_path=out.with_suffix(".trajectory.jsonl"),
        lesson_path=out.parent / "lessons.jsonl",
    )
    report = result["report"]
    print(f"\nOutput : {result['output_path']}")
    print(f"Report : {result['report_path']}")
    print(f"Shots  : {report['n_shots']}  Lessons learned: {report['lessons_learned']}")
    for s in report["shots"]:
        print(
            f"  shot {s['shot_idx']}: revisions={s['revisions_used']} "
            f"converged={s['converged']} gen_calls={s['gen_calls']} "
            f"scores={s['score_history']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
