#!/usr/bin/env python3
"""End-to-end Maestro run (v0.2.2 — mock pipeline, CPU-only).

This is the *demo-friendly* entry point: one command, every innovation
visible in stdout. For programmatic / server use prefer:

    maestro run-once --prompt "..." -o out.mp4     # same behavior, on PATH
    maestro serve   --port 8000                    # FastAPI deployment
    maestro smoke                                  # operator healthcheck

Example:
    python scripts/run_pipeline.py \
        --prompt "a ball is thrown and bounces; a person runs through a city" \
        --music data/track.mp3 --image data/hero.png \
        --output outputs/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))     # so the script runs without `pip install -e .`

from maestro.config import load_config              # noqa: E402
from maestro.pipeline.run import run_maestro        # noqa: E402
from maestro.tools.base import default_registry     # noqa: E402


def _summarize_tools() -> None:
    """UniVA-style tool manifest banner so operators see what's wired."""
    reg = default_registry()
    by_cat: dict[str, list[str]] = {}
    for spec in reg.list_specs():
        by_cat.setdefault(spec.category, []).append(spec.name)
    print("Tools (UniVA-style registry):")
    for cat in sorted(by_cat):
        print(f"  {cat:11s} → {', '.join(sorted(by_cat[cat]))}")


def _summarize_trajectory(path: Path) -> dict:
    """Count per-action occurrences so operators see the live wiring."""
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a = json.loads(line)["action"]
        counts[a] = counts.get(a, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--source", nargs="*", default=[], help="source video paths")
    ap.add_argument("--image", nargs="*", default=[], help="reference image paths")
    ap.add_argument("--music", default=None)
    ap.add_argument("--output", default="outputs/demo.mp4")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--quiet", action="store_true",
        help="suppress innovation-evidence panel (just print output paths).",
    )
    args = ap.parse_args()

    if not args.quiet:
        _summarize_tools()
        print()

    config = load_config(args.config)
    out = Path(args.output)
    trajectory_path = out.with_suffix(".trajectory.jsonl")
    result = run_maestro(
        user_prompt=args.prompt,
        output_path=out,
        source_videos=[Path(p) for p in args.source],
        images=[Path(p) for p in args.image],
        music=Path(args.music) if args.music else None,
        config=config,
        cache_dir=out.parent / "cache",
        trajectory_path=trajectory_path,
        lesson_path=out.parent / "lessons.jsonl",
    )
    report = result["report"]
    print(f"\nOutput : {result['output_path']}")
    print(f"Report : {result['report_path']}")
    print(f"Traj   : {trajectory_path}")
    print(f"Shots  : {report['n_shots']}  Lessons learned (C4): {report['lessons_learned']}")

    # ── per-shot panel: every innovation gets at least one visible column ──
    for s in report["shots"]:
        fm = s["final_metrics"]
        print(
            f"  shot {s['shot_idx']}: "
            f"rev={s['revisions_used']}  "
            f"conv={s['converged']}  "
            f"gen_calls={s['gen_calls']}  "
            f"escalations(C5)={s['escalations']}  "
            f"tier_used(C5)={s['tier_used']}  "
            f"p1(C1)={fm.get('p1_physics')}  "
            f"p2(C6)={fm.get('p2_law_consistency')}  "
            f"scores={s['score_history']}"
        )
        if s.get("skipped_items"):
            print(f"    escape_hatch_skipped: {s['skipped_items']}")

    if not args.quiet:
        # Trajectory action distribution exposes every load-bearing agent:
        # annotate_physics (C6), plan_fix (C2), review (C3), tool_call (UniVA),
        # verify (Verifier), validate_plan (PlanValidator CCV), …
        counts = _summarize_trajectory(trajectory_path)
        print("\nTrajectory action distribution:")
        for action in sorted(counts, key=lambda a: (-counts[a], a)):
            print(f"  {action:18s} × {counts[action]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
