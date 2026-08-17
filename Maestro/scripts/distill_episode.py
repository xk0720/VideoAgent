#!/usr/bin/env python
"""离线蒸馏:从归档 run 目录的 storyboard.json 补记一条剧本级 episode。

用途(2026-08-13 用户令):
  • 回填历史成功案例进长期记忆(线上蒸馏在 --no-review 跑法下曾被跳过);
  • 产出 episode 示例文件供检查(--out)。

用法:
  python scripts/distill_episode.py --run outputs/movie_20260811_022309 \\
      --prompt "晨光面包店" --out docs/EPISODE_EXAMPLE.json
  不给 --memory 则只产示例文件,不写库;给了则追加进该 JSONL 库。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from maestro.memory.episode_memory import EpisodeMemory   # noqa: E402
from maestro.memory.storyboard import StoryboardMemory    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 目录(含 storyboard.json)")
    ap.add_argument("--prompt", required=True, help="当时的用户输入/片名")
    ap.add_argument("--out", default="", help="示例 JSON 输出路径(可选)")
    ap.add_argument("--memory", default="",
                    help="episode 库 JSONL 路径(可选;给了才真正入库)")
    args = ap.parse_args()

    sb_path = Path(args.run) / "storyboard.json"
    if not sb_path.exists():
        print(f"❌ 找不到 {sb_path}")
        return 2
    sb = StoryboardMemory.load(sb_path)
    mem = EpisodeMemory(Path(args.memory) if args.memory else None)
    final = Path(args.run) / "movie.mp4"
    rec = mem.distill_episode(args.prompt, sb,
                              final_video=str(final if final.exists()
                                              else ""))
    print(f"episode {rec.episode_id}: outcome={rec.outcome} "
          f"shots={rec.n_shots} steps={len(rec.trajectory)} "
          f"registry={len(rec.header.get('reference_registry', {}))}")
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(asdict(rec), ensure_ascii=False,
                                   indent=2))
        print(f"example → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
