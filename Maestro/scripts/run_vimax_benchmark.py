#!/usr/bin/env python
"""vimax_benchmark 测试驱动(2026-08-13 用户令,用户自跑)。

流程:读 ZH 题库(scripts/translate_benchmark.py 的产物)→ 逐故事
把预分镜 JSON 适配成我们的剧本契约(带显式镜头结构的 content 文本,
scene_write 的 PRE-STORYBOARDED SCRIPT LAW 负责结构照抄)→ 串行调
test_window_movie 真跑 → 每故事一个输出目录 → 汇总表落盘。

用法:
  python scripts/run_vimax_benchmark.py --pilot         # 每型 1 个,共 3
  python scripts/run_vimax_benchmark.py                 # 全量 35 个
  python scripts/run_vimax_benchmark.py --only chef_international_kitchens_typeA
  python scripts/run_vimax_benchmark.py --review        # 开评审(默认关,省钱)
  # 断点续跑(以盘上文件名为准):<story>*/movie.mp4 在 = 完成,直接
  # 跳过;半截目录留在原地,重跑进新目录 <story>_r2…;--redo 强制重来

产物(固定目录 outputs/benchmark/,换批次实验用 --out-root 另指):
  outputs/benchmark/<story>[_rN]/...(逐故事 run 目录,movie.mp4 在内)
  outputs/benchmark/summary.json(登记簿;丢了可按盘上文件自动补记)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def adapt_story(d: dict) -> str:
    """预分镜 benchmark JSON → 我们的剧本 content 文本。

    形状:总览一句 + 逐场逐镜"场景N 镜头M:【开场画面】…【本镜动作】…"
    —— 镜头结构显式可见,scene_write 按预分镜法照抄切分,只做标注。"""
    lines = [f"故事总览:{d.get('story_overview', '').strip()}", ""]
    for sc in d.get("scenes", []):
        n = sc.get("scene_num")
        for sh in sc.get("shots", []):
            lines.append(
                f"场景{n} 镜头{sh.get('shot_id')}:"
                f"【开场画面】{sh.get('first_frame', '').strip()}"
                f"【本镜动作】{sh.get('video_prompt', '').strip()}")
            lines.append("")
    return "\n".join(lines).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(REPO / "vimax_benchmark_zh"),
                    help="题库目录(默认中文版;想跑英文原版就指过去)")
    ap.add_argument("--config",
                    default=str(REPO / "configs/bailian.yaml"))
    ap.add_argument("--out-root",
                    default=str(REPO / "outputs" / "benchmark"),
                    help="固定目录(2026-08-13 用户令:断点续跑以【盘上"
                         "文件名】为准 —— movie.mp4 在即完成,汇总表"
                         "丢了照样接得上;换批次实验才另指目录)")
    ap.add_argument("--pilot", action="store_true",
                    help="每型抽第 1 个,共 3 故事(全链验证档)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--review", action="store_true",
                    help="开评审/修复(默认 --no-review 省钱)")
    ap.add_argument("--redo", action="store_true",
                    help="忽略汇总表里的完成记录,强制重跑")
    ap.add_argument("--audio", action="store_true", default=True)
    args = ap.parse_args()

    bench = Path(args.bench)
    idx_p = bench / "benchmark_index.json"
    if not idx_p.exists():
        print(f"❌ 题库缺 index:{idx_p}(先跑 translate_benchmark.py,"
              f"或 --bench vimax_benchmark 跑英文原版)")
        return 2
    idx = json.loads(idx_p.read_text())

    stories = list(idx["stories"])
    if args.only:
        keep = set(args.only)
        stories = [s for s in stories
                   if s["file"].replace(".json", "") in keep]
    elif args.pilot:
        seen_types: set = set()
        picked = []
        for s in stories:
            if s["type"] not in seen_types:
                picked.append(s)
                seen_types.add(s["type"])
        stories = picked

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_p = out_root / "summary.json"
    summary = (json.loads(summary_p.read_text())
               if summary_p.exists() else {})

    def _done_movie(key: str):
        """完成判据 = 文件名:key*/movie.mp4 存在即完成(重试目录
        key_r2… 也认);summary 只是登记簿,盘上文件才是事实。"""
        hits = sorted(out_root.glob(f"{key}*/movie.mp4"))
        return hits[-1] if hits else None

    def _next_run_dir(key: str) -> Path:
        """每次重跑用【新目录】(项目军规):key → key_r2 → key_r3…
        半截目录留在原地供验尸,绝不覆盖。"""
        d = out_root / key
        n = 2
        while d.exists():
            d = out_root / f"{key}_r{n}"
            n += 1
        return d

    for s in stories:
        key = s["file"].replace(".json", "")
        prior = _done_movie(key)
        if not args.redo and prior is not None:
            if summary.get(key, {}).get("status") != "ok":
                # 汇总表缺账但片在 —— 按盘补记(断点续跑的事实源)
                summary[key] = {"type": s["type"], "status": "ok",
                                "movie": str(prior),
                                "run_dir": str(prior.parent),
                                "recovered_from_disk": True}
                summary_p.write_text(json.dumps(summary,
                                                ensure_ascii=False,
                                                indent=2))
            print(f"skip (movie exists): {key}", flush=True)
            continue
        story_p = bench / s["file"]
        if not story_p.exists():
            summary[key] = {"status": "missing_translation"}
            continue
        d = json.loads(story_p.read_text())
        n_shots = sum(len(sc.get("shots", []))
                      for sc in d.get("scenes", []))
        # 适配成剧本契约文件(role 空 —— benchmark 无钦定角色图)
        screenplay = {"content": adapt_story(d), "role": {}}
        sp_path = out_root / f"{key}.screenplay.json"
        sp_path.write_text(json.dumps(screenplay, ensure_ascii=False,
                                      indent=2))
        run_dir = _next_run_dir(key)
        cmd = [sys.executable, "scripts/test_window_movie.py",
               "--config", args.config,
               "--screenplay", str(sp_path),
               "--prompt", d.get("story_overview", key)[:60],
               "--out-dir", str(run_dir),
               "--prompt-enhancer", "--n-candidates", "1"]
        if args.audio:
            cmd.append("--audio")
        if not args.review:
            cmd.append("--no-review")
        print(f"\n===== RUN {key} ({s['type']}, {n_shots} shots) =====",
              flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=REPO)
        movie = run_dir / "movie.mp4"
        dur = None
        if movie.exists():
            pr = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=nw=1:nk=1",
                 str(movie)], capture_output=True, text=True)
            try:
                dur = round(float(pr.stdout.strip()), 1)
            except ValueError:
                dur = None
        summary[key] = {
            "type": s["type"],
            "status": "ok" if r.returncode == 0 and movie.exists()
                      else f"exit_{r.returncode}"
                      + ("" if movie.exists() else "_no_movie"),
            "n_shots_requested": n_shots,
            "movie": str(movie) if movie.exists() else "",
            "duration_s": dur,
            "run_dir": str(run_dir),
            "wall_minutes": round((time.time() - t0) / 60, 1),
        }
        summary_p.write_text(json.dumps(summary, ensure_ascii=False,
                                        indent=2))
        print(f"===== {key}: {summary[key]['status']} "
              f"({summary[key]['wall_minutes']} min) =====", flush=True)

    ok = sum(1 for v in summary.values() if v.get("status") == "ok")
    print(f"\nSUMMARY: {ok}/{len(summary)} ok → {summary_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
