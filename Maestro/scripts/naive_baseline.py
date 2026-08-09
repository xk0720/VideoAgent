"""素朴基线(2026-08-09 用户令,对比实验):复用指定 run 的分镜台账,
逐镜【纯文字 t2v】直出 + 直接拼接。

- 人物:不挂肖像,正典 static 描述符以文字写进 prompt;
- 背景:不挂板,setting 文字描述进 prompt;
- 无派生/无钉帧/无承接/无评审;台词逐字进 prompt,对白镜开原生音频;
- 每镜时长对齐原 run 实际成片时长(公平对照)。

用法:python scripts/naive_baseline.py --src outputs/movie_XXX --tag s4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from maestro.models import build_video_gen                    # noqa: E402
from maestro.pipeline.audio_stage import (any_audio,          # noqa: E402
                                          normalize_for_concat)
from maestro.tools.video_concat import VideoConcatTool        # noqa: E402

_SHOT_PREFIX = re.compile(r"^\s*Shot\s+\d+\s*:\s*", re.IGNORECASE)


def _plain(text: str) -> str:
    """台账文本 → 素朴散文:剥 Shot 前缀/尖括号记号/旁白/音效标注。"""
    t = _SHOT_PREFIX.sub("", str(text or "").strip())
    t = re.sub(r"<([^<>]{1,24})>", r"\1", t)
    t = re.sub(r"(?:画外)?旁白[:：]?\s*[\"“][^\"“”]*[\"”]。?\s*", "", t)
    t = re.sub(r"(?:画外)?旁白[:：].*?(?=音效[:：]|$)", "", t, flags=re.S)
    t = re.sub(r"音效[:：][^。]*。?", "", t)
    return t.strip()


def _dur(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True)
    try:
        return max(3.0, float(out.stdout.strip()))
    except ValueError:
        return 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原 run 输出目录")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="configs/bailian.yaml")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(REPO / args.config))
    vg = build_video_gen(cfg["models"]["video_gen"])

    sb = json.load(open(Path(args.src) / "storyboard.json"))
    cast = sb.get("cast") or {}
    setting = _plain(sb.get("setting") or "")
    run_dir = REPO / "outputs" / (
        f"naive_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(vg, "set_call_log"):
        vg.set_call_log(run_dir / "calls.jsonl")

    clips = []
    for e in sb["entries"]:
        desc = _plain(e.get("description"))
        names = [n for n in cast if n in desc]
        # 人物 = 文字描述(正典 static 部分),不挂肖像
        who = "".join(
            f"{n}:{str(cast[n]).split(';')[0].replace('static:', '').strip()}"
            f"。" for n in names)
        dial = e.get("dialogue") or ""
        if isinstance(dial, dict):
            dial = str(dial.get("line") or "")
        spk = str(e.get("dialogue_speaker") or "")
        line = (f'{spk}说:"{dial}"。' if dial else "")
        want_audio = bool(dial) or ("声" in desc)
        audio = ("音频:只有角色对白的人声与画面内的自然环境声——无背景"
                 "音乐。" if want_audio else "")
        prompt = f"{desc}{line}{who}背景:{setting}{audio}"
        src_v = e.get("video_path")
        dur = _dur(Path(src_v)) if src_v and Path(src_v).exists() else 5.0
        out = run_dir / f"shot{e['shot_idx']:03d}.mp4"
        vg.generate_audio = want_audio
        print(f"[{args.tag}] shot {e['shot_idx']} dur={dur:.0f}s "
              f"audio={want_audio}\n  {prompt[:160]}", flush=True)
        clips.append(Path(vg.generate(prompt, dur, out, fps=24, seed=0)))

    concat_in = (normalize_for_concat(clips, run_dir / "concat_norm")
                 if any_audio(clips) else clips)
    movie = VideoConcatTool().run(concat_in, run_dir / "movie.mp4")
    print(f"[{args.tag}] 成片: {movie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
