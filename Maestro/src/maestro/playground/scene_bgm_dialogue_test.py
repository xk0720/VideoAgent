#!/usr/bin/env python
"""Playground:用户点名的两个简单思路,各一个子命令。

(1) bgm      —— scene 级背景音乐:模拟 brain 写一段音乐描述 →
               text-to-music 生成一条曲(scene 内所有 shot 共享);
               给了 --scene-video 就顺手完成混音(闪避+响度归一)试听。
(2) dialogue —— 音画同步(对口型)via 现有模型:Seedance 原生音频 +
               引号台词驱动嘴型;重点验证【能否只出人声、不带背景音】
               (prompt 里显式压制音乐/环境声,靠耳朵验收)。

用法:
    export WAVESPEED_API_KEY=...     # 或仓库根 .env

    # 思路(1):scene BGM(默认 sonilo/text-to-music;--engine ace_step 换最便宜档)
    python src/maestro/playground/scene_bgm_dialogue_test.py bgm \
        --desc "warm playful orchestral, light pizzicato, 95bpm, cozy morning" \
        --duration 15 [--scene-video outputs/attempt3/xxx.mp4] [--engine sonilo]

    # 思路(2):对白口型(--image 给近景首帧走 i2v 更稳;不给则 t2v)
    python src/maestro/playground/scene_bgm_dialogue_test.py dialogue \
        --line "Time for breakfast!" [--image cat_closeup.png] [--allow-bgm]

验收标准(耳朵+眼睛):
    bgm     ① 音乐符合描述(情绪/速度)② 时长对 ③ 混音后对白处音乐自动压低
    dialogue① 嘴动且与台词节奏贴 ② 除人声外是否安静(--allow-bgm 对照组
            会故意不压制,两次对听)③ 台词内容正确
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from maestro.models.video_gen_backends import WaveSpeedClient  # noqa: E402
from maestro.playground.bgm_mix_test import _mix, _probe       # noqa: E402

# 思路(2)的默认对白镜(与猫片世界一致;台词 ≤6 词是口型甜点)
DEFAULT_CHARACTER = ("close-up of a small orange-and-white shorthair cat "
                     "with amber eyes, facing the camera in a warm daylit "
                     "living room, fixed camera")
DEFAULT_LINE = "Time for breakfast!"
# 压制背景音的话术 —— 本实验的核心待验证项
AUDIO_ONLY_VOICE = ("Audio: only the character's voice speaking the line, "
                    "clean and prominent. No background music, no ambient "
                    "sound, no sound effects — silence except the voice.")

_ENGINES = {
    "sonilo": ("sonilo/text-to-music", 0.0025),          # $/音乐秒
    "ace_step": ("wavespeed-ai/ace-step-1.5", 0.02 / 60),
}


def _out_dir(tag: str) -> Path:
    d = REPO_ROOT / "outputs" / f"playground_{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_bgm(args) -> int:
    model_id, rate = _ENGINES[args.engine]
    est = round(rate * args.duration, 4)
    payload = {"prompt": args.desc, "duration": int(args.duration)}
    print(f"[bgm] engine={model_id} est=${est}\n[payload] "
          f"{json.dumps(payload, ensure_ascii=False)}")
    if args.dry_run:
        return 0
    out = _out_dir("scene_bgm")
    client = WaveSpeedClient({})
    music = out / "scene_music.mp3"
    client._run_task(model_id, payload, music)
    print(f"[music] {music}")
    report = {"mode": "bgm", "desc": args.desc, "engine": model_id,
              "duration": args.duration, "est_cost_usd": est,
              "music": str(music)}
    if args.scene_video:
        video = Path(args.scene_video).resolve()
        dur, has_audio = _probe(video)
        mixed = _mix(video, music, has_audio, dur, out)
        report.update(mixed=str(mixed), scene_video=str(video),
                      had_audio=has_audio)
        print(f"[mixed] {mixed}")
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {out / 'report.json'}\n"
          "[listen] ① 情绪/速度符合描述? ② 时长对? ③ 对白处音乐压低了?")
    return 0


def run_dialogue(args) -> int:
    prompt = (f"{args.character}. The cat looks at the camera and says: "
              f'"{args.line}". Its mouth moves with the words.')
    if not args.allow_bgm:
        prompt += " " + AUDIO_ONLY_VOICE
    est = round(0.092 * args.duration, 2)   # seedance 480p 实测价目
    print(f"[dialogue] i2v={bool(args.image)} allow_bgm={args.allow_bgm} "
          f"est=${est}\n[prompt] {prompt}")
    if args.dry_run:
        return 0
    out = _out_dir("dialogue")
    # 唯一与主管线不同的配置:generate_audio 打开(管线里默认关着)
    client = WaveSpeedClient({"generate_audio": True,
                              "resolution": args.resolution})
    clip = out / "dialogue.mp4"
    client.generate(
        prompt=prompt, duration=float(args.duration), out_path=clip,
        first_frame=(Path(args.image).resolve() if args.image else None),
        seed=args.seed)
    print(f"[clip] {clip}")
    # 抽出纯音轨方便对听;有无音轨也是第一道验收
    wav = out / "dialogue_audio.wav"
    got_audio = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-i", str(clip), "-vn", str(wav)],
        capture_output=True).returncode == 0
    report = {"mode": "dialogue", "prompt": prompt, "line": args.line,
              "image": args.image, "allow_bgm": args.allow_bgm,
              "duration": args.duration, "est_cost_usd": est,
              "clip": str(clip),
              "audio_track_present": got_audio and wav.exists(),
              "audio_wav": str(wav) if got_audio else None}
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audio] track_present={report['audio_track_present']} → {wav}\n"
          f"[report] {out / 'report.json'}\n"
          "[listen] ① 嘴动且贴台词节奏? ② 人声之外安静吗?"
          "(跑一次 --allow-bgm 当对照组两相对听) ③ 台词内容对?")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("bgm", help="scene 级 BGM:brain 描述 → text-to-music")
    b.add_argument("--desc", required=True,
                   help="音乐描述(模拟 brain 的 music_plan 输出)")
    b.add_argument("--duration", type=int, default=15)
    b.add_argument("--engine", default="sonilo", choices=sorted(_ENGINES))
    b.add_argument("--scene-video", default="",
                   help="可选:scene 拼接视频,给了就完成混音试听")
    b.add_argument("--dry-run", action="store_true")

    d = sub.add_parser("dialogue", help="对白口型:原生音频 + 引号台词")
    d.add_argument("--line", default=DEFAULT_LINE, help="台词(≤6 词)")
    d.add_argument("--character", default=DEFAULT_CHARACTER)
    d.add_argument("--image", default="", help="近景首帧(i2v,口型更稳)")
    d.add_argument("--allow-bgm", action="store_true",
                   help="对照组:不压制背景音,验证压制话术的效果差")
    d.add_argument("--duration", type=int, default=5)
    d.add_argument("--resolution", default="480p")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    return run_bgm(args) if args.mode == "bgm" else run_dialogue(args)


if __name__ == "__main__":
    raise SystemExit(main())
