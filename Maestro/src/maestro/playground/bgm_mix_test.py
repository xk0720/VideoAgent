#!/usr/bin/env python
"""Playground:BGM 工具菜单实测 —— 随机选一个音乐工具(模拟未来 brain 的
选择位),给成片配乐并完成确定性混音链。

流程(M0-3 实验的可执行版):
    成片 mp4 → 【随机选工具】→ WaveSpeed 生成 BGM →
    ffmpeg 混音(有对白音轨则 sidechain 闪避,无则直铺)→
    两遍 loudnorm 收口(-14 LUFS / TP -1.5)→ mixed.mp4 + report.json

工具菜单(与提案 docs/MUSIC_INTEGRATION_2026_07_28.md 对应):
    video_to_music  sonilo/video-to-music      $0.009/视频秒(自动贴节奏)
    text_to_music   sonilo/text-to-music       $0.0025/音乐秒(文字意图)
    ace_step        wavespeed-ai/ace-step-1.5  <$0.02/分钟(最便宜)

诚实注意:三个端点的 payload 字段名按 wavespeed 文档页编写,首跑即是
schema 验证 —— 若 API 报字段错,用 --extra-json 现场修正并回报文档;
--dry-run 只打印 payload 不花钱。混音链是确定性本地 ffmpeg,不属于
工具选择面。

用法:
    export WAVESPEED_API_KEY=...        # 或仓库根 .env
    python src/maestro/playground/bgm_mix_test.py \
        --video outputs/attempt3/final_movie.mp4 \
        [--tool random|video_to_music|text_to_music|ace_step] \
        [--music-prompt "..."] [--seed 7] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from maestro.models.video_gen_backends import (  # noqa: E402
    WaveSpeedClient,
    upload_media,
)

# 猫片风格的默认音乐意图(文字类工具用;--music-prompt 覆盖)
DEFAULT_MUSIC_PROMPT = (
    "warm playful orchestral, light pizzicato strings, gentle woodwinds, "
    "95bpm, cozy morning mood, soft build, instrumental, no vocals"
)


def _probe(video: Path) -> tuple[float, bool]:
    """(时长秒, 是否有音轨)。探测失败按 (0.0, False) 诚实返回。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(video)],
            capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        has_audio = any(s.get("codec_type") == "audio"
                        for s in data.get("streams", []))
        return dur, has_audio
    except Exception as exc:
        print(f"[warn] ffprobe failed: {exc}")
        return 0.0, False


def _build_call(tool: str, video: Path, video_url: str, dur: float,
                music_prompt: str) -> tuple[str, dict, float]:
    """→ (model_id, payload, 成本估算$)。字段名按 wavespeed 文档页;
    首跑即 schema 验证。"""
    n = max(1, int(round(dur)))
    if tool == "video_to_music":
        return ("sonilo/video-to-music",
                {"video": video_url, "prompt": music_prompt},
                round(0.009 * n, 3))
    if tool == "text_to_music":
        return ("sonilo/text-to-music",
                {"prompt": music_prompt, "duration": n},
                round(0.0025 * n, 3))
    if tool == "ace_step":
        return ("wavespeed-ai/ace-step-1.5",
                {"prompt": music_prompt, "duration": n},
                round(0.02 * n / 60.0, 4))
    raise SystemExit(f"unknown tool: {tool}")


def _loudnorm_two_pass(audio_in: Path, audio_out: Path) -> dict:
    """两遍 loudnorm:先测量后线性应用(单遍会有泵感,C 报告结论)。"""
    target = "I=-14:LRA=11:TP=-1.5"
    p1 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(audio_in),
         "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    tail = p1.stderr[p1.stderr.rfind("{"):p1.stderr.rfind("}") + 1]
    m = json.loads(tail)
    filt = (f"loudnorm={target}:measured_I={m['input_i']}:"
            f"measured_LRA={m['input_lra']}:measured_TP={m['input_tp']}:"
            f"measured_thresh={m['input_thresh']}:"
            f"offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(audio_in),
                    "-af", filt, str(audio_out)], check=True,
                   capture_output=True)
    return m


def _mix(video: Path, music: Path, has_audio: bool, dur: float,
         out_dir: Path) -> Path:
    """确定性混音链:闪避(有对白)/直铺(无)→ 两遍 loudnorm → 封装。"""
    mix_wav = out_dir / "mix_raw.wav"
    if has_audio:
        # 原音轨(含对白/音效)做 sidechain 钥匙,把 BGM 压下去;
        # threshold/ratio 取 OpenMontage 实测参数,attack/release 单位 ms。
        fc = ("[1:a]volume=0.9,apad[a1];"
              "[0:a]asplit=2[key][voice];"
              "[a1][key]sidechaincompress="
              "threshold=0.02:ratio=9:attack=200:release=500[duck];"
              "[voice][duck]amix=inputs=2:duration=first:normalize=0[mix]")
        subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(video),
                        "-i", str(music), "-filter_complex", fc,
                        "-map", "[mix]", "-t", f"{dur:.3f}", str(mix_wav)],
                       check=True, capture_output=True)
    else:
        subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(music),
                        "-af", "apad", "-t", f"{dur:.3f}", str(mix_wav)],
                       check=True, capture_output=True)
    norm_wav = out_dir / "mix_norm.wav"
    measured = _loudnorm_two_pass(mix_wav, norm_wav)
    print(f"[loudnorm] measured I={measured['input_i']} LUFS "
          f"→ target -14 LUFS (two-pass linear)")
    mixed = out_dir / "mixed.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(video),
                    "-i", str(norm_wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(mixed)], check=True,
                   capture_output=True)
    return mixed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--tool", default="random",
                    choices=["random", "video_to_music", "text_to_music",
                             "ace_step"])
    ap.add_argument("--music-prompt", default=DEFAULT_MUSIC_PROMPT)
    ap.add_argument("--seed", type=int, default=None,
                    help="随机选择的种子(可复现)")
    ap.add_argument("--extra-json", default="",
                    help='payload 补丁,如 \'{"video_url": "..."}\'(schema 修正用)')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    dur, has_audio = _probe(video)
    print(f"[probe] duration={dur:.2f}s audio_stream={has_audio}")

    # 【模拟 brain 的位置】真实管线里这一步是 brain 按菜单+skill 决策;
    # playground 先随机,验证的是"每条工具路径都能走通、都能混出片"。
    rng = random.Random(args.seed)
    tool = rng.choice(["video_to_music", "text_to_music", "ace_step"]) \
        if args.tool == "random" else args.tool
    print(f"[choose] 模拟 brain 选择 → {tool}"
          + (f" (seed={args.seed})" if args.seed is not None else ""))

    client = WaveSpeedClient({})
    video_url = ""
    if tool == "video_to_music" and not args.dry_run:
        video_url = upload_media(client.api_key, video)
        print(f"[upload] {video_url[:60]}...")
    model_id, payload, est = _build_call(tool, video, video_url, dur,
                                         args.music_prompt)
    if args.extra_json:
        payload.update(json.loads(args.extra_json))
    print(f"[call] model={model_id} est=${est}\n"
          f"[payload] {json.dumps(payload, ensure_ascii=False)[:300]}")
    if args.dry_run:
        print("[dry-run] not submitting")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / f"playground_bgm_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    music = out_dir / "music.mp3"
    t0 = time.time()
    client._run_task(model_id, payload, music)
    print(f"[music] {music} ({time.time() - t0:.1f}s)")

    mixed = _mix(video, music, has_audio, dur, out_dir)
    report = {
        "video": str(video), "duration_s": dur, "had_audio": has_audio,
        "tool": tool, "model_id": model_id, "payload": payload,
        "est_cost_usd": est, "music": str(music), "mixed": str(mixed),
        "mix_chain": ("sidechain duck (thr 0.02, ratio 9, 200/500ms) + "
                      "two-pass loudnorm -14 LUFS" if has_audio
                      else "music bed + two-pass loudnorm -14 LUFS"),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {mixed}\n[report] {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
