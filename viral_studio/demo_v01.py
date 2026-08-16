#!/usr/bin/env python3
"""Demo 2(规则写死): v01 复刻 —— 三人依次入场开场 + 三段外景旁白解说 = 36s。

结构:
  A  0-6s   三人依次入场(叠加式) —— seedance t2v, 4 张参考图
            @Image1 = v01 原片开场干净背景(手机边框+粉绿放射+星光贴纸)
            @Image2/3/4 = 三个人物 hook; 0-2/2-4/4-6s 逐一弹入并留在画面
            音频 = v01 BGM[0-6s] + 合成 pop 音效卡在 2/4/6s(音乐生成API不可用)
            标题后期烧(用户裁决), 不指望模型写字
  B1 6-16s  粉色款 · 樱花外景
  B2 16-26s 浅绿款 · 草地外景
  B3 26-36s 白蓝款 · 海边外景
            generate_audio=false(画面纯净无声) → 旁白走 MiniMax TTS,
            字幕后期 ffmpeg drawtext 烧(100%准确, 不让模型写中文),
            混音 = TTS 旁白 + v01 BGM 压低

与 Demo 1 的差别: 不要求音画同步 → 动作彻底放开; 音频/字幕全部后期可控。

用法: python demo_v01.py [--dry-run] [--yes] [--reuse-gen DIR]
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from studio.backends.minimax_tts import MiniMaxTTSClient    # noqa: E402
from studio.backends.seedance import SeedanceClient          # noqa: E402
from studio.config import load_dotenv                        # noqa: E402

log = logging.getLogger("demo")

SRC_WAV = Path("/Users/kevin/Desktop/viral_studio/"
               "lQbPJxIr6dUx9e0AALAYjGtLmthYAgnZ42flSooA.wav")   # v01 分离产物: 4ch
BGM_CHANNELS = (0, 1)   # v01 分离产物的 BGM 在 ch0/1(与 v02 的 ch2/3 相反!):
                        # 开场无人声段 ch0/1=-18.1dB vs ch2/3=-26.7dB, 且 ASR 证实
                        # 全片无人声VO(两个 stem 都只转出 whisper 幻觉)
OPENING_BG = HERE / "examples/bg/v01_opening_bg.jpg"
HOOKS = [HERE / f"examples/hooks/person_hook_{i}.png" for i in (1, 2, 3)]

W, H, FPS = 720, 1280, 30
V_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
A_ENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
BGM_VOL_A = 1.0            # 开场: BGM 给足
BGM_VOL_B = 0.20           # 解说段: 压低让位旁白
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"   # 中文字幕字体(本机无 PingFang)
TITLE = "夏日穿搭合集"
VOICE = "Lively_Girl"      # 用户裁决: 年轻活泼

OPEN_TAIL = 0.5   # 开场段尾部余量: 给 6s 那一击留衰减空间

OPENING = {
    "duration": 6,
    "beats": [2, 4, 6],    # 合成 pop 音效的落点 = 人物弹入时刻
    "windows": [
        (0, 2, "the woman from @Image2 pops into the center of the phone frame, "
               "landing in a confident standing pose with one hand on her hip"),
        (2, 4, "the woman from @Image3 pops in beside her on the left; both "
               "stay on screen holding their poses"),
        (4, 6, "the woman from @Image4 pops in on the right; all three stand "
               "in a row facing the camera, smiling"),
    ],
}

# 三段外景: 背景按配色推导(粉→樱花, 绿→草地, 白蓝→海边)
SEGMENTS = [
    {"name": "B1_pink", "hook": HOOKS[0], "color": "pink",
     "scene": "under blooming cherry blossom trees in a spring park, soft warm "
              "sunlight filtering through pink petals, gentle breeze",
     "action": "she walks toward the camera and stops, tilts her head with a "
               "bright surprised smile, sweeps a strand of hair behind her ear, "
               "then spreads the hem of the sweatshirt with both hands and "
               "gives a small playful shrug",
     "line": "先说这件粉色，水洗棉的面料，上身软得像云一样，穿一天都不闷。"},
    {"name": "B2_green", "hook": HOOKS[1], "color": "soft sage green",
     "scene": "on a grassy lawn beside a tree-lined path in a botanical garden, "
              "dappled green light, blurred foliage in the background",
     "action": "she strolls a few steps along the path, turns back to the "
               "camera with a mischievous raised eyebrow, points at the "
               "contrast stitching on her cuff, then slips both hands into her "
               "pockets and rocks her shoulders",
     "line": "浅绿这件最显白，领口和袖口的撞色缝线，细节做得特别用心。"},
    {"name": "B3_blue", "hook": HOOKS[2], "color": "white and soft blue",
     "scene": "on a white terrace overlooking the sea, clear blue sky and "
              "ocean horizon behind her, bright natural daylight",
     "action": "she leans on the railing and looks out at the sea, turns "
               "around with wind in her hair, opens both arms wide to present "
               "the outfit, then makes a heart shape with her hands",
     "line": "白蓝色最百搭，上课通勤周末都能穿。三个颜色一个价，链接放下面了。"},
]


def run(cmd) -> None:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(map(str,cmd))[:160]}\n{p.stderr[-500:]}")


def build_opening_prompt() -> str:
    head = ("Locked-off static camera, vertical 9:16. The exact background from "
            "@Image1 — a phone frame over a pink-and-green pastel sunburst with "
            "sparkle stickers — stays fixed and unchanged for the whole clip. "
            "A 6-second video.")
    body = [f"{t0}-{t1}s: {desc}." for t0, t1, desc in OPENING["windows"]]
    tail = ("Each woman keeps her own face, hairstyle and outfit exactly as in "
            "her reference image. No morphing between people, no blending of "
            "faces, no extra people. Cut-out sticker look with a thin white "
            "outline around each woman, matching the reference background's "
            "collage style. No text, no captions, no camera movement.")
    return " ".join([head, *body, tail])


def build_segment_prompt(seg: dict) -> str:
    return (f"Handheld camera with a subtle natural sway, vertical 9:16 full-body "
            f"shot, {seg['scene']}. The woman from @Image1, wearing the "
            f"{seg['color']} sweatshirt with a watercolor horse print, is the "
            f"only person in frame. Over the 10 seconds, {seg['action']}. Her "
            f"expressions are lively and varied — she smiles, raises her "
            f"eyebrows, and looks straight into the lens. Cinematic natural "
            f"light, shallow depth of field. Same woman and same outfit "
            f"throughout. No text, no captions, no subtitles, no extra people.")


# ── 音频 ─────────────────────────────────────────────────────────
def cut_bgm(t0: float, t1: float, dst: Path, vol: float = 1.0) -> Path:
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}",
         "-i", SRC_WAV,
         "-af", f"pan=stereo|c0=c{BGM_CHANNELS[0]}|c1=c{BGM_CHANNELS[1]},volume={vol}",
         "-ar", "44100", dst])
    return dst


MUSIC_PROMPT = (
    "Upbeat fashion-haul intro, 120 BPM, punchy electronic drums, strong kick "
    "and clap accents landing exactly on 2s, 4s and 6s, short riser sweeps "
    "building into each accent, bright synth plucks, crisp hi-hats, energetic "
    "and clean, no vocals.")


def make_opening_music(audio_dir: Path, beats: list, dur: float) -> Path:
    """开场卡点音乐 = sonilo/text-to-music 生成 + 落点 impact 强化。

    实测(2026-08-16, 目标拍点 2/4/6s):
      text-to-music   偏差 0.035s, onset 强度 0.28  ← 最准, 选它
      video-to-music  偏差 0.085s, 6s 处强度 0.03(几乎无重音)
      自造鼓点         偏差 0.070s, 4s/6s 落点很弱
      "生成8s裁6s"     反而更差(生成物自身 BPM 89, 与 2/4/6s 不对齐)
    生成的音乐"准但不狠"(落点低频能量 0.38) → punch_up 叠合成 impact 后 0.88,
    6s 处 0.05→0.73 提升最大(那正是生成模型最弱的收尾处)。
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw = audio_dir / "music_raw.mp3"
    dst = audio_dir / "music_opening.wav"
    from studio.backends.sonilo_music import SoniloMusicClient
    c = SoniloMusicClient(api_key=os.environ.get("WAVESPEED_API_KEY", ""))
    # 生成比成片段落更长: 最后一击(6s)需要 ~0.55s 衰减空间, 否则拼接后被截掉
    ok, tid, err = c.text_to_music(MUSIC_PROMPT, int(dur) + 2, str(raw))
    if not ok:
        log.warning("音乐生成失败(%s), 退回原片 BGM", err[:120])
        return cut_bgm(0, dur, audio_dir / "bgm_A.wav", BGM_VOL_A)
    punched = audio_dir / "music_punched.wav"
    subprocess.run([sys.executable, str(HERE / "tools/punch_up.py"), str(raw),
                    str(punched), ",".join(str(b) for b in beats), "0.85"],
                   capture_output=True, text=True, check=True)
    # 裁到 段长+0.5s: 保住最后一击的衰减尾, 同时不带入多余乐句
    run(["ffmpeg", "-y", "-v", "error", "-i", punched, "-t", f"{dur + OPEN_TAIL:.3f}",
         "-ar", "44100", dst])
    log.info("开场音乐: 生成 + %d 个落点强化 → %s", len(beats), dst.name)
    return dst


def build_beat_track(bgm: Path, beats: list, dst: Path, dur: float) -> Path:
    """卡点音效: BGM 打底 + 每个 beat 处叠一个合成 pop(音乐生成API不可用, 自己造)。
    pop = 1200Hz 短正弦 + 快速衰减包络, 听感接近贴纸弹出音。"""
    pops = []
    for i, b in enumerate(beats):
        p = dst.parent / f"_pop{i}.wav"
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "sine=frequency=1200:duration=0.18",
             "-af", "afade=t=out:st=0.02:d=0.16,volume=0.5", "-ar", "44100", p])
        pops.append((b, p))
    inputs, filters, mixes = ["-i", str(bgm)], [], ["[0:a]"]
    for i, (b, p) in enumerate(pops, start=1):
        inputs += ["-i", str(p)]
        delay_ms = int(max(0.0, b - 0.18) * 1000)      # pop 落在 beat 上(提前一个音长起音)
        filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[p{i}]")
        mixes.append(f"[p{i}]")
    fc = ";".join(filters) + ";" + "".join(mixes) + \
         f"amix=inputs={len(pops)+1}:duration=first:dropout_transition=0[a]"
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc,
         "-map", "[a]", "-t", f"{dur:.3f}", "-ar", "44100", dst])
    return dst


# ── 画面装配 ─────────────────────────────────────────────────────
def make_text_png(text: str, dst: Path, size: int, y_frac: float,
                  box: bool = True, max_chars: int = 14) -> Path:
    """用 Pillow 渲染字幕层(透明 PNG, 整幅 720x1280)。

    本机 ffmpeg 未编译 drawtext(无 libfreetype) → 走 overlay 叠图这条路,
    反而更可控: 字体/描边/底条/换行全在 Python 里算, 且中文 100% 准确
    (对比 Demo 1 让生成模型写中文, 会写出语义错乱的字形)。
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, size)
    lines, cur = [], ""
    for ch in text:                                   # 按字数简单折行(中文等宽)
        cur += ch
        if len(cur) >= max_chars and ch in "，。！？、 ":
            lines.append(cur.strip("，。 ")); cur = ""
    if cur:
        lines.append(cur.strip("，。 "))
    lh = int(size * 1.45)
    total_h = lh * len(lines)
    y0 = int(H * y_frac) - total_h // 2
    if box:
        widths = [d.textlength(l, font=font) for l in lines]
        pad = 22
        d.rounded_rectangle(
            [(W - max(widths)) / 2 - pad, y0 - pad,
             (W + max(widths)) / 2 + pad, y0 + total_h + pad // 2],
            radius=18, fill=(0, 0, 0, 105))
    for i, line in enumerate(lines):
        w = d.textlength(line, font=font)
        d.text(((W - w) / 2, y0 + i * lh), line, font=font, fill=(255, 255, 255, 255),
               stroke_width=max(3, size // 14), stroke_fill=(0, 0, 0, 230))
    img.save(dst)
    return dst


def conform(video: Path, dst: Path, duration: float, audio: Path,
            subtitle: str = "", title: str = "") -> Path:
    """统一规格 + 掐时长 + overlay 烧字(字幕/标题) + 配音轨。"""
    base = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration=2")
    layers, inputs, idx = [], ["-i", str(video), "-i", str(audio)], 2
    if subtitle:
        p = dst.parent / f"_sub_{dst.stem}.png"
        make_text_png(subtitle, p, size=40, y_frac=0.83, max_chars=13)
        inputs += ["-i", str(p)]; layers.append(idx); idx += 1
    if title:
        p = dst.parent / f"_title_{dst.stem}.png"
        make_text_png(title, p, size=76, y_frac=0.16, box=False, max_chars=10)
        inputs += ["-i", str(p)]; layers.append(idx); idx += 1

    fc, cur = f"[0:v]{base}[v0]", "v0"
    for n, i in enumerate(layers, start=1):
        fc += f";[{cur}][{i}:v]overlay=0:0[v{n}]"
        cur = f"v{n}"
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc,
         "-map", f"[{cur}]", "-map", "1:a:0", "-t", f"{duration:.3f}",
         *V_ENC, *A_ENC, "-shortest", dst])
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--reuse-gen", default=None, help="复用已生成视频, 只重做音频/字幕/装配")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    load_dotenv()

    for p in (SRC_WAV, OPENING_BG, *HOOKS):
        if not Path(p).exists():
            log.error("素材缺失: %s", p)
            return 1
    out = Path(args.out) if args.out else HERE / "outputs" / f"demo01_{time.strftime('%Y%m%d_%H%M%S')}"
    (out / "gen").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    log.info("输出目录: %s", out)

    op_prompt = build_opening_prompt()
    seg_prompts = [build_segment_prompt(s) for s in SEGMENTS]
    (out / "prompts.json").write_text(json.dumps(
        {"opening": op_prompt,
         "segments": [{"name": s["name"], "prompt": p, "line": s["line"]}
                      for s, p in zip(SEGMENTS, seg_prompts)]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("── 开场 prompt ──\n%s", op_prompt)
    for s, p in zip(SEGMENTS, seg_prompts):
        log.info("── %s ──\n%s\n   旁白: %s", s["name"], p, s["line"])

    if args.dry_run:
        log.info("dry-run 结束: 计费 ≈ 6s(开场) + 30s(三段) + 3×TTS")
        return 0
    if not args.reuse_gen and not args.yes:
        if input("将调用 4×seedance(6s+3×10s) + 3×TTS, 约36计费秒, 继续? [y/N] "
                 ).strip().lower() != "y":
            return 0

    # ── 生成画面 ─────────────────────────────────────────────────
    results = {}
    if args.reuse_gen:
        g = Path(args.reuse_gen)
        for key, fname in [("A", "A_opening.mp4")] + [(s["name"], f"{s['name']}.mp4")
                                                      for s in SEGMENTS]:
            p = g / fname
            results[key] = {"ok": p.exists(), "raw": str(p), "err": "", "task": "reused"}
        log.info("复用 %s 的 %d 个片段", g, sum(1 for r in results.values() if r["ok"]))
    else:
        sc = SeedanceClient(api_key=os.environ.get("WAVESPEED_API_KEY", ""),
                            resolution="720p", generate_audio=False)

        def gen_opening():
            raw = out / "gen/A_opening.mp4"
            refs = [str(OPENING_BG)] + [str(h) for h in HOOKS]
            ok, tid, err = sc.generate(op_prompt, OPENING["duration"], str(raw),
                                       reference_images=refs)
            return "A", {"ok": ok, "task": tid, "err": err, "raw": str(raw)}

        def gen_seg(i):
            s = SEGMENTS[i]
            raw = out / f"gen/{s['name']}.mp4"
            ok, tid, err = sc.generate(seg_prompts[i], 10, str(raw),
                                       reference_images=[str(s["hook"])])
            return s["name"], {"ok": ok, "task": tid, "err": err, "raw": str(raw)}

        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(gen_opening)] + [pool.submit(gen_seg, i) for i in range(3)]
            for f in as_completed(futs):
                k, rec = f.result()
                results[k] = rec
                log.info("%s → %s %s", k, "succeeded" if rec["ok"] else "FAILED",
                         rec["err"][:100])

    # ── 旁白 TTS ────────────────────────────────────────────────
    tts = MiniMaxTTSClient(api_key=os.environ.get("WAVESPEED_API_KEY", ""),
                           voice_id=args.voice)
    voices = {}
    for s in SEGMENTS:
        dst = out / f"audio/{s['name']}_vo.mp3"
        ok, tid, err = tts.speak(s["line"], str(dst))
        voices[s["name"]] = str(dst) if ok else ""
        log.info("TTS %s → %s %s", s["name"], "ok" if ok else "FAILED", err[:80])

    # ── 音频与装配 ───────────────────────────────────────────────
    parts = []
    if results.get("A", {}).get("ok"):
        beat_a = make_opening_music(out / "audio", OPENING["beats"], 6.0)
        # 段长 = 6s + 尾部余量: 末帧克隆定格(三人同框), 让 6s 那一击响完
        parts.append(conform(Path(results["A"]["raw"]), out / "conform_A.mp4",
                             6.0 + OPEN_TAIL, beat_a, title=TITLE))
    for i, s in enumerate(SEGMENTS):
        rec = results.get(s["name"], {})
        if not rec.get("ok"):
            continue
        t0 = 6 + i * 10
        bgm = cut_bgm(t0, t0 + 10, out / f"audio/bgm_{s['name']}.wav", BGM_VOL_B)
        mixed = out / f"audio/{s['name']}_mix.wav"
        if voices[s["name"]]:
            run(["ffmpeg", "-y", "-v", "error", "-i", voices[s["name"]], "-i", bgm,
                 "-filter_complex",
                 "[0:a]loudnorm=I=-18:TP=-1.5:LRA=11,apad[v];[1:a]anull[b];"
                 "[v][b]amix=inputs=2:duration=shortest:dropout_transition=0[a]",
                 "-map", "[a]", "-t", "10", "-ar", "44100", mixed])
        else:
            mixed = bgm
        parts.append(conform(Path(rec["raw"]), out / f"conform_{s['name']}.mp4",
                             10.0, mixed, subtitle=s["line"]))

    if not parts:
        log.error("无成功片段")
        return 1
    lst = out / "concat.txt"
    lst.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in parts), encoding="utf-8")
    final = out / "demo_final.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
         *V_ENC, *A_ENC, final])
    dur = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(final)], capture_output=True,
                         text=True).stdout.strip()
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    log.info("成片: %s (%.1fs, %d/4 段)", final, float(dur or 0), len(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
