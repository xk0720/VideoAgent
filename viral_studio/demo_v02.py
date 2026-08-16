#!/usr/bin/env python3
"""Demo(规则写死, 不走 agent): v02 卡点开场 + 三人物中文口播 = 35s 成片。

结构:
  A  0-5s   海报卡点开场 —— animate 动作驱动(wan-pro)
            驱动 = memory/assets/media/v02_poster_beat_reel.mp4(实测 pass_verified)
            参考 = person_hook_1;  音频 = 源 BGM[0-5s] 全量
  B1 5-15s  粉色款口播 —— seedance t2v 10s, 三动作(3+3+4), 中文台词, 音画同出
  B2 15-25s 浅绿款口播 —— 同上, person_hook_2
  B3 25-35s 白蓝款口播 —— 同上, person_hook_3
            音频 = seedance 人声 + 源 BGM[对应时段] 低音量混入

BGM: 用户分离的 6ch wav(5.1 容器塞了 3 个 stem), 实测 ch2/ch3 = 纯 BGM
     (ch0/1 是人声: 开场 -70dB / 口播 -34dB; ch4/5 是残留空轨)。
     按时间轴切片: A 用 [0,5), B1/B2/B3 用 [5,15)/[15,25)/[25,35), 盈余丢弃。

用法: python demo_v02.py [--dry-run] [--yes] [--skip-a] [--skip-b]
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

from studio.backends.bailian_animate import BailianAnimateClient   # noqa: E402
from studio.backends.seedance import SeedanceClient                # noqa: E402
from studio.config import load_dotenv                              # noqa: E402

log = logging.getLogger("demo")

SRC_WAV = Path("/Users/kevin/Desktop/viral_studio/"
               "lQbPJxIr6dUx9e0AALAYjGtLmthYAgnZ42flSooA.mp4.wav")  # 名字带错, 内容是 v02(50.2s)
BGM_CHANNELS = (2, 3)          # 实测: 纯 BGM stem
DRIVING = HERE / "memory/assets/media/v02_poster_beat_reel.mp4"
HOOKS = [HERE / f"examples/hooks/person_hook_{i}.png" for i in (1, 2, 3)]

W, H, FPS = 720, 1280, 30
V_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
A_ENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
BGM_VOL_TALK = 0.22            # 口播段 BGM 压低, 别盖住人声
VOICE_GAIN = 1.3               # 分离几乎无损(-18.2→-18.4dB), 只补 amix 归一的损失

# ── 三段口播: 一个配色一段, 卖点递进(面料 → 设计 → 场景+价格+CTA) ──────
SCENE = ("bright clean bedroom with a large window, soft daylight, "
         "plain light-colored wall")
TALKS = [
    {"name": "B1_pink", "hook": HOOKS[0], "color": "pink",
     "shots": [
         (0, 3, "walks in from the left edge of the frame, stops at the center "
                "and throws both arms open in a cheerful ta-da presentation, "
                "eyes wide with excitement",
          "这件卫衣我真的穿到不想脱"),
         (3, 6, "grabs the hem with both hands and swings it lightly left and "
                "right to show the loose cut, glances down at the print then "
                "snaps her eyes back up to the camera",
          "你看这个水彩小马印花"),
         (6, 10, "flicks one wide sleeve through the air, pivots about 45 "
                 "degrees to her side, then looks back over her shoulder and "
                 "winks at the camera",
          "水洗棉的面料，上身软得像云一样"),
     ]},
    {"name": "B2_green", "hook": HOOKS[1], "color": "soft sage green",
     "shots": [
         (0, 3, "leans slightly toward the camera and taps the contrast "
                "stitching on her collar with one finger, tilting her head",
          "同款还有这个浅绿色，超显白"),
         (3, 6, "slides both hands into her pockets and rocks her shoulders "
                "playfully from side to side, chin lifted",
          "你看领口和袖口的撞色缝线"),
         (6, 10, "takes two easy steps to her left, turns back to the camera "
                 "and sweeps her hair off her shoulder with one hand",
          "随便配条裤子就很好看，学生党闭眼入"),
     ]},
    {"name": "B3_blue", "hook": HOOKS[2], "color": "white and soft blue",
     "shots": [
         (0, 3, "holds up three fingers close to the camera, then spreads both "
                "arms wide to present the whole outfit",
          "三个颜色我全都买了"),
         (3, 6, "plants one hand on her hip and points straight down toward "
                "the bottom of the frame with the other, giving a playful wink",
          "上课通勤周末，穿它都不出错"),
         (6, 10, "turns halfway away, looks back over her shoulder with a "
                 "smile, then spins her hands into a small heart shape in "
                 "front of her chest",
          "三个颜色一个价，链接就在下面"),
     ]},
]


def build_prompt(talk: dict) -> str:
    """三动作 + 中文台词的分镜式 prompt(整数时间戳; 相机指令第一句)。"""
    head = (f"Locked-off static camera, vertical 9:16 full-body shot, {SCENE}. "
            f"A 10-second video in three shots, hard cut between shots. "
            f"The same young woman from @Image1, wearing the {talk['color']} "
            f"sweatshirt with a watercolor horse print, appears in every shot.")
    body = []
    for i, (t0, t1, action, line) in enumerate(talk["shots"], 1):
        body.append(f"Shot {i} ({t0}-{t1}s): she {action}, speaking to the "
                    f"camera in Mandarin Chinese, and says: \"{line}\"")
    tail = ("Same woman, same outfit, same room in all three shots. She speaks "
            "continuously with natural, accurate lip sync in Mandarin Chinese; "
            "clear female voice, upbeat and friendly, energetic performance. "
            "AUDIO: her speaking voice ONLY — a clean dry studio voice "
            "recording. No background music, no soundtrack, no instrumental, "
            "no ambient noise, no sound effects. "
            "SUBTITLES: burn in her spoken Chinese line as a caption at the "
            "bottom of the frame, white bold sans-serif text with a thin dark "
            "outline, one line at a time, changing with each shot, matching "
            "exactly what she is saying. No other text or graphics, no extra "
            "people.")
    return " ".join([head, *body, tail])


def isolate_voice(src: Path, dst: Path) -> Path:
    """人声分离(demucs htdemucs): 生成音轨里模型自带的BGM实测有 -30dB 能量,
    与我们自己的 BGM 会打架 → 只留 vocals stem。失败则原样返回(响亮告警)。"""
    work = dst.parent / "_demucs"
    try:
        # -d cpu: MPS 上 htdemucs 的 conv1d 输出通道超限(实测 NotImplementedError),
        # 10s 音频走 CPU 也就几十秒
        subprocess.run([sys.executable, "-m", "demucs.separate", "-n", "htdemucs",
                        "-d", "cpu", "--two-stems", "vocals",
                        "-o", str(work), str(src)],
                       capture_output=True, text=True, check=True)
        # 按 src 文件名精确定位: rglob 取首个匹配会让多段共用同一条人声轨
        # (2026-08-16 踩过 —— 三段成片配了同一个人的语音, 口型全错)
        voc = work / "htdemucs" / src.stem / "vocals.wav"
        if not voc.exists():
            raise FileNotFoundError(f"未找到分离产物: {voc}")
        run(["ffmpeg", "-y", "-v", "error", "-i", voc, "-ar", "44100", "-ac", "2", dst])
        log.info("人声分离完成: %s", dst.name)
        return dst
    except Exception as e:                              # noqa: BLE001
        log.warning("人声分离失败(%s), 退回原始音轨(会带模型自生成的BGM)", str(e)[:120])
        return src


def run(cmd) -> None:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(map(str,cmd))[:160]}\n{p.stderr[-500:]}")


def cut_bgm(t0: float, t1: float, dst: Path) -> Path:
    """从 6ch wav 取 BGM stem(ch2/3) 的 [t0,t1) 切片。"""
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}",
         "-i", SRC_WAV, "-af", f"pan=stereo|c0=c{BGM_CHANNELS[0]}|c1=c{BGM_CHANNELS[1]}",
         "-ar", "44100", dst])
    return dst


def conform(video: Path, dst: Path, duration: float, bgm: Path,
            voice: Path = None) -> Path:
    """统一规格+掐时长。voice 给定时(口播段): 用分离出的纯人声 + 我们的 BGM 混音;
    否则(卡点段): 只铺 BGM, 丢弃视频自带音轨。"""
    vf = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
          f"tpad=stop_mode=clone:stop_duration=2")
    if voice is not None:
        # loudnorm: 各段生成的人声响度差可达 11dB(实测 B1 -30.8 vs B2 -19.4),
        # 不归一会听成"忽大忽小" → 统一到 -18 LUFS 再混 BGM
        fc = (f"[1:a]loudnorm=I=-18:TP=-1.5:LRA=11,volume={VOICE_GAIN}[v];"
              f"[2:a]volume={BGM_VOL_TALK}[b];"
              f"[v][b]amix=inputs=2:duration=first:dropout_transition=0,"
              f"aresample=44100[a]")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", voice, "-i", bgm,
               "-filter_complex", fc, "-map", "0:v:0", "-map", "[a]"]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", bgm,
               "-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-vf", vf, "-t", f"{duration:.3f}", *V_ENC, *A_ENC, "-shortest", dst]
    run(cmd)
    return dst


def assemble(results: dict, out: Path, bgm_a: Path, bgm_b: list) -> int:
    """装配: A(BGM全量) + B1/B2/B3(分离出的纯人声 + BGM压低) → 35s 成片。"""
    parts = []
    if results.get("A", {}).get("ok"):
        parts.append(conform(Path(results["A"]["raw"]), out / "conform_A.mp4",
                             5.0, bgm_a))
    for i, t in enumerate(TALKS):
        rec = results.get(t["name"], {})
        if not rec.get("ok"):
            continue
        raw_wav = out / f"audio/{t['name']}_raw.wav"     # 生成物的完整音轨
        run(["ffmpeg", "-y", "-v", "error", "-i", rec["raw"], "-vn",
             "-ar", "44100", "-ac", "2", raw_wav])
        voice = isolate_voice(raw_wav, out / f"audio/{t['name']}_voice.wav")
        parts.append(conform(Path(rec["raw"]), out / f"conform_{t['name']}.mp4",
                             10.0, bgm_b[i], voice=voice))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只出 prompt 与音频切片, 不调用模型")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--skip-a", action="store_true", help="跳过 animate 段")
    ap.add_argument("--reuse-a", default=None,
                    help="复用已生成的 A 段 mp4(省 5 计费秒)")
    ap.add_argument("--reuse-gen", default=None,
                    help="复用某次运行的 gen/ 目录, 只重做音频处理与装配(零成本)")
    ap.add_argument("--skip-b", action="store_true", help="跳过口播段")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    load_dotenv()

    for p in (SRC_WAV, DRIVING, *HOOKS):
        if not Path(p).exists():
            log.error("素材缺失: %s", p)
            return 1
    out = Path(args.out) if args.out else HERE / "outputs" / f"demo_{time.strftime('%Y%m%d_%H%M%S')}"
    (out / "gen").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    log.info("输出目录: %s", out)

    # ── 音频切片: A[0,5) + B1/B2/B3[5,15)/[15,25)/[25,35) ──────────────
    bgm_a = cut_bgm(0, 5, out / "audio/bgm_A.wav")
    bgm_b = [cut_bgm(5 + i * 10, 15 + i * 10, out / f"audio/bgm_B{i+1}.wav")
             for i in range(3)]
    log.info("BGM 切片就绪(源 ch%d/%d): A[0-5s] + B[5-35s], 盈余 %.1fs 丢弃",
             *BGM_CHANNELS, 50.2 - 35)

    prompts = [build_prompt(t) for t in TALKS]
    (out / "prompts.json").write_text(json.dumps(
        [{"name": t["name"], "hook": str(t["hook"]), "prompt": p,
          "lines": [s[3] for s in t["shots"]]} for t, p in zip(TALKS, prompts)],
        ensure_ascii=False, indent=2), encoding="utf-8")
    for t, p in zip(TALKS, prompts):
        log.info("── %s ──\n%s", t["name"], p)

    if args.dry_run:
        log.info("dry-run 结束: 计费预估 = 5s(animate wan-pro) + 30s(seedance) = 35s")
        return 0

    if args.reuse_gen:                       # 零成本重装配: 复用已生成的视频片段
        g = Path(args.reuse_gen)
        results = {"A": {"ok": (g / "A_poster.mp4").exists(), "task": "reused",
                         "err": "", "raw": str(g / "A_poster.mp4")}}
        for t in TALKS:
            p = g / f"{t['name']}.mp4"
            results[t["name"]] = {"ok": p.exists(), "task": "reused",
                                  "err": "" if p.exists() else "缺片段", "raw": str(p)}
        log.info("复用 %s 的 %d 个片段, 仅重做音频与装配", g,
                 sum(1 for r in results.values() if r["ok"]))
        return assemble(results, out, bgm_a, bgm_b)

    if not args.yes:
        if input("将真实调用 1×animate(wan-pro,5s) + 3×seedance(10s), 约35计费秒, 继续? [y/N] "
                 ).strip().lower() != "y":
            return 0

    # ── 生成 ─────────────────────────────────────────────────────────
    results: dict = {}
    if args.reuse_a:
        results["A"] = {"ok": True, "task": "reused", "err": "", "raw": args.reuse_a}
        log.info("A 段复用已有素材: %s", args.reuse_a)
    elif not args.skip_a:
        ac = BailianAnimateClient(api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                                  mode="wan-pro")
        raw_a = out / "gen/A_poster.mp4"
        ok, tid, err = ac.animate(str(HOOKS[0]), str(DRIVING), str(raw_a))
        results["A"] = {"ok": ok, "task": tid, "err": err, "raw": str(raw_a)}
        log.info("A 卡点开场 → %s %s", "succeeded" if ok else "FAILED", err[:100])

    if not args.skip_b:
        sc = SeedanceClient(api_key=os.environ.get("WAVESPEED_API_KEY", ""),
                            resolution="720p", generate_audio=True)

        def gen(i: int):
            t, prompt = TALKS[i], prompts[i]
            raw = out / f"gen/{t['name']}.mp4"
            ok, tid, err = sc.generate(prompt, 10, str(raw),
                                       reference_images=[str(t["hook"])])
            return t["name"], {"ok": ok, "task": tid, "err": err, "raw": str(raw)}

        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in as_completed({pool.submit(gen, i) for i in range(3)}):
                name, rec = f.result()
                results[name] = rec
                log.info("%s → %s %s", name, "succeeded" if rec["ok"] else "FAILED",
                         rec["err"][:100])

    return assemble(results, out, bgm_a, bgm_b)


if __name__ == "__main__":
    raise SystemExit(main())
