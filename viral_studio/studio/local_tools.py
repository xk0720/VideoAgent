"""本地工具层 —— 调用计划里 local: true 的那些步骤。

实现逐条来自两条 demo 的实战代码, 连同当时踩出来的参数一并保留:
  · isolate_voice  demucs 必须走 CPU(MPS 上 conv1d 通道超限); 按文件名精确取产物
    (rglob 取首个会让多段共用同一条人声轨 —— 踩过, 口型全错)
  · burn_subtitle  本机 ffmpeg 未编译 drawtext → Pillow 渲染 PNG 再 overlay,
    中文 100% 准确(让视频模型写中文会写错字形)
  · mix_audio      人声先 loudnorm 再混 BGM(各段人声响度差可达 11dB)
  · punch_up       生成音乐"准而不狠" → 叠合成 impact 把落点砸实
所有产出统一 720x1280@30fps, 音轨 44.1kHz 立体声, 便于 concat。
"""
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("viral_studio")

W, H, FPS = 720, 1280, 30
V_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
A_ENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
VF_NORM = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,"
           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
           f"tpad=stop_mode=clone:stop_duration=2")


def run(cmd) -> None:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))[:180]}\n{p.stderr[-600:]}")


def probe_duration(path) -> float:
    p = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(p.stdout.strip() or 0)


# ── 音频 ──────────────────────────────────────────────────
def isolate_voice(source: str, out: Path, engine: str = "demucs",
                  device: str = "cpu", **_) -> str:
    """剥离视频自带音轨里的背景音乐, 只留人声。

    生成模型在音画同出时会自配 BGM(实测静默窗仍有 -30dB 能量), 与我们自己的
    音乐叠加会变成两首曲子同响。
    """
    work = out.parent / "_demucs"
    raw = out.parent / f"{out.stem}_raw.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", source, "-vn",
         "-ar", "44100", "-ac", "2", raw])
    try:
        subprocess.run([sys.executable, "-m", "demucs.separate", "-n", "htdemucs",
                        "-d", device, "--two-stems", "vocals", "-o", str(work), str(raw)],
                       capture_output=True, text=True, check=True)
        # 按文件名精确定位: rglob 取首个会让多段共用同一条人声轨
        voc = work / "htdemucs" / raw.stem / "vocals.wav"
        if not voc.exists():
            raise FileNotFoundError(f"未找到分离产物 {voc}")
        run(["ffmpeg", "-y", "-v", "error", "-i", voc, "-ar", "44100", "-ac", "2", out])
        return str(out)
    except Exception as e:                                  # noqa: BLE001
        log.warning("人声分离失败(%s), 退回原始音轨", str(e)[:120])
        return str(raw)


def concat_audio(audios: List[str], out: Path, slots: Optional[List[dict]] = None,
                 duration: Optional[float] = None, **_) -> str:
    """把多条语音按各自起点落到一条时间轴上(段落式旁白)。"""
    dur = float(duration or 0) or sum(probe_duration(a) for a in audios)
    inputs, filters, mixes = [], [], []
    for i, a in enumerate(audios):
        inputs += ["-i", str(a)]
        t0 = float((slots[i] if slots and i < len(slots) else {}).get("t0", 0))
        filters.append(f"[{i}:a]adelay={int(t0*1000)}|{int(t0*1000)}[a{i}]")
        mixes.append(f"[a{i}]")
    fc = ";".join(filters) + ";" + "".join(mixes) + \
         f"amix=inputs={len(audios)}:duration=longest:dropout_transition=0[out]"
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc,
         "-map", "[out]", "-t", f"{dur:.3f}", "-ar", "44100", out])
    return str(out)


def punch_up(audio: str, out: Path, beats: List[float], gain: float = 0.85,
             trim_to: Optional[float] = None, **_) -> str:
    """在指定拍点叠合成重击, 把生成音乐的落点砸实(低频冲击 0.38→0.88)。"""
    tool = Path(__file__).resolve().parents[1] / "tools" / "punch_up.py"
    subprocess.run([sys.executable, str(tool), str(audio), str(out),
                    ",".join(str(b) for b in beats), str(gain)],
                   capture_output=True, text=True, check=True)
    if trim_to:
        cut = out.with_name(out.stem + "_cut.wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", out, "-t", f"{float(trim_to):.3f}",
             "-ar", "44100", cut])
        return str(cut)
    return str(out)


# ── 画面 ──────────────────────────────────────────────────
def has_audio(path) -> bool:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return bool(p.stdout.strip())


def concat_av(videos: List[str], out: Path, voices: Optional[List[str]] = None,
              **_) -> str:
    """把多段视频顺序拼接; 给了 voices 就逐段换成对应人声轨再拼。

    两处必须显式处理, 否则整片会坏在看不见的地方:
      · 无音轨的片段(纯画面收尾)要铺静音 —— 否则 concat 出来的成片音轨会在
        它之前就结束, 尾巴几秒彻底没声。
      · 时长要用 -t 明确切死, 不能靠 -shortest —— 没有音轨时它无从约束,
        VF_NORM 里那 2 秒 tpad 补帧会原样漏进成片, 变成结尾冻帧。
    """
    parts = []
    for i, v in enumerate(videos):
        dst = out.parent / f"{out.stem}_p{i}.mp4"
        vdur = probe_duration(v)
        voice = voices[i] if voices and i < len(voices) and voices[i] else None
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(v)]
        if voice:                                   # 人声可略长: tpad 的 2s 用来兜住
            target = min(probe_duration(voice), vdur + 2.0)
            cmd += ["-i", str(voice), "-map", "0:v:0", "-map", "1:a:0"]
        elif has_audio(v):
            target = vdur
            cmd += ["-map", "0:v:0", "-map", "0:a:0"]
        else:
            target = vdur
            cmd += ["-f", "lavfi", "-t", f"{vdur:.3f}",
                    "-i", "anullsrc=r=44100:cl=stereo",
                    "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-vf", VF_NORM, *V_ENC, *A_ENC, "-t", f"{target:.3f}", str(dst)]
        run(cmd)
        parts.append(dst)
    lst = out.with_suffix(".txt")
    lst.write_text("\n".join(f"file '{p.resolve()}'" for p in parts), encoding="utf-8")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         *V_ENC, *A_ENC, str(out)])
    return str(out)


def mix_audio(video: str, out: Path, voice: Optional[str] = None,
              bgm=None, duration: Optional[float] = None,
              voice_loudnorm: Optional[float] = None,
              bgm_volume: float = 1.0, **_) -> str:
    """给画面配音轨: 人声(可选)+ BGM(可选), 统一规格并掐到目标时长。

    bgm 可以是文件路径, 也可以是 {source, t0, t1} —— 后者表示从整片共享音轨里切片。
    """
    dur = float(duration or probe_duration(video))
    bgm_path = None
    if isinstance(bgm, dict) and bgm.get("source"):
        bgm_path = out.parent / f"{out.stem}_bgm.wav"
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{float(bgm['t0']):.3f}",
             "-t", f"{float(bgm['t1']) - float(bgm['t0']):.3f}",
             "-i", str(bgm["source"]), "-vn", "-ar", "44100", "-ac", "2", bgm_path])
    elif bgm:
        bgm_path = Path(bgm)

    inputs = ["-i", str(video)]
    chains, mixes = [], []
    idx = 1
    if voice:
        inputs += ["-i", str(voice)]
        ln = (f"loudnorm=I={voice_loudnorm}:TP=-1.5:LRA=11,"
              if voice_loudnorm is not None else "")
        chains.append(f"[{idx}:a]{ln}aresample=44100[v]")
        mixes.append("[v]"); idx += 1
    if bgm_path:
        inputs += ["-i", str(bgm_path)]
        chains.append(f"[{idx}:a]volume={bgm_volume},aresample=44100[b]")
        mixes.append("[b]"); idx += 1

    cmd = ["ffmpeg", "-y", "-v", "error", *inputs]
    if mixes:
        fc = ";".join(chains)
        if len(mixes) > 1:
            fc += ";" + "".join(mixes) + \
                  "amix=inputs=2:duration=first:dropout_transition=0,aresample=44100[a]"
            amap = "[a]"
        else:
            amap = mixes[0]
        cmd += ["-filter_complex", fc, "-map", "0:v:0", "-map", amap]
    else:                                   # 无音轨 → 铺静音, 保证 concat 流一致
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-map", "0:v:0", "-map", f"{idx}:a:0"]
    cmd += ["-vf", VF_NORM, "-t", f"{dur:.3f}", *V_ENC, *A_ENC, "-shortest", str(out)]
    run(cmd)
    return str(out)


def _text_png(text: str, dst: Path, size: int, y_frac: float,
              box: bool = True, max_chars: int = 14) -> Path:
    """Pillow 渲染字幕层 —— 本机 ffmpeg 无 drawtext, 且中文由我们自己写才准确。"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, size)
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= max_chars and ch in "，。！？、 ":
            lines.append(cur.strip("，。 ")); cur = ""
    if cur:
        lines.append(cur.strip("，。 "))
    lh = int(size * 1.45)
    y0 = int(H * y_frac) - lh * len(lines) // 2
    if box and lines:
        wmax = max(d.textlength(l, font=font) for l in lines)
        pad = 22
        d.rounded_rectangle([(W - wmax) / 2 - pad, y0 - pad,
                             (W + wmax) / 2 + pad, y0 + lh * len(lines) + pad // 2],
                            radius=18, fill=(0, 0, 0, 105))
    for i, line in enumerate(lines):
        w = d.textlength(line, font=font)
        d.text(((W - w) / 2, y0 + i * lh), line, font=font, fill=(255, 255, 255, 255),
               stroke_width=max(3, size // 14), stroke_fill=(0, 0, 0, 230))
    img.save(dst)
    return dst


def burn_text(video: str, out: Path, text: str, y_frac: float = 0.16,
              size: int = 76, **_) -> str:
    """整段常驻的标题。"""
    png = _text_png(text, out.parent / f"{out.stem}_title.png", size, y_frac,
                    box=False, max_chars=10)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(png),
         "-filter_complex", "[0:v][1:v]overlay=0:0[v]", "-map", "[v]", "-map", "0:a?",
         *V_ENC, "-c:a", "copy", str(out)])
    return str(out)


def burn_subtitle(video: str, out: Path, text: Optional[str] = None,
                  segments: Optional[List[dict]] = None, y_frac: float = 0.83,
                  size: int = 40, max_chars: int = 13, **_) -> str:
    """字幕: 给 segments 就按时间段分别显示, 只给 text 就全程常驻。"""
    segs = segments or ([{"t0": 0, "t1": probe_duration(video), "text": text}]
                        if text else [])
    if not segs:
        return str(video)
    inputs, chains, cur = ["-i", str(video)], [], "0:v"
    for i, s in enumerate(segs, start=1):
        png = _text_png(str(s["text"]), out.parent / f"{out.stem}_sub{i}.png",
                        size, y_frac, max_chars=max_chars)
        inputs += ["-i", str(png)]
        nxt = f"v{i}"
        chains.append(f"[{cur}][{i}:v]overlay=0:0:"
                      f"enable='between(t,{float(s['t0'])},{float(s['t1'])})'[{nxt}]")
        cur = nxt
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(chains),
         "-map", f"[{cur}]", "-map", "0:a?", *V_ENC, "-c:a", "copy", str(out)])
    return str(out)


REGISTRY = {"isolate_voice": isolate_voice, "concat_audio": concat_audio,
            "punch_up": punch_up, "concat_av": concat_av, "mix_audio": mix_audio,
            "burn_text": burn_text, "burn_subtitle": burn_subtitle}
