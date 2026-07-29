"""§F 音频阶段(2026-07-29,用户批准的两条极简策略落地)。

(1) scene 级 BGM:剧本的 music_plan(scene 号 → brain 写的音乐描述)→
    每 scene 一次 text-to-music(scene 内所有 shot 共享一条曲,曲内自洽,
    天然规避"多段音乐互不一致");按 scene 起止拼成整片音乐床 → 混音。
(2) 对白混音配套:对白镜带原生音轨(window_loop 生成时开 generate_audio
    + prompt 压制背景音),这里负责 concat 前的音轨统一(无声镜补静音轨,
    否则 -c copy concat 会坏)与混音时的人声闪避。

确定性混音链(参数来源:OpenMontage 实测 + 平台标准,见
docs/MUSIC_INTEGRATION_2026_07_28.md):
    sidechaincompress(threshold 0.02, ratio 9, attack 200ms, release 500ms)
    + 两遍 loudnorm(I=-14 LUFS, LRA=11, TP=-1.5)

诚实链:music_plan 空 → 响亮记录"静音片"并原样返回;ffmpeg 缺失/任一
步失败 → 返回 None,调用方保留无配乐成片(配乐是增强层,绝不毁正片)。
依赖注意:滤镜图用到 apad=whole_dur / amix normalize=0,需要
**ffmpeg ≥ 4.4**;更老版本会在 _build_bed 处失败并按上述诚实链跳过配乐。
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..logging_utils import get_logger

log = get_logger("audio")

# 引擎表(极简版:sonilo 首选,ace_step 最便宜档;混音链与引擎无关)
ENGINES = {
    "sonilo": ("sonilo/text-to-music", 0.0025),          # ($/音乐秒)
    "ace_step": ("wavespeed-ai/ace-step-1.5", 0.02 / 60),
}


def probe(path: Path) -> tuple[float, bool]:
    """(时长秒, 是否有音轨);探测失败 → (0.0, False),不编。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        has_audio = any(s.get("codec_type") == "audio"
                        for s in data.get("streams", []))
        return dur, has_audio
    except Exception:
        return 0.0, False


def any_audio(clips: list) -> bool:
    return any(probe(Path(c))[1] for c in clips)


def normalize_for_concat(clips: list, work_dir: Path) -> list[Path]:
    """concat 前统一音轨:无声镜补静音 AAC,有声镜转码为同规格 AAC ——
    否则 `-c copy` 的 concat 在音轨参差时产出坏文件。任一失败 → 抛给
    调用方降级(保留原 clips 走旧路)。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, c in enumerate(clips):
        src = Path(c)
        dst = work_dir / f"norm_{i:03d}.mp4"
        _dur, has_audio = probe(src)
        if has_audio:
            cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(src),
                   "-c:v", "copy", "-c:a", "aac", "-ar", "48000",
                   "-ac", "2", "-b:a", "192k", str(dst)]
        else:
            cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(src),
                   "-f", "lavfi", "-i",
                   "anullsrc=channel_layout=stereo:sample_rate=48000",
                   "-shortest", "-c:v", "copy", "-c:a", "aac",
                   "-b:a", "192k", str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
        out.append(dst)
    return out


def scene_spans(storyboard) -> list[tuple[int, float, float]]:
    """[(scene 号, 起点秒, 时长秒)] —— 按台账顺序累加各镜【实际】时长
    (ffprobe 逐镜文件;任一镜探测不到 → RuntimeError,调用方跳过配乐
    —— 宁可不配,不配错位的)。"""
    spans: dict[int, list[float]] = {}
    order: list[int] = []
    t = 0.0
    for e in storyboard.entries:
        if not e.video_path:
            continue
        dur, _ = probe(Path(e.video_path))
        if dur <= 0:
            # 审查修正:0 长处理会让后续所有 scene 的起点整体前移,
            # 整张音乐床错位 —— 宁可不配乐,不配错位的乐。
            raise RuntimeError(
                f"shot {e.label} duration unknown — refusing to lay a "
                f"misaligned music bed")
        if e.scene_idx not in spans:
            spans[e.scene_idx] = [t, 0.0]
            order.append(e.scene_idx)
        spans[e.scene_idx][1] += dur
        t += dur
    return [(i, spans[i][0], spans[i][1]) for i in order]


def _loudnorm_two_pass(audio_in: Path, audio_out: Path) -> None:
    """两遍 loudnorm(先测量后线性应用;单遍有泵感)→ -14 LUFS。"""
    target = "I=-14:LRA=11:TP=-1.5"
    try:
        p1 = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(audio_in),
             "-af", f"loudnorm={target}:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True, check=True)
        tail = p1.stderr[p1.stderr.rfind("{"):p1.stderr.rfind("}") + 1]
        m = json.loads(tail)
        if any(str(m[k]) in ("-inf", "inf", "nan")
               for k in ("input_i", "input_lra", "input_tp",
                         "input_thresh")):
            raise ValueError("degenerate loudnorm measurement "
                             "(silent input?)")
        filt = (f"loudnorm={target}:measured_I={m['input_i']}:"
                f"measured_LRA={m['input_lra']}:"
                f"measured_TP={m['input_tp']}:"
                f"measured_thresh={m['input_thresh']}:"
                f"offset={m['target_offset']}:linear=true")
    except Exception as exc:
        # 审查修正:测量脆断不应放弃整个配乐阶段 —— 单遍兜底(有轻微
        # 泵感但成片可用),响亮记录降级。
        log.warning("audio: two-pass loudnorm measurement failed (%s) — "
                    "single-pass fallback", exc)
        filt = f"loudnorm={target}"
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(audio_in),
                    "-af", filt, str(audio_out)], check=True,
                   capture_output=True)


def _build_bed(tracks: list[tuple[Path, float, float]], total: float,
               out_wav: Path) -> None:
    """scene 曲目 → 整片音乐床:各曲裁到本 scene 时长、按起点延时,叠加。"""
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    parts = []
    for i, (path, start, dur) in enumerate(tracks):
        cmd += ["-i", str(path)]
        delay = int(round(start * 1000))
        parts.append(f"[{i}:a]atrim=duration={dur:.3f},apad="
                     f"whole_dur={dur:.3f},adelay={delay}|{delay}[t{i}]")
    joins = "".join(f"[t{i}]" for i in range(len(tracks)))
    fc = (";".join(parts)
          + f";{joins}amix=inputs={len(tracks)}:duration=longest:"
            f"normalize=0,atrim=duration={total:.3f}[bed]")
    cmd += ["-filter_complex", fc, "-map", "[bed]", str(out_wav)]
    subprocess.run(cmd, check=True, capture_output=True)


def _mix_onto(video: Path, bed: Path, has_dialogue_audio: bool,
              total: float, out_dir: Path, out_path: Path) -> Path:
    """音乐床 + 成片 → 终混:有人声则 sidechain 闪避,无则直铺;
    两遍 loudnorm 后封装(视频流 copy,不重编码画面)。"""
    mix_wav = out_dir / "mix_raw.wav"
    if has_dialogue_audio:
        # 审查修正:人声轨常比画面短几毫秒(concat 的 -shortest residue),
        # duration=first 会让终混提前收尾、封装时截掉画面尾巴 —— 人声轨
        # 也 apad,amix 取 longest,统一由 atrim/-t 收口。
        fc = ("[1:a]volume=0.9,apad[m];"
              "[0:a]apad,asplit=2[key][voice];"
              "[m][key]sidechaincompress="
              "threshold=0.02:ratio=9:attack=200:release=500[duck];"
              "[voice][duck]amix=inputs=2:duration=longest:normalize=0,"
              f"atrim=duration={total:.3f}[mix]")
        subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(video),
                        "-i", str(bed), "-filter_complex", fc,
                        "-map", "[mix]", "-t", f"{total:.3f}",
                        str(mix_wav)], check=True, capture_output=True)
    else:
        subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(bed),
                        "-af", "apad", "-t", f"{total:.3f}", str(mix_wav)],
                       check=True, capture_output=True)
    norm_wav = out_dir / "mix_norm.wav"
    _loudnorm_two_pass(mix_wav, norm_wav)
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(video),
                    "-i", str(norm_wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{total:.3f}", str(out_path)], check=True,
                   capture_output=True)
    return out_path


def add_music(final_video: Path, storyboard, video_gen, out_path: Path,
              engine: str = "sonilo",
              music_fn: Optional[Callable] = None) -> Optional[Path]:
    """给成片配乐(§F 主入口)。music_plan 空 → None(响亮记录,静音片
    照发);任一步失败 → None(增强层绝不毁正片)。

    music_fn(desc, duration_s, out_path) 可注入(测试/换引擎);缺省走
    video_gen._run_task(text-to-music)—— 调用自动进 wavespeed_calls 日志。
    """
    plan = dict(getattr(storyboard, "music_plan", {}) or {})
    if not plan:
        log.info("audio: music_plan is empty — shipping a silent film "
                 "(the script chose no music)")
        return None
    if not shutil.which("ffmpeg"):
        log.warning("audio: ffmpeg missing — skipping music stage")
        return None
    try:
        out_dir = out_path.parent / "audio_stage"
        out_dir.mkdir(parents=True, exist_ok=True)
        spans = scene_spans(storyboard)
        total, has_audio = probe(Path(final_video))
        if total <= 0:
            log.warning("audio: final video unreadable — skipping music")
            return None

        model_id, rate = ENGINES.get(engine, ENGINES["sonilo"])

        def _default_music(desc: str, dur_s: float, out: Path) -> Path:
            payload = {"prompt": desc, "duration": int(math.ceil(dur_s))}
            log.info("audio: %s scene track %.1fs (~$%.4f) — %s",
                     model_id, dur_s, rate * dur_s, desc[:80])
            return video_gen._run_task(model_id, payload, out)

        gen = music_fn or _default_music
        tracks: list[tuple[Path, float, float]] = []
        for scene_idx, start, dur in spans:
            desc = str(plan.get(scene_idx, "") or "").strip()
            if not desc or dur <= 0:
                if not desc:
                    log.info("audio: scene %d has no music description — "
                             "left silent", scene_idx)
                continue
            track = gen(desc, dur, out_dir / f"scene{scene_idx:02d}.mp3")
            tracks.append((Path(track), start, dur))
        if not tracks:
            return None
        bed = out_dir / "music_bed.wav"
        _build_bed(tracks, total, bed)
        return _mix_onto(Path(final_video), bed, has_audio, total,
                         out_dir, out_path)
    except Exception as exc:
        log.warning("audio: music stage failed (%s) — keeping the "
                    "unscored film", exc)
        return None
