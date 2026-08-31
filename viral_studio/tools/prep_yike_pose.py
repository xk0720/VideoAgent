#!/usr/bin/env python3
"""yike 卡点连拍资产预处理(一次性): 精修切点 → 裁UI → 拆13镜 → 短镜回文补帧 → 抽音轨。

产物落 memory/assets/yike_pose/:
  shots/sNN.mp4   驱动片段(无声, 已裁右侧UI, 均已补到 >=2.1s 满足 API 地板)
  bgm.m4a         原片完整音轨(纯音乐, 无口播)
  meta.json       每镜真实时长表(slot_durations) + 精修切点
切点表是量出来的(帧差), 烘死在这里; 运行时不做任何检测。
"""
import json
import subprocess
from pathlib import Path

import numpy as np

VS = Path(__file__).resolve().parents[1]
SRC = VS / "examples/new/yike/viral_video1.mp4"
OUT = VS / "memory/assets/yike_pose"
FPS = 30
# 镜头边界 = 用户逐帧裁决(2026-08-31), 闭区间帧号。此前帧差自动检测的边界
# 普遍晚约1帧, 片段首尾带上了邻镜的跳变帧 —— 帧号硬切在构造上杜绝串帧。
FRAME_RANGES = [(0, 24), (25, 42), (43, 59), (60, 89), (90, 126),
                (127, 195), (196, 208), (209, 224), (225, 256)]
CROP = "610:944:0:0"          # 只裁右侧UI列(按钮+头像脸, 免干扰检测); 左侧不裁, 避免切到走动的人
MIN_S = 2.1                   # API 时长地板(实测 2s, 留余量)


def frames_gray(t0: float, t1: float) -> np.ndarray:
    w, h = 96, 126
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(SRC),
         "-vf", f"fps={FPS},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    n = len(a) // (w * h)
    return a[:n * w * h].reshape(n, h, w).astype(np.float32)


def refine(t: float) -> float:
    """粗切点(10fps 分辨率) → 30fps 帧级: 在 ±0.25s 窗内找最大帧差。"""
    a = frames_gray(t - 0.25, t + 0.25)
    if len(a) < 3:
        return t
    d = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    # 不取全窗最大: 片尾闪变/淡出会盖过真切点。在 >=60% 峰值的候选里取离粗估最近的
    cand = [i for i, v in enumerate(d) if v >= d.max() * 0.6]
    mid = (len(d) - 1) / 2
    i = min(cand, key=lambda x: abs(x - mid))
    return round(t - 0.25 + (i + 1) / FPS, 3)


def main() -> int:
    shots = OUT / "shots"
    if shots.exists():                     # 旧边界的片段全部作废(用户裁决: 有串帧)
        for f in shots.iterdir():
            f.unlink()
    shots.mkdir(parents=True, exist_ok=True)
    durs = [round((b - a + 1) / FPS, 3) for a, b in FRAME_RANGES]
    print("帧号区间:", FRAME_RANGES)
    print("每镜时长:", durs)

    for i, (fa, fb) in enumerate(FRAME_RANGES, 1):
        raw = OUT / "shots" / f"s{i:02d}_raw.mp4"
        dst = OUT / "shots" / f"s{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(SRC), "-an",
                        "-vf", (f"select='between(n\,{fa}\,{fb})',"
                                f"setpts=N/{FPS}/TB,crop={CROP}"),
                        "-r", str(FPS),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        str(raw)], check=True)
        need = durs[i - 1]
        if need >= MIN_S:
            raw.rename(dst)
        else:
            # 回文补帧: 正放|倒放交替拼到 >=2.1s(文件级 concat, 稳)。第一段就是
            # 原始正放, 所以 assemble 取前 need 秒即原动作; 补的部分只为过 API
            # 地板, 不进成片。
            rev = raw.with_name(raw.stem + "_rev.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw),
                            "-vf", "reverse", "-c:v", "libx264", "-preset", "fast",
                            "-crf", "18", str(rev)], check=True)
            reps = int(np.ceil((MIN_S + 0.3) / need)) + 1
            lst = raw.with_suffix(".txt")
            lst.write_text("\n".join(
                f"file '{(raw if k % 2 == 0 else rev).resolve()}'"
                for k in range(reps)), encoding="utf-8")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                            "-safe", "0", "-i", str(lst), "-t", f"{MIN_S + 0.4:.2f}",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                            str(dst)], check=True)
            for f in (raw, rev, lst):
                f.unlink()
        got = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(dst)], capture_output=True, text=True).stdout)
        print(f"  s{i:02d}: 真实 {need:.2f}s → 驱动 {got:.2f}s"
              + ("  (回文补帧)" if need < MIN_S else ""))

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(SRC), "-vn",
                    "-c:a", "copy", str(OUT / "bgm.m4a")], check=True)
    total = round(sum(durs), 3)
    meta = {"source": str(SRC.relative_to(VS)), "frame_ranges": FRAME_RANGES,
            "slot_durations": durs, "total_s": total,
            "crop": CROP, "min_drive_s": MIN_S}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"完成: {len(durs)} 镜 + bgm.m4a + meta.json → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
