#!/usr/bin/env python3
"""卡点评测: 音乐的重音到底落在哪 —— 用起音检测客观打分, 不靠听感。

给定目标拍点(如 2/4/6s), 对每个候选音轨:
  1. librosa 起音强度包络 → 峰值时刻(onsets)
  2. 每个目标拍点找最近的强起音, 记偏差(秒)
  3. 目标点附近的相对起音强度(越强 = 砸得越实)
排序规则: 偏差小优先, 同档比落点强度。

用法: python tools/beat_check.py 2,4,6 a.mp3 b.mp3 ...
"""
import sys
from pathlib import Path

import librosa
import numpy as np


def analyze(path: str, targets: list) -> dict:
    y, sr = librosa.load(path, sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(env, sr=sr)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=0.25, wait=5)
    onset_t = times[peaks] if len(peaks) else np.array([])
    try:
        tempo = float(np.atleast_1d(librosa.feature.rhythm.tempo(
            onset_envelope=env, sr=sr))[0])
    except Exception:                                   # noqa: BLE001 旧版 API
        tempo = float(np.atleast_1d(librosa.beat.tempo(onset_envelope=env, sr=sr))[0])

    rows, devs, strengths = [], [], []
    for t in targets:
        if onset_t.size == 0:
            rows.append((t, None, None, 0.0)); devs.append(9.9); continue
        i = int(np.argmin(np.abs(onset_t - t)))
        near = float(onset_t[i]); dev = near - t
        j = int(np.argmin(np.abs(times - t)))
        w = env[max(0, j - 3):j + 4]
        rel = float(w.max() / (env.max() + 1e-9))
        rows.append((t, near, dev, rel))
        devs.append(abs(dev)); strengths.append(rel)
    return {"path": path, "duration": len(y) / sr, "tempo": tempo,
            "onsets": int(onset_t.size), "rows": rows,
            "mean_dev": float(np.mean(devs)),
            "mean_strength": float(np.mean(strengths)) if strengths else 0.0}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__); return 1
    targets = [float(x) for x in sys.argv[1].split(",")]
    reports = []
    for p in sys.argv[2:]:
        if not Path(p).exists():
            print(f"跳过(不存在): {p}"); continue
        r = analyze(p, targets); reports.append(r)
        print(f"\n=== {Path(p).name} ===")
        print(f"  时长 {r['duration']:.2f}s | 估计 BPM {r['tempo']:.0f} | 起音数 {r['onsets']}")
        for t, near, dev, rel in r["rows"]:
            if near is None:
                print(f"  目标 {t:.0f}s: 未检测到起音"); continue
            flag = "OK " if abs(dev) <= 0.12 else ("~  " if abs(dev) <= 0.25 else "BAD")
            print(f"  目标 {t:.0f}s: 最近起音 {near:5.2f}s (偏差 {dev:+.2f}s) "
                  f"强度 {rel:.2f} [{flag}]")
        print(f"  平均偏差 {r['mean_dev']:.3f}s | 平均落点强度 {r['mean_strength']:.2f}")
    if len(reports) > 1:
        best = sorted(reports, key=lambda r: (round(r["mean_dev"], 2),
                                              -r["mean_strength"]))[0]
        print(f"\n>>> 卡点最准: {Path(best['path']).name} "
              f"(平均偏差 {best['mean_dev']:.3f}s, 落点强度 {best['mean_strength']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
