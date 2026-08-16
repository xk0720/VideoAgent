#!/usr/bin/env python3
"""卡点强化: 在生成音乐的指定时刻叠加合成重击(impact), 把落点砸实。

为什么需要: 生成模型能把重音放在对的位置(实测偏差 0.035s), 但落点强度只有
全曲峰值的 0.2-0.3 —— "准但不狠"。这里用 numpy 合成真正有冲击力的 impact
叠上去, 位置精确到样本级。

impact = 三层叠加(经典电子鼓设计):
  · sub  55Hz→35Hz 扫频正弦 + 指数衰减 → 胸腔冲击的"轰"
  · body 180Hz 正弦快衰减        → 鼓皮的"咚"
  · tick 高通白噪 8ms            → 起音的"啪", 让落点在小音箱上也听得见

用法: python tools/punch_up.py in.mp3 out.wav 2,4,6 [gain]
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100


def synth_impact(dur: float = 0.55) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    # sub: 55Hz→35Hz 扫频, 指数衰减
    f = 55 * np.exp(-t * 3.0) + 35
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7.0)
    # body: 180Hz 快衰减
    body = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 26.0) * 0.35
    # tick: 8ms 高通白噪起音
    nt = int(SR * 0.008)
    tick = np.zeros(n)
    noise = np.random.default_rng(7).standard_normal(nt)
    tick[:nt] = np.convolve(noise, [1, -0.85], mode="same") * \
        np.exp(-np.arange(nt) / (SR * 0.002)) * 0.5
    x = sub + body + tick
    return x / (np.abs(x).max() + 1e-9)


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__); return 1
    src, dst = sys.argv[1], sys.argv[2]
    beats = [float(b) for b in sys.argv[3].split(",")]
    gain = float(sys.argv[4]) if len(sys.argv) > 4 else 0.85

    raw = Path(dst).with_suffix(".src.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ar", str(SR), "-ac", "2", str(raw)], check=True)
    with wave.open(str(raw), "rb") as w:
        frames, n = w.readframes(w.getnframes()), w.getnframes()
    music = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    music = music.reshape(-1, 2)

    imp = synth_impact()
    for b in beats:
        i = int(b * SR)
        j = min(len(music), i + len(imp))
        if i >= len(music):
            continue
        seg = imp[: j - i] * gain
        music[i:j, 0] += seg
        music[i:j, 1] += seg

    peak = np.abs(music).max()
    if peak > 0.99:                       # 软限幅, 避免削波
        music *= 0.99 / peak
    out = (music * 32767).astype(np.int16)
    with wave.open(dst, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(out.tobytes())
    raw.unlink(missing_ok=True)
    print(f"已强化 {len(beats)} 个落点 → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
