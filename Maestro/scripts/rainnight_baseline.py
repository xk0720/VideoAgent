#!/usr/bin/env python3
"""rainnight 基线对照版 × WaveSpeed(2026-08-06 用户令,体例同
xiaoming_baseline.py):与 rainnight_manual.py 同 6 镜、同素材,但
【零钉帧、零转场桥、prompt 无任何首帧/续接约束、随手粗写】,每镜
独立 t2v,直接硬拼。模型降档:seedance-v1-lite 480p(404 逐级回退)。

用法: python scripts/rainnight_baseline.py
输出: outputs/rainnight_baseline_<ts>/
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.models.video_gen import build_video_gen
from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
from maestro.pipeline.window_loop import _probe_seconds
from maestro.tools.video_concat import VideoConcatTool

SRC = Path("/Users/kevin/Desktop/Kevin/repositories/VideoAgent/Maestro/"
           "outputs/movie_20260806_010135")
BG_PLATE = SRC / "anchors" / "bg_bg_1.png"
PORTRAITS = {"黑帮老大": SRC / "portraits" / "黑帮老大.png",
             "女子": SRC / "portraits" / "女子.png"}

# 便宜差档候选(前者 404 → 依次回退保 run 通)
CHEAP_MODELS = ["bytedance/seedance-v1-lite-t2v-480p",
                "bytedance/seedance-v1-pro-t2v-480p",
                "bytedance/seedance-2.0/text-to-video"]

# 全部独立 t2v,prompt 随手粗写;@Image1=背景板,人物顺延
SHOTS = [
    dict(cast=["黑帮老大"], duration=8,
         prompt=("@Image1的雨夜暗巷,黑色轿车停着。车里@Image2点了根雪茄,"
                 "红光一明一灭,雨水顺着车窗流。"
                 "音频:只有雨滴敲击车窗声和雪茄吸入声——无背景音乐、无人声。")),
    dict(cast=["女子"], duration=5,
         prompt=("@Image1的雨夜。副驾驶上@Image2穿着晚礼服,看起来又紧张"
                 "又害怕,霓虹光照在脸上。"
                 "音频:只有低沉心跳声——无背景音乐、无人声。")),
    dict(cast=["黑帮老大", "女子"], duration=5,
         prompt=("@Image1的雨夜车里。@Image2拿金色手枪抵住@Image3的额头,"
                 "她在哭。音频:只有冰冷金属摩擦声——无背景音乐、无人声。")),
    dict(cast=["女子"], duration=5,
         prompt=("@Image1的雨夜。车窗突然碎了,玻璃碎片飞起来,@Image2"
                 "一脸绝望。音频:只有玻璃碎裂声——无背景音乐、无人声。")),
    dict(cast=["黑帮老大"], duration=5,
         prompt=("@Image1的雨夜。@Image2开枪,金色枪口火焰照亮巷子,火星"
                 "乱溅。音频:只有突发的巨大枪声——无背景音乐、无人声。")),
    dict(cast=["黑帮老大", "女子"], duration=5,
         prompt=("@Image1的雨夜。烟雾散开,@Image2和@Image3的轮廓在余光里"
                 "不动。音频:只有金属回音——无背景音乐、无人声。")),
]


def main() -> None:
    assert BG_PLATE.exists(), f"背景板缺失: {BG_PLATE}"
    for n, p in PORTRAITS.items():
        assert p.exists(), f"肖像缺失: {n} → {p}"
    out_dir = (Path("outputs")
               / f"rainnight_baseline_{time.strftime('%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    model_id = None
    vg = None
    for cand in CHEAP_MODELS:
        vg = build_video_gen({"name": "wavespeed", "model_id": cand,
                              "resolution": "480p",
                              "call_log": str(out_dir /
                                              "wavespeed_calls.jsonl")})
        # 探针:第一镜提交成功即锁定该模型;失败 → 换下一档
        try:
            sh = SHOTS[0]
            vg.generate_audio = True
            vg.generate(sh["prompt"], sh["duration"],
                        out_dir / "shot000.mp4", fps=24, seed=0,
                        reference_images=[BG_PLATE] + [PORTRAITS[n]
                                                       for n in sh["cast"]])
            model_id = cand
            print("模型锁定:", cand)
            break
        except Exception as exc:
            print(f"模型 {cand} 不可用({str(exc)[:120]})— 回退下一档")
    if model_id is None:
        raise SystemExit("全部候选模型不可用")

    clips = [out_dir / "shot000.mp4"]
    ledger = [{"shot": 0, "ok": True, "model": model_id}]
    for i, sh in enumerate(SHOTS[1:], start=1):
        outp = out_dir / f"shot{i:03d}.mp4"
        vg.generate_audio = True
        print(f"[shot {i+1}/{len(SHOTS)}] t2v {sh['duration']}s")
        try:
            vg.generate(sh["prompt"], sh["duration"], outp, fps=24, seed=0,
                        reference_images=[BG_PLATE] + [PORTRAITS[n]
                                                       for n in sh["cast"]])
            clips.append(outp)
            ledger.append({"shot": i, "ok": True})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            ledger.append({"shot": i, "ok": False, "error": str(exc)[:300]})
    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))
    if clips:
        concat_in = normalize_for_concat(clips, out_dir / "concat_norm") \
            if any_audio(clips) else clips
        final = VideoConcatTool().run(concat_in, out_dir / "movie.mp4")
        print("基线成片:", final, f"{_probe_seconds(Path(final)):.1f}s")


if __name__ == "__main__":
    main()
