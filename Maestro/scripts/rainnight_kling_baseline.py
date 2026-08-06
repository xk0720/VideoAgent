#!/usr/bin/env python3
"""rainnight 可灵阉割版(2026-08-06 用户令):对照 run4
(outputs/movie_20260806_010939)——同 6 镜、同背景板、同肖像、同可灵,
但【零转场策略:不钉帧(不用上一镜尾帧)、无自动桥、无连续性语句、
prompt 降档粗写】,每镜独立 ref2v,直接硬拼。
隔离出的变量 = 钉/切路由 + 桥 + prompt 工艺。

用法: python scripts/rainnight_kling_baseline.py
输出: outputs/rainnight_kling_baseline_<ts>/
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

RUN4 = Path("/Users/kevin/Desktop/Kevin/repositories/VideoAgent/Maestro/outputs/movie_20260806_010939")
BG_PLATE = RUN4 / "anchors" / "bg_bg_1.png"
PORTRAITS = {"黑帮老大": RUN4 / "portraits" / "黑帮老大.png",
             "女子": RUN4 / "portraits" / "女子.png"}

# 全部独立 ref2v;refs 首位 = 背景板;prompt 随手粗写,声效写明
SHOTS = [
    dict(cast=["黑帮老大"], duration=8,
         prompt=("<<<image_1>>>的雨夜暗巷,黑色豪华轿车停着。车内"
                 "<<<image_2>>>点燃一支雪茄,红光在黑暗里明灭,雨水在"
                 "车窗上流淌,映着五彩霓虹。"
                 "音频:只有雨滴敲击车窗声和雪茄吸入声——无背景音乐、无人声。")),
    dict(cast=["女子"], duration=5,
         prompt=("<<<image_1>>>的雨夜。车内副驾驶座上<<<image_2>>>穿着"
                 "晚礼服,神情紧张恐惧,柔和霓虹光映在脸上。"
                 "音频:只有低沉心跳声——无背景音乐、无人声。")),
    dict(cast=["黑帮老大", "女子"], duration=5,
         prompt=("<<<image_1>>>的雨夜车内。<<<image_2>>>把一把金色手枪"
                 "缓缓抵在哭泣的<<<image_3>>>额头上,高对比度戏剧打光。"
                 "音频:只有冰冷金属摩擦声——无背景音乐、无人声。")),
    dict(cast=["女子"], duration=5,
         prompt=("<<<image_1>>>的雨夜。车侧窗突然碎裂,玻璃碎片在空中"
                 "飞舞,映着<<<image_2>>>绝望的目光,慢动作。"
                 "音频:只有玻璃碎裂声——无背景音乐、无人声。")),
    dict(cast=["黑帮老大", "女子"], duration=5,
         prompt=("<<<image_1>>>的雨夜。急推镜头,<<<image_2>>>扣动扳机,"
                 "一道金色枪口火焰照亮<<<image_3>>>的脸和黑暗巷道,火星"
                 "飞溅。音频:只有突发的巨大枪声——无背景音乐、无人声。")),
    dict(cast=["黑帮老大", "女子"], duration=5,
         prompt=("<<<image_1>>>的雨夜。烟雾弥漫,火星坠落,金色余光里"
                 "<<<image_2>>>与<<<image_3>>>的轮廓静止。"
                 "音频:只有金属回音渐弱——无背景音乐、无人声。")),
]


def main() -> None:
    assert BG_PLATE.exists(), f"背景板缺失: {BG_PLATE}"
    for n, p in PORTRAITS.items():
        assert p.exists(), f"肖像缺失: {n} → {p}"
    out_dir = (Path("outputs")
               / f"rainnight_kling_baseline_{time.strftime('%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    vg = build_video_gen({
        "name": "bailian_kling",
        "kling_model": "kling/kling-v3-video-generation",
        "kling_omni_model": "kling/kling-v3-omni-video-generation",
        "mode": "std", "aspect_ratio": "16:9",
        "generate_audio": False, "poll_interval": 15.0, "timeout": 900,
        "call_log": str(out_dir / "kling_calls.jsonl")})

    clips, ledger = [], []
    for i, sh in enumerate(SHOTS):
        outp = out_dir / f"shot{i:03d}.mp4"
        vg.generate_audio = True          # 全片音效驱动(无对白无音乐)
        refs = [BG_PLATE] + [PORTRAITS[n] for n in sh["cast"]]
        print(f"[shot {i+1}/{len(SHOTS)}] ref2v {sh['duration']}s "
              f"refs={len(refs)}")
        try:
            vg.generate(sh["prompt"], sh["duration"], outp, fps=24, seed=0,
                        reference_images=refs)
            clips.append(outp)
            ledger.append({"shot": i, "ok": True, "prompt": sh["prompt"]})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            ledger.append({"shot": i, "ok": False, "error": str(exc)[:300]})
    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))
    if clips:
        concat_in = normalize_for_concat(clips, out_dir / "concat_norm") \
            if any_audio(clips) else clips
        final = VideoConcatTool().run(concat_in, out_dir / "movie.mp4")
        print("阉割版成片:", final, f"{_probe_seconds(Path(final)):.1f}s")


if __name__ == "__main__":
    main()
