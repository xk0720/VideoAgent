#!/usr/bin/env python3
"""重生成两座转场桥(2026-08-05 用户令:旋转/位移太快太夸张)。

吃 scripts/outputs/seedance_manual_001918 里现成的桥端点 PNG,用
【前景遮挡】新 prompt 重生成 bridge_12 / bridge_23(480p 对齐原片),
然后用原 6 镜 + 新桥重拼完整成片。只花 2 次 flf2v。

用法: python scripts/regen_bridges.py
输出: scripts/outputs/bridges_regen_<ts>/
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.models.video_gen import build_video_gen
from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
from maestro.pipeline.window_loop import _probe_seconds
from maestro.tools.video_concat import VideoConcatTool

SRC = Path(__file__).resolve().parent / "outputs" / "seedance_manual_001918"

BRIDGE_12 = ("转场运镜:镜头从含泪少女的面部特写缓缓转开,平稳转向"
             "她对面,银发女子挽着军装男子手臂的中近景随转动进入"
             "画面,构图对齐后停稳;转动缓慢柔和,无剪辑感,人物"
             "不在原地变换面貌或服装。")
BRIDGE_23 = ("转场运镜:镜头从挽臂而立的二人中近景缓缓平移离开,"
             "移向人群边缘,两名并肩军官的双人中近景随移动进入"
             "画面后停稳;移动缓慢平稳,无剪辑感,人物不在原地"
             "变换面貌或服装。")


def main() -> None:
    out_dir = (Path(__file__).resolve().parent / "outputs"
               / f"bridges_regen_{time.strftime('%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    vg = build_video_gen({"name": "wavespeed", "resolution": "480p",
                          "call_log": str(out_dir / "wavespeed_calls.jsonl")})
    vg.generate_audio = False
    bridges = {}
    for tag, prompt in [("12", BRIDGE_12), ("23", BRIDGE_23)]:
        fa, fb = SRC / f"bridge_{tag}_prev.png", SRC / f"bridge_{tag}_next.png"
        assert fa.exists() and fb.exists(), f"端点帧缺失: {fa} / {fb}"
        print(f"[桥 {tag}] flf2v 4s…")
        bp = vg.frame_to_frame(
            prompt=prompt, first_frame=fa, last_frame=fb,
            out_path=out_dir / f"bridge_{tag}.mp4", duration=4, seed=777)
        bridges[tag] = Path(bp)
        print(f"[桥 {tag}] OK → {bp}")

    # 重拼:000,001,[桥12],002,[桥23],003,004,005
    clips = [SRC / "shot000.mp4", SRC / "shot001.mp4", bridges["12"],
             SRC / "shot002.mp4", bridges["23"], SRC / "shot003.mp4",
             SRC / "shot004.mp4", SRC / "shot005.mp4"]
    clips = [c for c in clips if Path(c).exists()]
    concat_in = normalize_for_concat(clips, out_dir / "concat_norm") \
        if any_audio(clips) else clips
    final = VideoConcatTool().run(concat_in, out_dir / "movie.mp4")
    print("重拼成片:", final, f"{_probe_seconds(Path(final)):.1f}s")


if __name__ == "__main__":
    main()
