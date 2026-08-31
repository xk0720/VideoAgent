#!/usr/bin/env python3
"""[预处理助手, 不属于卡片] 运镜视频 → 三档景别参考图。

用法 A(已有运镜视频):  python3 tools/make_ref_frames.py dolly.mp4 输出目录
用法 B(顺带生成视频):  python3 tools/make_ref_frames.py --gen hook.jpg 输出目录

产物 ref_full/ref_medium/ref_close.jpg → 填进商品 brief 的 ref_frames 字段。
三档取帧窗可按 --windows "0.1-0.9,3.0-3.7,4.1-4.95" 调整; 每窗取最清晰帧。
运镜 prompt 模板见 outputs/yike_refs/dolly_prompt.txt(人静止, 全身匀速推至胸像)。
"""
import argparse
import sys
from pathlib import Path

VS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VS))
from studio.local_tools import grab_frame                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="运镜视频; 配 --gen 时为 hook 图片")
    ap.add_argument("out_dir")
    ap.add_argument("--gen", action="store_true", help="先用 kling 生成运镜视频(计费)")
    ap.add_argument("--windows", default="0.1-0.9,3.0-3.7,4.1-4.95",
                    help="full,medium,close 三窗(秒)")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    video = args.source
    if args.gen:
        import os
        from dotenv import load_dotenv
        load_dotenv(VS / ".env")
        from studio.backends.bailian_kling import BailianKlingClient
        prompt = (VS / "outputs/yike_refs/dolly_prompt.txt").read_text(encoding="utf-8")
        video = str(out / "dolly.mp4")
        ok, tid, err = BailianKlingClient(
            api_key=os.environ["DASHSCOPE_API_KEY"], mode="pro").generate(
            prompt=prompt, duration=5, save_to=video,
            refer=[args.source], audio=False, aspect_ratio="9:16")
        if not ok:
            print(f"生成失败: {err}"); return 1

    for tier, win in zip(("full", "medium", "close"),
                         args.windows.split(",")):
        t0, t1 = (float(x) for x in win.split("-"))
        grab_frame(video, out / f"ref_{tier}.jpg", window=[t0, t1])
        print(f"ref_{tier}.jpg ← {win}s")
    print(f"完成 → {out}  (填入商品 brief 的 ref_frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
