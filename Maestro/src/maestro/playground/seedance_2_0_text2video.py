#!/usr/bin/env python
"""Playground:测 bytedance/seedance-2.0/text-to-video 的【参考视频续接】能力。

任务:给一段 reference video,写 prompt 让模型把它当作【本镜的开头】接着
往下拍 —— 本质是在 t2v + reference_videos 通道上测 extend 能力;另外附加
一个小要求,验证"续接的同时还能听指挥"。

Prompt 怎么写(设计思路,写死成默认模板,--extra/--prompt 可覆盖):
1. 把续接契约说成机械事实:本镜从 @Video1 的【最后一刻】开始,是同一个
   不间断镜头 —— 不是"风格像它",是"接着它拍";
2. 显式搬运全部场景状态:同一批主体、同一运镜、同一光线、同一环境,
   主体的运动从 @Video1 结尾处自然延续;
3. 把已知失败模式写成禁令:不许重播/复现 @Video1 的内容(常见翻车:模型
   把参考视频重演一遍而不是续拍)、不许切镜、不许换场景;
4. 附加小要求(默认:镜头缓推向主体)—— 可肉眼验收,且对任何素材都成立。

用法:
    export WAVESPEED_API_KEY=...           # 或写在仓库根 .env
    python src/maestro/playground/seedance_2_0_text2video.py \
        --video /path/to/reference.mp4 \
        [--extra "a seagull lands on the railing"] \
        [--duration 8] [--resolution 480p]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from maestro.models.video_gen_backends import WaveSpeedClient  # noqa: E402
from maestro.pipeline.window_loop import _cut_tail             # noqa: E402


def _probe_seconds(video: Path) -> float:
    """ffprobe 时长(秒);探测不了返回 0.0(按合规长度处理,交给 API 把关)。"""
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0

MODEL_ID = "bytedance/seedance-2.0/text-to-video"

# 续接契约 + 禁令 + 附加小要求。@Video1 = 官方 reference_videos 提及语法。
PROMPT_TEMPLATE = (
    "The shot begins at the exact final moment of @Video1 and continues it "
    "seamlessly as the same uninterrupted take: keep the same scene, the "
    "same subjects, the same lighting and the same camera motion, and carry "
    "every subject's movement forward naturally from where @Video1 ends. "
    "Do not replay or re-show @Video1's content, do not cut, do not change "
    "location. In addition, {extra}."
)
DEFAULT_EXTRA = ("partway through the shot, the camera slowly pushes in "
                 "toward the main subject")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="参考视频(将作为本镜开头)")
    ap.add_argument("--extra", default=DEFAULT_EXTRA,
                    help="附加小要求(英文;默认:镜头缓推向主体)")
    ap.add_argument("--prompt", default=None,
                    help="完整覆盖默认模板(不建议;模板即本测试的测点)")
    ap.add_argument("--duration", type=int, default=None,
                    help="秒数 int 4-15(缺省不传 = API 默认)")
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    src = Path(args.video)
    if not src.is_file():
        print(f"❌ 参考视频不存在: {src}")
        return 2

    out_dir = Path(args.out_dir or REPO_ROOT / "outputs" / "playground"
                   / f"seedance_ref_extend_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 官方硬限:reference_videos 单条 ≤15s —— 超长取【尾段】15s(续接看的
    # 是结尾,尾段才是接点);裁剪不可用(无 ffmpeg)则原样上传并提醒。
    ref = src
    dur_s = _probe_seconds(src)
    if dur_s > 15.0:
        trimmed = _cut_tail(src, 15.0, out_dir / "ref_tail15.mp4")
        if trimmed is not None:
            ref = Path(trimmed)
            print(f"参考视频 {dur_s:.1f}s 超 15s → 取尾段 15s: {ref}")
        else:
            print("⚠ 无 ffmpeg,参考视频原样上传(>15s 会被 API 拒绝)")

    prompt = args.prompt or PROMPT_TEMPLATE.format(extra=args.extra)
    client = WaveSpeedClient(config={
        "model_id": MODEL_ID,
        "resolution": args.resolution,
        "call_log": str(out_dir / "calls.jsonl"),
    })
    out = out_dir / "extended.mp4"
    print(f"模型: {MODEL_ID}\nprompt:\n  {prompt}\n"
          f"duration: {args.duration if args.duration is not None else '(API 默认)'}"
          f"  resolution: {args.resolution}\n生成中…")
    t0 = time.time()
    path = client.generate(prompt=prompt, duration=args.duration,
                           out_path=out, reference_video=ref)
    print(f"\n✅ 完成({time.time() - t0:.0f}s)\n  产物: {path}\n"
          f"  调用日志: {out_dir / 'calls.jsonl'}\n"
          "验收:开头是否无缝接上参考视频结尾(不重播、不切镜)+ 附加要求"
          "是否出现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
