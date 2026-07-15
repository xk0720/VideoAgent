#!/usr/bin/env python
"""Playground:测 gemini-3.5-flash 的【时间定位】能力(原生视频输入)。

上传一段视频(标准 generateContent POST:x-goog-api-key + 文本标签 +
inline_data video/mp4 —— 与生产 GeminiVLM 完全同一条传输层,>18MB 自动
360p 转码,永不抽帧),默认问题:

    画面中【没有红色苹果】的时间范围是多少?

要求模型报秒级区间 + 每段"画面里是什么"的证据句,顺带自报视频总时长
(校验它对时间轴的感知)。--question 可换任意时间定位问题。

用法:
    export GEMINI_API_KEY=...              # 或写在仓库根 .env
    python src/maestro/playground/gemini_3_5_flash.py --video /path/to/v.mp4
    python src/maestro/playground/gemini_3_5_flash.py --video v.mp4 \
        --question "During which time ranges is the camera moving?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from maestro.models.mllm_backends import GeminiVLM, _extract_json  # noqa: E402

# 默认问题:红苹果的【缺席】区间 —— 比"在哪出现"更狠:模型必须对整条
# 时间轴保持注意,漏看任何一段都会报错区间。
DEFAULT_QUESTION = (
    "During which time ranges is there NO red apple visible anywhere in the "
    "frame? Watch the WHOLE video before answering. Report every such range "
    "in seconds from the start of the video, with ~0.5 s precision. If a red "
    "apple is visible for the entire video, return an empty ranges list."
)

_INSTRUCTION = """You are answering a TEMPORAL LOCALIZATION question about \
the video above.

QUESTION: {question}

Rules:
- Times are SECONDS from the start of the video (floats, ~0.5 s precision).
- Ranges must not overlap and must be in chronological order.
- For each range give one short "evidence" sentence describing what IS on \
screen during that range (so the answer can be spot-checked).
- Also report the total video duration as you perceive it.
- Judge only what is visible; if unsure about a boundary, say so in evidence.

Reply with ONLY a JSON object, no markdown fence:
{{"video_duration_s": <float>,
 "ranges": [{{"start_s": <float>, "end_s": <float>, "evidence": "<one \
sentence>"}}],
 "note": "<one sentence overall, e.g. 'apple visible throughout' if empty>"}}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="要提问的视频")
    ap.add_argument("--question", default=DEFAULT_QUESTION,
                    help="时间定位问题(英文;默认:无红苹果的时间范围)")
    args = ap.parse_args()

    src = Path(args.video)
    if not src.is_file():
        print(f"❌ 视频不存在: {src}")
        return 2

    vlm = GeminiVLM("gemini", {})          # $GEMINI_MODEL / $GEMINI_API_KEY
    print(f"模型: {vlm.model} @ {vlm.base_url}\n视频: {src} "
          f"({src.stat().st_size / 1e6:.1f} MB)\n问题: {args.question}\n"
          "上传并提问中…")

    parts = vlm._video_part("THE VIDEO", src)
    if not parts:
        print("❌ 视频无法内联(超限且无 ffmpeg,或不是真视频文件)")
        return 2
    reply = vlm._generate(
        parts + [{"text": _INSTRUCTION.format(question=args.question)}])
    if reply is None:
        print("❌ 调用失败(重试后仍无回复)—— 看上方 WARN 日志")
        return 1

    print(f"\n── 原始回复 ──\n{reply}\n")
    data = _extract_json(reply)
    if not isinstance(data, dict):
        print("⚠ 回复不是可解析 JSON —— 以上原文自行判读")
        return 1
    print("── 解析结果 ──")
    print(f"模型感知的视频总长: {data.get('video_duration_s', '?')} s")
    ranges = data.get("ranges") or []
    if not ranges:
        print(f"区间: (空) — {data.get('note', '')}")
    for r in ranges:
        print(f"  {float(r.get('start_s', 0)):6.2f}s → "
              f"{float(r.get('end_s', 0)):6.2f}s   {r.get('evidence', '')}")
    if data.get("note"):
        print(f"备注: {data['note']}")
    print("\n验收:拿区间去原视频逐段核对(播放器拖到 start_s/end_s 看画面"
          "里有没有红苹果),同时核对总时长是否与 ffprobe 一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
