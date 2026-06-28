#!/usr/bin/env python
"""端到端真实后端连通测试 —— LLM / VLM 用 OpenAI，视频用 WaveSpeed。

这不是 pytest（会真的打 API、花钱、需要网络）。它验证三个真实模型调用都能跑通，
并且把它们串成 Maestro 的实际链路：

    ① OpenAI(纯语言)  规划一句镜头描述
    ② WaveSpeed       根据描述真生成一段 mp4
    ③ OpenAI(多模态)  视觉评审那段生成的 mp4

每一步独立打印结果，任一步失败都能单独定位。三个调用全部通过 Maestro 的统一
抽象（build_llm / build_mllm / build_video_gen），不是直接拼 HTTP —— 这正是
我们相对 UniVA 那些扁平函数的“统一 tool call”：上层只调 .complete() /
.assess_semantic() / .generate()，不关心背后是谁。

用法：
    export OPENAI_API_KEY=...
    export WAVESPEED_API_KEY=...
    python scripts/test_real_backends.py
    # 可选：--prompt "..."  --model gpt-4o  --keep（保留生成的 mp4）
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# 让脚本无需 `pip install -e .` 也能 import maestro（直接指向 src/）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maestro.models import build_llm, build_mllm, build_video_gen  # noqa: E402
from maestro.types import CandidateClip, ShotSpec  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def stage_llm(model: str, user_prompt: str) -> str:
    """① 纯语言 LLM（OpenAI）—— build_llm({name:openai}).complete(prompt)."""
    _section("① LLM (OpenAI) — 规划镜头描述")
    llm = build_llm({"name": "openai", "model": model})
    print(f"  backend = {type(llm).__name__}  model = {model}")
    ask = (
        "You are a video director. In ONE vivid English sentence, describe a "
        f"single 5-second shot for this idea: {user_prompt}. Output only the sentence."
    )
    shot_desc = llm.complete(ask).strip()
    print(f"  ✅ LLM 返回: {shot_desc}")
    return shot_desc


def stage_video(shot_desc: str, out_path: Path, model_id: str) -> Path:
    """② 视频生成（WaveSpeed）—— build_video_gen({name:wavespeed}).generate(...)."""
    _section("② Video (WaveSpeed) — 真生成 mp4")
    vg = build_video_gen({"name": "wavespeed", "model_id": model_id})
    print(f"  backend = {type(vg).__name__}  model_id = {model_id}")
    print("  提交任务并轮询（首次可能 ~30-90s）…")
    path = vg.generate(prompt=shot_desc, duration=5.0, out_path=out_path, fps=8)
    size = Path(path).stat().st_size
    print(f"  ✅ 视频已下载: {path}  ({size/1024:.0f} KB)")
    if size < 1024:
        raise RuntimeError("生成文件 < 1KB，可能不是真实视频")
    return Path(path)


def stage_vlm(model: str, video_path: Path, shot_desc: str) -> None:
    """③ 多模态 VLM（OpenAI）—— build_mllm({name:openai}).assess_semantic(clip, spec).

    这一步把 ① 的文本和 ② 的视频串起来：VLM 真的解码 mp4 抽帧、看图判断是否
    符合描述。返回 [] 说明 clip 解不出帧（需要装 opencv/imageio）。"""
    _section("③ VLM (OpenAI) — 视觉评审生成的视频")
    vlm = build_mllm({"name": "openai", "model": model})
    print(f"  backend = {type(vlm).__name__}  model = {model}")
    spec = ShotSpec(shot_idx=0, duration=5.0, prompt=shot_desc)
    clip = CandidateClip(shot_idx=0, video_path=video_path)
    items = vlm.assess_semantic(clip, spec)   # [(question, passed, fix), ...]
    if not items:
        print("  ⚠️  VLM 返回空 —— 通常是 mp4 解不出帧（pip install "
              "opencv-python-headless imageio[ffmpeg]）或模型未返回可解析 JSON。")
        return
    print(f"  ✅ VLM 评审 {len(items)} 条:")
    for q, passed, fix in items:
        mark = "PASS" if passed else "FAIL"
        print(f"     [{mark}] {q}" + (f"  → fix: {fix}" if fix else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red kite tumbling onto wet pavement")
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI 模型（同时用于 LLM 与 VLM）")
    ap.add_argument("--video-model", default="bytedance/seedance-v1-pro-t2v-480p",
                    help="WaveSpeed t2v 模型 id")
    ap.add_argument("--keep", action="store_true", help="保留生成的 mp4")
    args = ap.parse_args()

    # 前置检查：缺 key 立即说清楚，而不是中途 401。
    missing = [k for k in ("OPENAI_API_KEY", "WAVESPEED_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"❌ 缺少环境变量: {', '.join(missing)}\n"
              f"   export OPENAI_API_KEY=...  WAVESPEED_API_KEY=...  后重试。")
        return 2

    print(f"用户指令: {args.prompt}")
    tmpdir = Path(tempfile.mkdtemp(prefix="maestro_realtest_"))
    out = tmpdir / "shot.mp4"
    results = {"llm": False, "video": False, "vlm": False}
    try:
        shot_desc = stage_llm(args.model, args.prompt)
        results["llm"] = True

        video_path = stage_video(shot_desc, out, args.video_model)
        results["video"] = True

        stage_vlm(args.model, video_path, shot_desc)
        results["vlm"] = True
    except Exception as exc:
        _section("❌ 失败")
        import traceback
        traceback.print_exc()
        print(f"\n失败于某一阶段: {exc}")
    finally:
        _section("结果")
        for k in ("llm", "video", "vlm"):
            print(f"  {'✅' if results[k] else '❌'} {k}")
        if args.keep and out.exists():
            print(f"\n生成的视频保留在: {out}")
        elif out.exists():
            out.unlink()
            try:
                tmpdir.rmdir()
            except OSError:
                pass
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
