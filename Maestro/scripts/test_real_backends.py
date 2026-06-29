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

所有产物（生成的 mp4 等）都存到带时间戳的目录：
    <repo>/outputs/realtest_<时间戳>/        （可用 --out-dir 或 $MAESTRO_OUTPUT_ROOT 改）
脚本结尾会打印这个目录，绝不再丢到 /tmp。

用法：
    export OPENAI_API_KEY=...
    export WAVESPEED_API_KEY=...
    python scripts/test_real_backends.py
    # 可选：--prompt "..."  --model gpt-4o  --out-dir /data/maestro_out
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 让脚本无需 `pip install -e .` 也能 import maestro（直接指向 src/）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maestro.models import build_llm, build_mllm, build_video_gen  # noqa: E402
from maestro.physics.annotate import annotate_physics  # noqa: E402
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
    """③ 多模态 VLM（OpenAI）—— 视觉评审生成的视频。

    把 ① 的文本和 ② 的视频串起来：VLM 真的解码 mp4 抽帧、看图判断。逻辑：
    _sample_frames(抽帧) → _chat(帧+prompt 发 chat/completions) → _extract_json(解析)。
    本脚本包一层 _chat，把发给模型的 prompt 和模型的【原始回复】都打印出来，
    方便你核对逻辑，然后再展示解析后的结构化结果。"""
    _section("③ VLM (OpenAI) — 视觉评审生成的视频")
    vlm = build_mllm({"name": "openai", "model": model})
    print(f"  backend = {type(vlm).__name__}  model = {model}  n_frames = {vlm.n_frames}")

    # —— 包一层 _chat：打印发出去的 prompt + 模型原始回复 ——
    _orig_chat = vlm._chat

    def _chat_verbose(frames, text):
        print(f"\n  ── 发给 VLM 的 prompt（附 {len(frames)} 帧图）──")
        print("    " + text.replace("\n", "\n    "))
        reply = _orig_chat(frames, text)
        print("  ── VLM 原始回复（raw reply）──")
        shown = reply if reply is not None else "<None：HTTP 请求失败，已降级为不出判决>"
        print("    " + str(shown).replace("\n", "\n    "))
        return reply

    vlm._chat = _chat_verbose  # type: ignore[method-assign]

    spec = ShotSpec(shot_idx=0, duration=5.0, prompt=shot_desc)
    spec.physics_annotation = annotate_physics(spec)   # assess_physics 需要预期模式
    clip = CandidateClip(shot_idx=0, video_path=video_path)

    # 先检查能不能抽到帧（解不出帧 → 所有判决都会是空，是 mp4/解码问题不是模型问题）
    if vlm._sample_frames(clip) is None:
        print("\n  ⚠️  抽不到帧：mp4 解不出来。装 `pip install opencv-python-headless "
              "imageio[ffmpeg]`，否则 VLM 没有像素可看、只能返回空。")
        return

    print("\n  >>> assess_semantic（语义：是否展现 prompt 关键元素）")
    items = vlm.assess_semantic(clip, spec)
    print(f"  解析结果（{len(items)} 条）:")
    for q, passed, fix in items:
        print(f"     [{'PASS' if passed else 'FAIL'}] {q}" + (f"  → fix: {fix}" if fix else ""))
    if not items:
        print("     （空：模型未返回可解析 JSON —— 看上面的 raw reply 判断原因）")

    print("\n  >>> assess_physics（物理：可见的定律违例，空=合理）")
    verdicts = vlm.assess_physics(clip, spec, fps=8)
    print(f"  解析结果（{len(verdicts)} 条）:")
    for v in verdicts:
        print(f"     [{v.mode.value}] 帧{v.frame_range} 严重度={v.severity:.2f}"
              f"  → {v.suggested_intervention}")
    if not verdicts:
        print("     （空：模型认为运动物理合理，或未返回可解析 JSON —— 见 raw reply）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red kite tumbling onto wet pavement")
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI 模型（同时用于 LLM 与 VLM）")
    ap.add_argument("--video-model", default="bytedance/seedance-v1-pro-t2v-480p",
                    help="WaveSpeed t2v 模型 id")
    ap.add_argument("--out-dir", default=None,
                    help="输出根目录（默认 $MAESTRO_OUTPUT_ROOT 或 <repo>/outputs）")
    args = ap.parse_args()

    # 前置检查：缺 key 立即说清楚，而不是中途 401。
    missing = [k for k in ("OPENAI_API_KEY", "WAVESPEED_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"❌ 缺少环境变量: {', '.join(missing)}\n"
              f"   export OPENAI_API_KEY=...  WAVESPEED_API_KEY=...  后重试。")
        return 2

    # 时间戳目录，所有产物都落在这里、明确可见（不再用 /tmp）。
    base = Path(args.out_dir or os.getenv("MAESTRO_OUTPUT_ROOT") or REPO_ROOT / "outputs")
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = base / f"realtest_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "shot.mp4"
    print(f"用户指令: {args.prompt}")
    print(f"输出目录: {run_dir.resolve()}")
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
        print(f"\n📂 本次所有产物都在: {run_dir.resolve()}")
        if out.exists():
            print(f"   生成的视频: {out.resolve()}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
