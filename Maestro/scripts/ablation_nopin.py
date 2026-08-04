#!/usr/bin/env python3
"""消融实验(用户令):用 brain 的【初版草稿 prompt】(无润色)、
【去掉首帧依赖】(不钉帧,只带 背景板+肖像 参考)把全部分镜重跑一遍,
结果按顺序直接拼接 —— 与主跑(钉帧+润色+修复)对照接缝自然度。

用法: python scripts/ablation_nopin.py <run_dir> [--config configs/bailian.yaml]
输出: outputs/ablation_<runname>_<ts>/(新目录,不碰主跑产物)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from maestro.memory.storyboard import StoryboardMemory
from maestro.models.video_gen import build_video_gen
from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
from maestro.pipeline.window_loop import (_name_slot_map, _probe_seconds,
                                          _slot_manifest, _with_dialogue)
from maestro.tools.video_concat import VideoConcatTool

# 首帧依赖句剥除:提及钉帧/上镜续接的整句丢弃(草稿是为钉帧写的,
# 消融跑没有钉帧 —— 这些句子会让模型凭空"续"不存在的东西)。
_PIN_RE = re.compile(
    r"[^.!?]*\b(pinned|first frame|previous shot|previous frame|"
    r"taking over|continues? (?:from|the established)|handoff)\b[^.!?]*[.!?]",
    re.IGNORECASE)


def strip_pin_dependency(prompt: str) -> str:
    out = _PIN_RE.sub(" ", str(prompt or ""))
    return re.sub(r"\s{2,}", " ", out).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default="configs/bailian.yaml")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    sb = StoryboardMemory.load(run_dir / "storyboard.json")

    out_dir = (Path("outputs") /
               f"ablation_{run_dir.name}_{time.strftime('%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=False)
    print("消融输出目录:", out_dir)

    cfg = yaml.safe_load(open(args.config))
    vg_cfg = dict((cfg.get("models") or cfg).get("video_gen") or {})
    vg_cfg.setdefault("call_log", str(out_dir / "wavespeed_calls.jsonl"))
    vg = build_video_gen(vg_cfg)

    ledger = []
    clips = []
    for e in sb.entries:
        draft = (e.draft_prompt or "").strip() \
            or str((e.condition or {}).get("final_prompt") or "").strip() \
            or e.description
        prompt = strip_pin_dependency(draft)

        # 参考 = 背景板 + 本镜出场者肖像(与 ref2v 槽位清单同序:先图后像)
        bg = (sb.backgrounds or {}).get(
            getattr(e, "bg_id", "") or f"scene_{e.scene_idx}") or {}
        refs = [bg["path"]] if bg.get("path") else []
        cast_in = {n: p for n, p in (sb.portraits or {}).items()
                   if f"<{n}>" in e.description or n in e.description}
        refs += [cast_in[n] for n in sorted(cast_in)]

        # 台词兜底(记号化,与主链同一函数)
        rows = [{"slot": f"<<<image_{i+1}>>>", "referenceable": True,
                 **({"name": n} if i >= (1 if bg.get("path") else 0)
                    else {})}
                for i, n in enumerate(
                    ([None] if bg.get("path") else []) + sorted(cast_in))]
        if getattr(e, "dialogue", ""):
            vg.generate_audio = True
            prompt = _with_dialogue(prompt, e, sb.cast,
                                    name_to_slot=_name_slot_map(rows))
        else:
            vg.generate_audio = False

        dur = 5
        src = run_dir / f"shot{e.shot_idx:03d}"
        for f in sorted(src.glob("*_w_s0*.mp4")):
            got = _probe_seconds(f)
            if got:
                dur = max(4, min(10, round(got)))
                break
        outp = out_dir / f"shot{e.shot_idx:03d}.mp4"
        print(f"[{e.label}] ref2v x{len(refs)} refs, {dur}s")
        print("  prompt:", prompt[:140], "…")
        try:
            vg.generate(prompt, dur, outp, fps=24,
                        reference_images=refs, seed=0)
            clips.append(outp)
            ledger.append({"shot": e.shot_idx, "prompt": prompt,
                           "refs": [str(r) for r in refs], "ok": True})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            ledger.append({"shot": e.shot_idx, "prompt": prompt,
                           "error": str(exc)[:300], "ok": False})

    (out_dir / "ablation_ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))
    if clips:
        concat_in = normalize_for_concat(clips, out_dir / "concat_norm") \
            if any_audio(clips) else clips
        final = VideoConcatTool().run(concat_in, out_dir / "movie.mp4")
        print("消融成片:", final, f"{_probe_seconds(Path(final)):.1f}s")


if __name__ == "__main__":
    main()
