#!/usr/bin/env python3
"""基线对照版(2026-08-05 用户令):与 seedance_manual.py 同一套手写
分镜,但【零转场、零钉帧、prompt 无任何首帧/续接约束】,每镜独立
t2v(背景板+肖像引用保留,隔离变量=接缝机械与模型档位),直接拼接。
模型降档:seedance-v1-lite 480p(更差更便宜;id 404 时逐级回退)。

用法: python scripts/seedance_manual_baseline.py
输出: outputs/seedance_baseline_<ts>/
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.models.video_gen import build_video_gen
from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
from maestro.pipeline.script_input import parse_script_json
from maestro.pipeline.window_loop import _probe_seconds
from maestro.tools.video_concat import VideoConcatTool

AUDIO_TAIL = "音频:只有角色说这句台词的人声——无背景音乐、无音效。"

BG_PLATE_PROMPT = (
    "Interior of a grand 19th-century European royal palace ballroom at "
    "night: gilded walls with red damask panels, several blazing crystal "
    "chandeliers, polished cream marble floor with clear reflections, a "
    "raised ceremonial stage with steps at the far end; formally dressed "
    "anonymous period guests and officers lining both side walls in the "
    "middle distance with small unobtrusive faces, the wide central floor "
    "open and empty, deep focus, eye-level wide establishing view, no "
    "principal characters, no modern objects.")

# 便宜差档模型候选(前者 404 → 依次回退;最后回退 2.0@480p 保run通)
CHEAP_MODELS = ["bytedance/seedance-v1-lite-t2v-480p",
                "bytedance/seedance-v1-pro-t2v-480p",
                "bytedance/seedance-2.0/text-to-video"]

# 全部 t2v,每镜独立;refs 首位 @BG=背景板
SHOTS = [
    dict(refs=["@BG", "安莉希娅", "芬莱克殿下", "安娜"], duration=8,
         audio=False,
         prompt=("大远景:@Image1的舞厅里,@Image2挽着@Image3站在舞台前,"
                 "@Image4一个人站在对面。")),
    dict(refs=["@BG", "芬莱克殿下", "安莉希娅", "安娜"], duration=8,
         audio=True,
         prompt=("@Image1的舞厅。先拍地板上@Image2和@Image3的倒影,"
                 "@Image4走过来,镜头抬起来拍过肩,@Image2严厉地说:"
                 "“你这种女人不配做王后,安莉希娅才是真正合适的人选!”"
                 f"{AUDIO_TAIL}")),
    dict(refs=["@BG", "安娜"], duration=6, audio=True,
         prompt=("@Image1的舞厅。@Image2脸部特写,眼里含泪,嘴唇发抖,"
                 f"说:“……为什么?”{AUDIO_TAIL}")),
    dict(refs=["@BG", "安莉希娅", "芬莱克殿下"], duration=6, audio=True,
         prompt=("@Image1的舞厅。中近景,@Image2挽着@Image3的手臂看着他,"
                 "假装不忍心,说:“芬莱克殿下……这对于安娜小姐来说,"
                 f"会不会太过了?”{AUDIO_TAIL}")),
    dict(refs=["@BG", "男性军官"], duration=5, audio=True,
         prompt=("@Image1的舞厅。人群边上两个军官,左边的@Image2凑近另一个"
                 f"低声说:“政治联姻的工具罢了。”{AUDIO_TAIL}")),
    dict(refs=["@BG", "男性军官"], duration=5, audio=True,
         prompt=("@Image1的舞厅。还是那两个军官,右边的(长相同@Image2)"
                 f"回头说:“真可怜啊,公爵千金……”{AUDIO_TAIL}")),
    dict(refs=["@BG", "持折扇女子"], duration=6, audio=True,
         prompt=("@Image1的舞厅。@Image2用黑折扇挡着半张脸,一边收扇子"
                 f"一边冷笑说:“真可怜啊,公爵千金……”{AUDIO_TAIL}")),
]


def main() -> None:
    parsed = parse_script_json(
        Path("/Users/kevin/Desktop/script-wedding/script.json"))
    roles = parsed["roles"]
    out_dir = Path("outputs") / f"seedance_baseline_{time.strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    model_id = None
    vg = None
    bg_plate = None
    for cand in CHEAP_MODELS:
        vg = build_video_gen({"name": "wavespeed", "model_id": cand,
                              "resolution": "480p",
                              "call_log": str(out_dir /
                                              "wavespeed_calls.jsonl")})
        if bg_plate is None:
            bg_plate = vg.text_to_image(BG_PLATE_PROMPT,
                                        out_dir / "bg_plate.png")
            print("背景板:", bg_plate)
        # 探针:第一镜提交成功即锁定该模型;404/400 → 换下一档
        try:
            sh = SHOTS[0]
            vg.generate_audio = bool(sh["audio"])
            vg.generate(sh["prompt"], sh["duration"],
                        out_dir / "shot000.mp4", fps=24, seed=0,
                        reference_images=[bg_plate if n == "@BG"
                                          else roles[n]
                                          for n in sh["refs"]])
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
        vg.generate_audio = bool(sh["audio"])
        print(f"[shot {i+1}/{len(SHOTS)}] t2v {sh['duration']}s")
        try:
            vg.generate(sh["prompt"], sh["duration"], outp, fps=24, seed=0,
                        reference_images=[bg_plate if n == "@BG"
                                          else roles[n]
                                          for n in sh["refs"]])
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
