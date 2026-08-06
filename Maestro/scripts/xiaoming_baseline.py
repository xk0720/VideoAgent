#!/usr/bin/env python3
"""xiaoming 基线对照版(2026-08-06 用户令,体例同 seedance_manual_baseline):
与 xiaoming_manual.py 同一套分镜,但【零钉帧、prompt 无任何首帧/续接
约束、prompt 随手粗写】,每镜独立 t2v(背景板+肖像引用保留,隔离变量
=接缝机械+prompt 工艺+模型档位),直接拼接。
模型降档:seedance-v1-lite 480p(id 404 → 逐级回退保 run 通)。

用法: python scripts/xiaoming_baseline.py
输出: outputs/xiaoming_baseline_<ts>/
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
    "Empty seaside at dusk: a wide wet sand beach with the low golden "
    "setting sun near the horizon, powerful waves rolling in with white "
    "foam and spray, a few jagged black reef rocks at one side including "
    "one low flat-topped rock, warm golden light across the water, the "
    "central foreground open and empty, deep focus, eye-level wide view, "
    "no people, no birds, no modern objects.")

# 便宜差档候选(前者 404 → 依次回退;最后回退 2.0@480p 保 run 通)
CHEAP_MODELS = ["bytedance/seedance-v1-lite-t2v-480p",
                "bytedance/seedance-v1-pro-t2v-480p",
                "bytedance/seedance-2.0/text-to-video"]

# 全部 t2v,每镜独立,prompt 随手粗写;refs 首位 @BG=背景板
SHOTS = [
    dict(refs=["@BG", "小明"], duration=10, audio=True,
         prompt=("@Image1的黄昏海边。@Image2攥着拳头盯着海面,头发被吹得"
                 "乱糟糟,脚边放着湿公文包,手里拿着半块三明治,说:"
                 "“大海真大啊，大到能吞掉我所有的失败。”"
                 f"{AUDIO_TAIL}")),
    dict(refs=["@BG", "阿浪", "小明"], duration=12, audio=True,
         prompt=("@Image1的海边。@Image2俯冲下来抢走@Image3手里的三明治,"
                 "落在礁石上歪着头说:“喂，人类。你在那装什么深沉？眼泪"
                 f"都快比海水咸了，也没见你跳下去洗个澡。”{AUDIO_TAIL}")),
    dict(refs=["@BG", "小明"], duration=10, audio=True,
         prompt=("@Image1的海边。@Image2愣住苦笑,说:“你一只鸟懂什么？"
                 f"我在想，是不是该彻底消失算了。”{AUDIO_TAIL}")),
    dict(refs=["@BG", "阿浪"], duration=15, audio=True,
         prompt=("@Image1的海边礁石上。@Image2扑腾翅膀抖掉三明治屑,"
                 "说:“消失？大海只收垃圾，不收懦夫。你看这浪，拍碎了"
                 "一万次，下一秒照样卷土重来。真正的强大不是从不跌倒，"
                 f"而是满身沙砾还敢对着太阳大笑。”{AUDIO_TAIL}")),
    dict(refs=["@BG", "小明"], duration=8, audio=True,
         prompt=("@Image1的海边,夕阳金光。@Image2低头看看面包屑又抬头,"
                 f"问:“你……是在安慰我吗？”{AUDIO_TAIL}")),
    dict(refs=["@BG", "阿浪"], duration=12, audio=True,
         prompt=("@Image1的海边。@Image2翻个白眼展翅高飞,说:“少自作"
                 "多情！我只是怕你饿死了，没人给我投喂薯片！记住，明天"
                 "太阳升起时，别让我看见你还像个落汤鸡！”然后飞远。"
                 f"{AUDIO_TAIL}")),
    dict(refs=["@BG", "小明"], duration=10, audio=True,
         prompt=("@Image1的海边。@Image2弯腰捡起湿公文包甩甩水,对着大海"
                 "大喊:“好！那就从头再来！”海浪漫过脚踝,夕阳下沉。"
                 f"{AUDIO_TAIL}")),
]


def main() -> None:
    parsed = parse_script_json(
        Path("/Users/kevin/Desktop/script-xiaoming/script.json"))
    roles = parsed["roles"]
    out_dir = Path("outputs") / f"xiaoming_baseline_{time.strftime('%H%M%S')}"
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
        # 探针:第一镜提交成功即锁定该模型;失败 → 换下一档
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
