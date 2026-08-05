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
         prompt=("大远景,固定镜头:场景与@Image1所示的宫廷大舞厅完全一致"
                 "——同一空间、陈设与烛光。@Image2挽着@Image3的手臂静立在"
                 "舞台台阶前的中央,@Image4独自站在他们对面不远处,孤立"
                 "无援。宾客仅在远处轻微走动交谈,无人进入中央前景。整段"
                 "镜头保持固定,三人静立,画面庄重安静。")),
    dict(refs=["@BG", "芬莱克殿下", "安莉希娅", "安娜"], duration=8,
         audio=True,
         prompt=("特写起手,低机位:场景为@Image1所示大舞厅。抛光大理石"
                 "地面映出@Image2与@Image3并立的倒影,烛光闪烁;@Image4的"
                 "脚步走入倒影视野,步伐平稳。镜头从她脚部沿背影平稳上摇,"
                 "越过@Image4的肩膀形成过肩中近景;@Image2的脸清晰入画,"
                 "神情冷酷严厉,开口说:“你这种女人不配做王后,安莉希娅"
                 "才是真正合适的人选!”@Image3挽着他的手臂未动。说完后"
                 f"三人静止,镜头停稳。{AUDIO_TAIL}")),
    dict(refs=["@BG", "安娜"], duration=6, audio=True,
         prompt=("面部大特写,固定镜头:场景为@Image1所示大舞厅。@Image2"
                 "正对镜头,身后宾客完全虚化成暖色光斑。她蓝眸噙满泪水,"
                 "下唇微微颤抖,神情不可置信,凝望画外;泪水在眼眶里打转"
                 "而不落下。她艰难地轻声挤出一句:“……为什么?”说完后"
                 f"嘴唇停止颤动,面容僵住,镜头静止。{AUDIO_TAIL}")),
    dict(refs=["@BG", "安莉希娅", "芬莱克殿下"], duration=6, audio=True,
         prompt=("中近景,固定镜头:场景为@Image1所示大舞厅。@Image2站在"
                 "@Image3身旁,双手挽着他的手臂,仰头望向他,面露不忍,"
                 "嘴角却藏着一丝得意。她柔声说:“芬莱克殿下……这对于"
                 "安娜小姐来说,会不会太过了?”说完后她保持仰望,笑意含"
                 f"而不露,@Image3神情僵硬不动,镜头静止。{AUDIO_TAIL}")),
    dict(refs=["@BG", "男性军官"], duration=5, audio=True,
         prompt=("双人中近景,固定镜头:场景为@Image1所示大舞厅。两名同穿"
                 "深蓝镶金军装的年轻军官并肩站在宾客人群边缘,身后宾客"
                 "虚化。左侧的@Image2侧身凑近右侧同伴,抬起白手套半遮"
                 "嘴角,神情轻蔑而世故地低声说:“政治联姻的工具罢了。”"
                 f"右侧军官目视前方倾听。说完后两人静止。{AUDIO_TAIL}")),
    dict(refs=["@BG", "男性军官"], duration=5, audio=True,
         prompt=("双人中近景,固定镜头:场景为@Image1所示大舞厅。两名深蓝"
                 "镶金军装军官并肩而立。右侧军官(容貌同@Image2)转过脸"
                 "望向画面外远处,眉间浮起怜悯,低声回应:“真可怜啊,"
                 "公爵千金……”左侧军官(容貌同@Image2)沉默观察。说完"
                 f"后两人恢复并肩静止,镜头固定。{AUDIO_TAIL}")),
    dict(refs=["@BG", "持折扇女子"], duration=6, audio=True,
         prompt=("下半脸特写,固定镜头:场景为@Image1所示大舞厅。@Image2"
                 "以黑色蕾丝折扇微微遮住下半张脸,只露出眼睛与扇沿,身后"
                 "宾客虚化。她一边缓缓收拢折扇,一边露出轻蔑上扬的嘴角,"
                 "望向画外低声讥讽:“真可怜啊,公爵千金……”说完后折扇"
                 f"收拢停在下巴旁,轻蔑笑意定格,镜头静止。{AUDIO_TAIL}")),
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
