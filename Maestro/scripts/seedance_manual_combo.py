#!/usr/bin/env python3
"""合镜实验(2026-08-05 用户令):手写分镜 1+2 并成一条 15s 单镜——
大远景拉近 → 俯冲地板倒影 → 沿背影上摇过肩 → 退婚宣告。
无接缝(镜内连续运镜替代钉帧),一次 t2v,复用 224340 的背景板。

用法: python scripts/seedance_manual_combo.py
输出: outputs/seedance_combo_<ts>/
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.models.video_gen import build_video_gen
from maestro.pipeline.script_input import parse_script_json

AUDIO_TAIL = "音频:只有角色说这句台词的人声——无背景音乐、无音效。"

BG_PLATE = Path("outputs/seedance_manual_224340/bg_plate.png")

# @Image1=背景板 @Image2=安莉希娅 @Image3=芬莱克 @Image4=安娜
REFS = ["安莉希娅", "芬莱克殿下", "安娜"]
PROMPT = (
    "一镜到底:大远景起手,场景与@Image1所示的宫廷大舞厅完全一致——"
    "同一空间、陈设与烛光。@Image2挽着@Image3的手臂静立在舞台台阶前的"
    "中央,@Image4独自站在他们对面不远处,孤立无援;宾客仅在远处轻微"
    "走动交谈。镜头从大远景平稳逐渐推近,随后下俯落到抛光大理石地面的"
    "特写:地面映出@Image2与@Image3并立的倒影,烛光闪烁;@Image4的"
    "脚步走入倒影视野,步伐平稳。镜头再从@Image4的脚部沿背影平稳上摇,"
    "越过她的肩膀形成过肩中近景;@Image3的脸清晰入画,神情冷酷严厉,"
    "开口说:“你这种女人不配做王后,安莉希娅才是真正合适的人选!”"
    "@Image2挽着他的手臂未动。说完后三人静止,镜头停稳。全程一个"
    f"连续镜头,无剪辑感。{AUDIO_TAIL}")


def main() -> None:
    parsed = parse_script_json(
        Path("/Users/kevin/Desktop/script-wedding/script.json"))
    roles = parsed["roles"]
    assert BG_PLATE.exists(), f"背景板缺失: {BG_PLATE}"
    out_dir = Path("outputs") / f"seedance_combo_{time.strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    vg = build_video_gen({"name": "wavespeed", "resolution": "720p",
                          "call_log": str(out_dir / "wavespeed_calls.jsonl")})
    vg.generate_audio = True
    out = vg.generate(PROMPT, 15, out_dir / "combo_12.mp4", fps=24, seed=0,
                      reference_images=[BG_PLATE] + [roles[n] for n in REFS])
    print("合镜成片:", out)


if __name__ == "__main__":
    main()
