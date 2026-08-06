#!/usr/bin/env python3
"""xiaoming 手写分镜 × WaveSpeed seedance-2.0 对照实验(2026-08-06 用户令,
体例同 seedance_manual_v2.py)。

═══ 接缝设计(导演判断,逐缝说明)═══════════════════════════════
  1→2  软钉   阿浪【俯冲进入】小明的既有画面 —— 同画面续拍是这一缝的
              本体:ti2v 软钉,@Image1=上一镜末帧,肖像顺延,首句显式
              声明"画面从@Image1精确开始"。
  2→3  硬切   对话轴反打(阿浪落礁石说完 → 小明反应):两侧人物同场
              已确立,反打硬切成立,t2v+refs 全新构图。
  3→4  硬切   反打回阿浪(扑翅抖屑+大段台词)。
  4→5  硬切   反打回小明(怔住→低头→抬头问)。
  5→6  硬切   反打回阿浪(翻白眼→展翅高飞远去)。
  6→7  硬切   阿浪已出画,小明收束镜(捡包→甩水→对海大喊)。
  台词法:每镜一个说话人,整块台词逐字照抄,绝不截断;压制句在场。
  拼接:1,2,3,4,5,6,7(反打硬切无需转场桥)。
═══════════════════════════════════════════════════════════════════

用法: python scripts/xiaoming_manual.py          # 真实调用,花钱
输出: outputs/xiaoming_manual_<ts>/
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.models.video_gen import build_video_gen
from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
from maestro.pipeline.script_input import parse_script_json
from maestro.pipeline.window_loop import _last_frame, _probe_seconds
from maestro.tools.video_concat import VideoConcatTool

AUDIO_TAIL = "音频:只有角色说这句台词的人声——无背景音乐、无音效。"

# 背景板:黄昏海边,浪大,礁石在侧,中央留空,零主角。
BG_PLATE_PROMPT = (
    "Empty seaside at dusk: a wide wet sand beach with the low golden "
    "setting sun near the horizon, powerful waves rolling in with white "
    "foam and spray, a few jagged black reef rocks at one side including "
    "one low flat-topped rock, warm golden light across the water, the "
    "central foreground open and empty, deep focus, eye-level wide view, "
    "no people, no birds, no modern objects.")

# mode: t2v(硬切,带 refs)| pin(钉上镜末帧:@Image1=末帧,肖像顺延)
SHOTS = [
    dict(  # 1 定场+小明独白(t2v 冷开场)
        mode="t2v", refs=["@BG", "小明"], duration=10, audio=True,
        prompt=("特写起手:场景为@Image1所示的黄昏海边。@Image2紧攥的"
                "拳头占据画面,指节发白;镜头平稳后拉至中近景,@Image2"
                "死死盯着海面,乱发被海风吹得像鸟窝,湿透的公文包放在"
                "脚边,另一只手拿着半块三明治。@Image2望着海面说:"
                "“大海真大啊，大到能吞掉我所有的失败。”说完保持凝望,"
                f"镜头静止。{AUDIO_TAIL}")),
    dict(  # 2 软钉续拍:阿浪俯冲入画抢三明治→落礁石→毒舌
        mode="pin", refs=["阿浪", "小明"], duration=12, audio=True,
        # 运行时 @Image1=上一镜末帧;@Image2=阿浪,@Image3=小明
        prompt=("画面从@Image1精确开始——@Image1是上一镜的最后一帧,"
                "构图、人物与光线与其完全一致,不重新构图。@Image2从画面"
                "上方突然俯冲而下,精准抢走@Image3手里的半块三明治,"
                "随即落在旁边的礁石上歪着头;镜头跟随并停在@Image2的"
                "中近景。@Image2歪头看着画外说:“喂，人类。你在那装"
                "什么深沉？眼泪都快比海水咸了，也没见你跳下去洗个澡。”"
                f"说完保持歪头,镜头停稳。{AUDIO_TAIL}")),
    dict(  # 3 硬切反打:小明苦笑吐露心声
        mode="t2v", refs=["@BG", "小明"], duration=10, audio=True,
        prompt=("中近景,固定镜头:场景为@Image1所示的黄昏海边。@Image2"
                "愣住,望向画外礁石方向,眼神痛苦,嘴角挤出一丝苦笑,"
                "肩膀缓缓垮下。@Image2苦笑着说:“你一只鸟懂什么？我在想，"
                "是不是该彻底消失算了。”说完垂下目光,镜头静止。"
                f"{AUDIO_TAIL}")),
    dict(  # 4 硬切反打:阿浪抖屑+核心大段
        mode="t2v", refs=["@BG", "阿浪"], duration=15, audio=True,
        prompt=("中近景,固定镜头:场景为@Image1所示的黄昏海边礁石上。"
                "@Image2扑腾着翅膀,把三明治屑向画外抖去,眼神犀利;"
                "身后海浪拍碎又卷土重来。@Image2犀利地说:“消失？大海"
                "只收垃圾，不收懦夫。你看这浪，拍碎了一万次，下一秒照样"
                "卷土重来。真正的强大不是从不跌倒，而是满身沙砾还敢对着"
                f"太阳大笑。”说完直视画外,镜头静止。{AUDIO_TAIL}")),
    dict(  # 5 硬切反打:小明怔住→低头→抬头轻问
        mode="t2v", refs=["@BG", "小明"], duration=8, audio=True,
        prompt=("中近景,固定镜头:场景为@Image1所示的黄昏海边,夕阳把"
                "海面染成金色,波光粼粼。@Image2怔住,低头看着怀里的"
                "面包屑,又抬头望向画外礁石方向。@Image2轻声问:"
                "“你……是在安慰我吗？”说完目光停在画外,镜头静止。"
                f"{AUDIO_TAIL}")),
    dict(  # 6 硬切反打:阿浪翻白眼→大段告别→展翅高飞
        mode="t2v", refs=["@BG", "阿浪"], duration=12, audio=True,
        prompt=("中近景起手:场景为@Image1所示的黄昏海边礁石上。@Image2"
                "翻了个白眼,一边展开翅膀一边说:“少自作多情！我只是怕"
                "你饿死了，没人给我投喂薯片！记住，明天太阳升起时，别让"
                "我看见你还像个落汤鸡！”说完振翅起飞,镜头微微上摇目送"
                f"它远去,变成天边一个小点,镜头停稳。{AUDIO_TAIL}")),
    dict(  # 7 收束镜:小明捡包甩水,对海大喊,浪漫脚踝
        mode="t2v", refs=["@BG", "小明"], duration=10, audio=True,
        prompt=("中景,固定镜头:场景为@Image1所示的黄昏海边。@Image2"
                "深吸一口气,猛地弯腰捡起脚边湿透的公文包,用力甩干"
                "上面的水珠;随后挺直身体面向大海,声音颤抖却有力地"
                "大喊:“好！那就从头再来！”海浪涌上沙滩,温柔地漫过"
                "@Image2的脚踝;夕阳沉向海平线,@Image2握着公文包静立,"
                f"镜头静止。{AUDIO_TAIL}")),
]


def main() -> None:
    parsed = parse_script_json(
        Path("/Users/kevin/Desktop/script-xiaoming/script.json"))
    roles = parsed["roles"]
    out_dir = Path("outputs") / f"xiaoming_manual_{time.strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    vg = build_video_gen({"name": "wavespeed", "resolution": "480p",
                          "call_log": str(out_dir / "wavespeed_calls.jsonl")})
    bg_plate = vg.text_to_image(BG_PLATE_PROMPT, out_dir / "bg_plate.png")
    print("背景板:", bg_plate)
    shots_out: list = [None] * len(SHOTS)
    ledger = []

    for i, sh in enumerate(SHOTS):
        outp = out_dir / f"shot{i:03d}.mp4"
        vg.generate_audio = bool(sh["audio"])
        kw = {}
        tag = sh["mode"]
        use_prompt = sh["prompt"]
        if sh["mode"] == "pin":
            prev_v = shots_out[i - 1]
            lf = _last_frame(Path(prev_v),
                             out_dir / f"pin_{i:03d}.png") if prev_v else None
            if lf is None:
                # 降级铁律:撕钉帧前言,@Image1 改背景板,肖像编号不变
                print(f"[shot {i+1}] 上镜末帧缺失 — 降级为 背景板+refs")
                kw["reference_images"] = [bg_plate] + [
                    roles[n] for n in sh["refs"]]
                use_prompt = re.sub(
                    r"^画面从@Image1精确开始——[^。]*。",
                    "场景为@Image1所示的黄昏海边。", sh["prompt"])
                tag = "t2v_degraded"
            else:
                kw["reference_images"] = [lf] + [roles[n]
                                                 for n in sh["refs"]]
                tag = "ti2v_pin"
        else:
            kw["reference_images"] = [
                bg_plate if n == "@BG" else roles[n] for n in sh["refs"]]
        print(f"[shot {i+1}/{len(SHOTS)}] {tag} {sh['duration']}s "
              f"audio={sh['audio']}")
        try:
            vg.generate(use_prompt, sh["duration"], outp, fps=24,
                        seed=0, **kw)
            shots_out[i] = outp
            ledger.append({"shot": i, "mode": tag, "ok": True,
                           "prompt": use_prompt})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            ledger.append({"shot": i, "mode": tag, "ok": False,
                           "error": str(exc)[:300]})

    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))
    clips = [v for v in shots_out if v]
    if clips:
        concat_in = normalize_for_concat(clips, out_dir / "concat_norm") \
            if any_audio(clips) else clips
        final = VideoConcatTool().run(concat_in, out_dir / "movie.mp4")
        print("成片:", final, f"{_probe_seconds(Path(final)):.1f}s")


if __name__ == "__main__":
    main()
