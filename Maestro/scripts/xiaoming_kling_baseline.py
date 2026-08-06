#!/usr/bin/env python3
"""xiaoming 可灵基线消融(2026-08-06 用户令):对照 run6
(outputs/movie_20260805_201551)——同 12 镜、同背景板、同肖像、同可灵,
但【纯基线切镜:零钉帧(不用上一镜尾帧)、零转场桥、零连续性语句、
prompt 降档粗写】,每镜独立 ref2v,直接硬拼。
隔离出的变量 = 钉/切路由 + 接缝机械 + prompt 工艺。

用法: python scripts/xiaoming_kling_baseline.py
输出: outputs/xiaoming_kling_baseline_<ts>/
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

# run6 的背景板原样复用(不再花 t2i 钱)
BG_PLATE = Path("outputs/movie_20260805_201551/anchors/bg_bg_1.png")

# 12 镜,与 run6 分镜一一对应;prompt 降档粗写,台词逐字。
# cast: 引用图顺序(<<<image_1>>>=背景板,人物顺延);audio: 对白开
SHOTS = [
    dict(cast=["小明"], duration=8, audio=True,
         prompt=("<<<image_1>>>的黄昏海边。特写<<<image_2>>>攥紧的拳头,"
                 "拉到中近景,他盯着海面,头发被吹乱,脚边放着湿公文包,"
                 "手里拿着半块三明治,说:“大海真大啊，大到能吞掉我所有"
                 f"的失败。”{AUDIO_TAIL}")),
    dict(cast=["小明", "阿浪"], duration=5, audio=False,
         prompt=("<<<image_1>>>的海边。<<<image_3>>>俯冲下来抢走"
                 "<<<image_2>>>手里的三明治,落在礁石上。")),
    dict(cast=["小明", "阿浪"], duration=10, audio=True,
         prompt=("<<<image_1>>>的海边礁石上。<<<image_3>>>叼着三明治歪头,"
                 "说:“喂，人类。你在那装什么深沉？眼泪都快比海水咸了，"
                 f"也没见你跳下去洗个澡。”{AUDIO_TAIL}")),
    dict(cast=["小明"], duration=8, audio=True,
         prompt=("<<<image_1>>>的海边。<<<image_2>>>中近景,愣住苦笑,"
                 "说:“你一只鸟懂什么？我在想，是不是该彻底消失算了。”"
                 f"{AUDIO_TAIL}")),
    dict(cast=["阿浪"], duration=5, audio=True,
         prompt=("<<<image_1>>>的海边礁石上。<<<image_2>>>扑翅膀抖掉"
                 f"三明治屑,说:“消失？大海只收垃圾，不收懦夫。”{AUDIO_TAIL}")),
    dict(cast=["阿浪"], duration=10, audio=True,
         prompt=("<<<image_1>>>的海边礁石上。<<<image_2>>>用翅膀指着身后"
                 "的海浪,说:“你看这浪，拍碎了一万次，下一秒照样卷土重来。"
                 "真正的强大不是从不跌倒，而是满身沙砾还敢对着太阳大笑。”"
                 f"{AUDIO_TAIL}")),
    dict(cast=["小明"], duration=5, audio=True,
         prompt=("<<<image_1>>>的海边,夕阳金光。<<<image_2>>>低头看看"
                 f"怀里的面包屑再抬头,问:“你……是在安慰我吗？”{AUDIO_TAIL}")),
    dict(cast=["阿浪"], duration=8, audio=True,
         prompt=("<<<image_1>>>的海边礁石上。<<<image_2>>>翻了个白眼,"
                 "说:“少自作多情！我只是怕你饿死了，没人给我投喂薯片！”"
                 f"{AUDIO_TAIL}")),
    dict(cast=["小明", "阿浪"], duration=8, audio=True,
         prompt=("<<<image_1>>>的海边。<<<image_3>>>展翅起飞,一边回头"
                 "说:“记住，明天太阳升起时，别让我看见你还像个落汤鸡！”"
                 f"<<<image_2>>>在下面仰望。{AUDIO_TAIL}")),
    dict(cast=["小明"], duration=5, audio=False,
         prompt=("<<<image_1>>>的海边。海鸥飞远了,<<<image_2>>>深吸一口气,"
                 "低头看脚边的湿公文包。")),
    dict(cast=["小明"], duration=5, audio=False,
         prompt=("<<<image_1>>>的海边。<<<image_2>>>弯腰捡起湿公文包,"
                 "用力甩掉水珠。")),
    dict(cast=["小明"], duration=8, audio=True,
         prompt=("<<<image_1>>>的海边。<<<image_2>>>提着公文包对大海大喊:"
                 "“好！那就从头再来！”镜头拉远,海浪漫过脚踝,夕阳下沉。"
                 f"{AUDIO_TAIL}")),
]


def main() -> None:
    parsed = parse_script_json(
        Path("/Users/kevin/Desktop/script-xiaoming/script.json"))
    roles = parsed["roles"]
    assert BG_PLATE.exists(), f"背景板缺失: {BG_PLATE}"
    out_dir = (Path("outputs")
               / f"xiaoming_kling_baseline_{time.strftime('%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    vg = build_video_gen({
        "name": "bailian_kling",
        "kling_model": "kling/kling-v3-video-generation",
        "kling_omni_model": "kling/kling-v3-omni-video-generation",
        "mode": "std", "aspect_ratio": "16:9",
        "generate_audio": False, "poll_interval": 15.0, "timeout": 900,
        "call_log": str(out_dir / "kling_calls.jsonl")})

    clips, ledger = [], []
    for i, sh in enumerate(SHOTS):
        outp = out_dir / f"shot{i:03d}.mp4"
        vg.generate_audio = bool(sh["audio"])
        refs = [BG_PLATE] + [Path(roles[n]) for n in sh["cast"]]
        print(f"[shot {i+1}/{len(SHOTS)}] ref2v {sh['duration']}s "
              f"audio={sh['audio']} refs={len(refs)}")
        try:
            vg.generate(sh["prompt"], sh["duration"], outp, fps=24, seed=0,
                        reference_images=refs)
            clips.append(outp)
            ledger.append({"shot": i, "ok": True, "prompt": sh["prompt"]})
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
