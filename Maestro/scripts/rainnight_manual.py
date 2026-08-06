#!/usr/bin/env python3
"""rainnight 手写分镜 × WaveSpeed seedance-2.0(2026-08-06 用户令,
体例同 seedance_manual_v2.py;prompt 逐镜精写)。

═══ 接缝设计(导演判断,逐缝说明)═══════════════════════════════
  1→2  软钉   同车内续拍:镜头从车窗特写继续右移进入车内 —— ti2v
              软钉,@Image1=上一镜末帧,肖像顺延。
  2→3  转场桥 女子恐惧特写 → 双人枪戏(轴变+景别变):flf2v 桥,
              镜头缓慢拉远并左摇,车内空间连续,严禁原地变形。
  3→4  软钉   同构图续拍(侧窗碎裂是加诸既有画面的事件)。
  4→5  硬切   爆点切(枪口火焰的冲击力属于硬切,任何缓接都会泄劲)。
  5→6  软钉   烟雾余韵从火焰画面自然衰减,同帧续拍。
  拼接:1,2,[桥23],3,4,5,6
  素材:复用 run4(outputs/movie_20260806_010939)的背景板与生成肖像
  —— 三方对照(流水线/阉割版/手拍版)同一套身份与空间,变量干净。
═══════════════════════════════════════════════════════════════════

用法: python scripts/rainnight_manual.py          # 真实调用,花钱
输出: outputs/rainnight_manual_<ts>/
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.models.video_gen import build_video_gen
from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
from maestro.pipeline.window_loop import (_extract_frame0, _last_frame,
                                          _probe_seconds)
from maestro.tools.video_concat import VideoConcatTool

RUN4 = Path("/Users/kevin/Desktop/Kevin/repositories/VideoAgent/Maestro/"
            "outputs/movie_20260806_010939")
BG_PLATE = RUN4 / "anchors" / "bg_bg_1.png"
PORTRAITS = {"黑帮老大": RUN4 / "portraits" / "黑帮老大.png",
             "女子": RUN4 / "portraits" / "女子.png"}

# 转场桥(2→3):只写运镜,落在两端真实画面,缓速,防原地变形。
BRIDGE_23 = ("转场运镜:镜头从她的面部特写缓缓拉远并向左平摇,车内空间"
             "连续展开,直到两人同框的中近景构图对齐后停稳;速度舒缓,"
             "无剪辑感,人物不在原地变换面貌或服装。")

# mode: t2v(硬切/开场,带 refs)| pin(钉上镜末帧:@Image1=末帧,肖像顺延)
SHOTS = [
    dict(  # 1 定场缓推:暗巷轿车 → 车窗雪茄红光
        mode="t2v", refs=["@BG", "黑帮老大"], duration=8, audio=True,
        prompt=("大远景起手:场景与@Image1所示的霓虹雨夜暗巷完全一致——"
                "同一空间与灯色。一辆黑色豪华轿车静止在巷心,雨幕垂落,"
                "五彩霓虹在湿漉路面上拖出倒影。镜头沿巷道缓慢平稳推进,"
                "雨滴持续敲击车身;推至车窗外侧停稳,车窗上雨水蜿蜒流淌,"
                "折射着迷幻光斑;玻璃后@Image2点燃一支雪茄并深吸一口,"
                "红光在黑暗中明灭两次。镜头静止收尾,雪茄红光仍在玻璃后"
                "明灭。音频:只有雨滴敲击车窗声与雪茄燃起吸入声——"
                "无背景音乐、无人声。")),
    dict(  # 2 软钉续拍:穿入车内平移至女子
        mode="pin", refs=["女子"], duration=6, audio=True,
        # 运行时 @Image1=上一镜末帧;@Image2=女子
        prompt=("画面从@Image1精确开始——@Image1是上一镜的最后一帧,"
                "构图与光线与其完全一致,不重新构图。镜头从车窗外缓缓"
                "移入车内,向右平稳平移至副驾驶座:@Image2身穿晚礼服"
                "静坐,双手在膝上收紧,肩膀微微发抖,神情紧张恐惧;柔和"
                "的霓虹光斑缓慢扫过她的脸。镜头停稳在她的侧面中近景,"
                "她保持不安的静止。音频:只有低沉的心跳声——无背景音乐、"
                "无人声。")),
    dict(  # 3 双人枪戏(桥23 之后的全新构图)
        mode="t2v", refs=["@BG", "黑帮老大", "女子"], duration=6,
        audio=True,
        prompt=("车内双人中近景,固定镜头:场景光色与@Image1所示雨夜一致,"
                "高对比度戏剧打光。@Image2缓缓抬起一把金色手枪,以慢动作"
                "般的匀速将冰冷枪口抵在@Image3的额头上;@Image3闭眼"
                "哭泣,泪水沿脸颊滑落,身体僵直不敢动。金属在寂静中轻响。"
                "镜头全程静止,收尾定格在枪口抵额的构图。音频:只有冰冷"
                "金属摩擦声——无背景音乐、无人声。")),
    dict(  # 4 软钉续拍:侧窗碎裂加诸既有画面
        mode="pin", refs=["黑帮老大", "女子"], duration=5, audio=True,
        # 运行时 @Image1=上一镜末帧;@Image2=老大,@Image3=女子
        prompt=("画面从@Image1精确开始——@Image1是上一镜的最后一帧,"
                "枪口抵额的构图与光线完全一致,不重新构图。侧窗玻璃突然"
                "碎裂,碎片以慢动作在空中翻飞,每一片都映着@Image3绝望"
                "的目光;@Image2持枪的手纹丝不动。碎片纷纷坠落后画面"
                "复归死寂,镜头始终静止。音频:只有玻璃碎裂声——"
                "无背景音乐、无人声。")),
    dict(  # 5 爆点硬切:枪口火焰急推
        mode="t2v", refs=["@BG", "黑帮老大"], duration=5, audio=True,
        prompt=("急推镜头:场景为@Image1所示的黑暗雨夜巷道。镜头快速逼近"
                "@Image2手中的金色手枪,扣动扳机的瞬间一道金色枪口火焰"
                "撕裂黑暗,照亮整条雨巷;火星四溅,烟雾腾起,雨丝在火光"
                "中定格般发亮。镜头在火焰最亮处急停,画面定在爆发的一瞬。"
                "音频:只有突发的巨大枪声——无背景音乐、无人声。")),
    dict(  # 6 软钉余韵:烟雾衰减,双轮廓定格
        mode="pin", refs=["黑帮老大", "女子"], duration=5, audio=True,
        prompt=("画面从@Image1精确开始——@Image1是上一镜的最后一帧,"
                "火光与烟雾与其完全一致,不重新构图。金色火光缓缓衰减,"
                "烟雾向右缓慢弥漫,零星火星坠落熄灭;余光中@Image2与"
                "@Image3的轮廓渐渐沉入黑暗,只余雨幕与霓虹倒影。镜头"
                "静止,画面在近乎全暗中收尾。音频:只有逐渐减弱的金属"
                "回音与雨声——无背景音乐、无人声。")),
]

# (前镜索引, 后镜索引, 桥 prompt):桥插在后镜之前。
BRIDGES = [(1, 2, BRIDGE_23)]


def main() -> None:
    assert BG_PLATE.exists(), f"背景板缺失: {BG_PLATE}"
    for n, p in PORTRAITS.items():
        assert p.exists(), f"肖像缺失: {n} → {p}"
    roles = {n: str(p) for n, p in PORTRAITS.items()}
    out_dir = Path("outputs") / f"rainnight_manual_{time.strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir)

    vg = build_video_gen({"name": "wavespeed", "resolution": "480p",
                          "call_log": str(out_dir / "wavespeed_calls.jsonl")})
    bg_plate = BG_PLATE
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
                print(f"[shot {i+1}] 上镜末帧缺失 — 降级为 背景板+refs")
                kw["reference_images"] = [bg_plate] + [
                    roles[n] for n in sh["refs"]]
                use_prompt = re.sub(
                    r"^画面从@Image1精确开始——[^。]*。",
                    "场景为@Image1所示的霓虹雨夜。", sh["prompt"])
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

    # 转场桥:末帧(前)→ 首帧(后),flf2v,只写运镜
    bridge_before: dict = {}
    if hasattr(vg, "frame_to_frame"):
        for a, b, bprompt in BRIDGES:
            if not (shots_out[a] and shots_out[b]):
                continue
            try:
                fa = _last_frame(Path(shots_out[a]),
                                 out_dir / f"bridge_{a}{b}_prev.png")
                fb = _extract_frame0(Path(shots_out[b]),
                                     out_dir / f"bridge_{a}{b}_next.png")
                if fa is None or fb is None:
                    continue
                vg.generate_audio = False
                bp = vg.frame_to_frame(
                    prompt=bprompt, first_frame=fa, last_frame=fb,
                    out_path=out_dir / f"bridge_{a}{b}.mp4",
                    duration=4, seed=777)
                bridge_before[b] = Path(bp)
                print(f"[桥 {a+1}→{b+1}] OK")
                ledger.append({"bridge": [a, b], "ok": True})
            except Exception as exc:
                print(f"[桥 {a+1}→{b+1}] FAILED: {exc} — 硬切保底")
                ledger.append({"bridge": [a, b], "ok": False,
                               "error": str(exc)[:300]})

    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))
    clips = []
    for i, v in enumerate(shots_out):
        if i in bridge_before:
            clips.append(bridge_before[i])
        if v:
            clips.append(v)
    if clips:
        concat_in = normalize_for_concat(clips, out_dir / "concat_norm") \
            if any_audio(clips) else clips
        final = VideoConcatTool().run(concat_in, out_dir / "movie.mp4")
        print("成片:", final, f"{_probe_seconds(Path(final)):.1f}s")


if __name__ == "__main__":
    main()
