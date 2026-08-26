#!/usr/bin/env python3
"""《末班车》demo —— 双人对话 · 两段 30s seedance-2.5 · 双肖像锚定。

结构(2026-08-26 用户裁决):
  前 30s = 场景一(雨夜出租车内对话)   后 30s = 场景二(深夜路边面摊)
  每段一次生成;两段之间叙事硬切。故事发生在【同一夜】——服装天然
  不变,这是为一致性服务的剧作选择。

人物一致性(按重要度):
  ① 两段引用【同一对】官方肖像:@Image1=司机,@Image2=青年;
  ② 两个角色的外观 canon 句两段【逐字相同】(CANON 常量单一来源);
  ③ 角色在年龄/体型/服色上刻意拉开(防换脸):花白寸头灰毛衣 vs
     黑发牛仔外套 + 酒红吉他包(第二签名物,全程不离身);
  ④ 嗓音描述句逐字重复(VOICE)—— 已知风险:两次独立生成之间音色
     仍可能不一致,无硬解,台词刻意压少(前段 3 句、后段 1 句+哼唱)。

用法: python scripts/demo_lastcab.py                # 真实调用,花钱
      python scripts/demo_lastcab.py --dry-run      # 只看 prompt
      python scripts/demo_lastcab.py --portrait-driver X.png --portrait-youth Y.png
输出: outputs/demo_lastcab_<ts>/(每次重跑新目录)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MODEL_ID = os.environ.get("SEEDANCE_MODEL",
                          "bytedance/seedance-2.5/text-to-video")

# ── 逐字复用的三段(常量单一来源 → 两段 prompt 自然逐字相同)─────────
STYLE = ("写实电影质感,35mm 胶片色调,浅景深,雨夜霓虹的冷暖对比,"
         "画面中不出现任何文字。")
CANON = ("@Image1 是老司机:五十八岁男性,花白寸头,面容温和多皱纹,"
         "穿深灰色开衫毛衣、内搭棕色格子衬衫,胸前挂一副老花镜——"
         "外貌一律以 @Image1 为准,服装全片保持不变。"
         "@Image2 是青年:二十四岁男性,黑色微卷短发,身形清瘦,"
         "穿浅蓝色牛仔外套、白色T恤,始终背着酒红色吉他包——"
         "外貌一律以 @Image2 为准,服装全片保持不变。")
VOICE = ("司机的声音:低沉沙哑的中年男声,语速缓慢。"
         "青年的声音:清亮而略带疲惫的年轻男声。")

PORTRAIT_DRIVER = (
    "Waist-up studio portrait photograph of a 58-year-old East Asian man, "
    "short grey buzz cut, gentle wrinkled face, warm tired eyes, wearing a "
    "dark grey cardigan over a brown plaid shirt, reading glasses hanging "
    "on a cord on his chest, facing the camera, neutral light-grey "
    "background, soft even lighting, photorealistic, sharp focus, no text")
PORTRAIT_YOUTH = (
    "Waist-up studio portrait photograph of a 24-year-old East Asian man, "
    "short slightly curly black hair, slim build, wearing a light-blue "
    "denim jacket over a white t-shirt, a wine-red guitar case strap over "
    "his shoulder, facing the camera, neutral light-grey background, soft "
    "even lighting, photorealistic, sharp focus, no text")

SEG1 = f"""{STYLE}
场景:雨夜城市街道上行驶的出租车内,车窗外霓虹流动,全段唯一场景。
{CANON}
{VOICE}
音频总则:雨声、引擎低鸣与雨刷声贯穿全段;台词只有下方引号内三句;可伴随契合氛围的配乐,但不得盖过台词。

Shot 1: 车外中景转车窗特写。雨夜街道,一辆亮着顶灯的出租车驶来停下;透过挂满雨珠的车窗,@Image2 抱着酒红色吉他包坐进后座,关上车门。
Shot 2: 车内中景,从前排越过 @Image1 的肩膀拍后座。@Image2 靠窗而坐,霓虹光影流过他的脸,低头无言;@Image1 从后视镜里看了他一眼,开口说:"这么晚,去哪?"
Shot 3: 反打,后座视角拍前排。@Image2 抬起头,轻声说:"随便开吧……今晚是我最后一场演出,台下没有人。"
Shot 4: 特写。@Image1 的双手轻扣方向盘,目光看着前方雨路,沉默片刻,开口说:"我年轻的时候,也吹小号。"
Shot 5: 中景收尾。红灯,车缓缓停下;两人都没有再说话,雨刷规律摆动,@Image2 转头望向窗外霓虹。镜头静止,两人保持各自姿态。"""

SEG2 = f"""{STYLE}
场景:当夜更深时,街角路边面摊——暖黄灯串、翻滚的蒸汽、几张小桌,出租车停在摊旁,全段唯一场景。
{CANON}
{VOICE}
音频总则:蒸汽声、稀落夜市声与远处车流贯穿全段;台词只有下方引号内一句,另有青年的无词哼唱;可伴随契合氛围的配乐,但不得盖过台词与哼唱。

Shot 1: 远景。深夜街角面摊,暖黄灯串在夜色里亮着,出租车停在摊旁;@Image1 与背着酒红色吉他包的 @Image2 在小桌两侧对坐,面前各一碗冒着热气的面。
Shot 2: 中景。两人吃面;@Image1 放下筷子,看着对面的青年,缓缓说:"再唱一次,唱给我听。"
Shot 3: 特写。@Image2 愣住,筷子停在半空;片刻后他轻轻放下筷子,低头笑了一下。
Shot 4: 中近景。@Image2 轻声哼起一段温柔的旋律,没有歌词,蒸汽在灯光里缭绕;@Image1 靠着椅背静静地听,手指在桌沿轻轻打着拍子。
Shot 5: 缓慢拉远收尾。灯串下两人相视而笑,摊主在背景里添汤;镜头缓缓拉远,定格在夜色中这一角温暖的黄光。两人保持坐姿不动。"""


def _load_env() -> None:
    envf = REPO / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _build_vg(out_dir: Path, resolution: str):
    from maestro.models.video_gen import build_video_gen
    vg = build_video_gen({
        "name": "wavespeed", "model_id": MODEL_ID,
        "resolution": resolution,
        "duration_range": [4, 30],          # 2.5 的 30s 档;钳位逃生舱
        "call_log": str(out_dir / "wavespeed_calls.jsonl")})
    # 后端 range-family 白名单只认 "seedance-2.0" → 2.5 的参考图通道会被
    # 硬拒;实例级遮蔽只影响本 demo 进程,生产代码一字不动。
    vg._is_range_family = lambda mid: True
    return vg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--portrait-driver", default=None)
    ap.add_argument("--portrait-youth", default=None)
    ap.add_argument("--seg2-rolls", type=int, default=1,
                    help="后段掷几条(>1 时拼接用第 1 条,其余供人工挑)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt")
    a = ap.parse_args()

    if a.dry_run:
        print("═══ 肖像 t2i · 司机 ═══\n" + PORTRAIT_DRIVER)
        print("\n═══ 肖像 t2i · 青年 ═══\n" + PORTRAIT_YOUTH)
        print("\n═══ 前 30s(车内)═══\n" + SEG1)
        print("\n═══ 后 30s(面摊)═══\n" + SEG2)
        return

    _load_env()
    assert os.environ.get("WAVESPEED_API_KEY"), "缺 WAVESPEED_API_KEY(.env)"
    out_dir = Path("outputs") / f"demo_lastcab_{time.strftime('%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir, "| 模型:", MODEL_ID, "|", a.resolution)

    vg = _build_vg(out_dir, a.resolution)
    ledger: list = []

    # ① 双肖像(@Image1=司机,@Image2=青年 —— 顺序即编号,两段一致)
    ports: list = []
    for tag, given, prompt in (
            ("driver", a.portrait_driver, PORTRAIT_DRIVER),
            ("youth", a.portrait_youth, PORTRAIT_YOUTH)):
        if given:
            p = Path(given)
            assert p.exists(), f"肖像不存在: {p}"
            print(f"肖像 {tag}(复用):", p)
        else:
            p = vg.text_to_image(prompt, out_dir / f"portrait_{tag}.png")
            print(f"肖像 {tag}(新生成):", p,
                  "—— 不满意可重跑,或下次 --portrait-* 固定")
        ports.append(p)
        ledger.append({"step": f"portrait_{tag}", "path": str(p)})

    # ② 两段 30s(原生音频开:要台词与环境声)
    vg.generate_audio = True
    segs: list = []
    for tag, prompt, seed in (("seg1_cab", SEG1, a.seed),
                              ("seg2_noodles", SEG2, a.seed + 500)):
        rolls = a.seg2_rolls if tag == "seg2_noodles" else 1
        first = None
        for k in range(rolls):
            outp = out_dir / (f"{tag}.mp4" if k == 0
                              else f"{tag}_roll{k + 1}.mp4")
            print(f"[{tag}] roll {k + 1}/{rolls} 30s seed={seed + 101 * k} …")
            t0 = time.time()
            vg.generate(prompt, 30, outp, fps=24, seed=seed + 101 * k,
                        reference_images=list(ports))
            print(f"  完成 {time.time() - t0:.0f}s → {outp}")
            ledger.append({"step": tag, "roll": k + 1, "path": str(outp),
                           "seed": seed + 101 * k, "prompt": prompt})
            first = first or outp
        segs.append(first)
    if a.seg2_rolls > 1:
        print("⚠️ 后段掷了多条:拼接默认用第 1 条;人脸/音色更像的那条请"
              "改名为 seg2_noodles.mp4 后重跑拼接(或 ffmpeg 手拼)。")

    # ③ 拼接(硬切;管线同款音轨归一)
    from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
    from maestro.tools.video_concat import VideoConcatTool
    clips = [p for p in segs if p]
    concat_in = (normalize_for_concat(clips, out_dir / "concat_norm")
                 if any_audio(clips) else clips)
    final = VideoConcatTool().run(concat_in, out_dir / "lastcab_60s.mp4")
    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))
    print("成片:", final)


if __name__ == "__main__":
    main()
