#!/usr/bin/env python3
"""《灯塔守夜人》demo —— 单人 · 两段 30s seedance-2.5 · 肖像锚定一致性。

结构(2026-08-26 用户裁决):
  前 30s = 场景一(白日悬崖灯塔)   后 30s = 场景二(暴风雨夜灯室)
  每段一次生成 → 段内一致性免费;两段之间是叙事硬切,不做像素续接。

人物一致性(按重要度):
  ① 两段引用【同一份】官方肖像(@Image1);
  ② 外观 canon 句两段【逐字相同】(CANON 常量单一来源);
  ③ 剧作层:同一天,服装(芥末黄雨衣)不变;
  ④ 风格句逐字相同(STYLE),质感不跳;
  ⑤ --seg2-rolls 2 可为后段多掷,人工挑脸。

用法: python scripts/demo_lighthouse.py              # 真实调用,花钱
      python scripts/demo_lighthouse.py --dry-run    # 只看 prompt
      python scripts/demo_lighthouse.py --portrait outputs/.../portrait.png
输出: outputs/demo_lighthouse_<ts>/(每次重跑新目录)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# seedance-2.5 的 30s 档;真实 id 以 wavespeed 控制台为准,可用环境变量覆盖
MODEL_ID = os.environ.get("SEEDANCE_MODEL",
                          "bytedance/seedance-2.5/text-to-video")

# ── 逐字复用的两句(一致性纪律②④:常量单一来源,两段自然逐字相同)──
STYLE = ("写实电影质感,35mm 胶片色调,浅景深,自然光效,"
         "画面中不出现任何文字。")
CANON = ("@Image1 是守塔人:六十多岁男性,花白浓密络腮胡,深褐色风霜"
         "面孔,目光温和,头戴藏青色毛线帽,身穿芥末黄色连帽厚雨衣、"
         "内搭深灰色毛衣、脚穿黑色胶靴——外貌一律以 @Image1 为准,"
         "服装全片保持不变。")

PORTRAIT_PROMPT = (
    "Waist-up studio portrait photograph of a man in his late 60s, thick "
    "bushy white full beard, weathered tanned face with kind eyes, navy "
    "knit beanie, mustard-yellow hooded heavy rain jacket over a dark grey "
    "wool sweater, facing the camera, neutral light-grey background, soft "
    "even lighting, photorealistic, sharp focus on the face, no text")

SEG1 = f"""{STYLE}
场景:清晨,海雾未散的悬崖顶白色灯塔,灰蓝色调,全段唯一场景。
{CANON}
音频总则:全段只有海浪、海鸥与风声,无背景音乐、无人声。

Shot 1: 大远景,固定机位。海雾中的悬崖与白色灯塔,海鸥掠过;@Image1 沿塔外螺旋石阶稳步向上走,芥末黄雨衣在灰蓝色调中格外醒目。
Shot 2: 中景,缓慢推近。灯室内,@Image1 用绒布仔细擦拭巨大的菲涅尔透镜,透镜折射的晨光在他脸上缓缓流动。
Shot 3: 特写。布满老茧的双手转动黄铜旋钮,齿轮咬合转动;他抬起头,花白络腮胡与专注的侧脸。
Shot 4: 中近景。@Image1 立于灯塔环形露台的栏杆前眺望海面,海风吹动胡须,他抬手在眉前搭起凉棚。
Shot 5: 远景,缓慢拉远。天色渐沉,远处海平线上乌云压顶;@Image1 的黄色身影伫立塔顶望向乌云方向。镜头静止收尾,他保持眺望姿态不动。"""

SEG2 = f"""{STYLE}
场景:当天深夜,暴风雨中的灯塔灯室与外景,黑蓝色调、油灯暖光,全段唯一场景。
{CANON}
音频总则:全段只有狂风、暴雨、雷声与灯械转动声,无背景音乐、无人声。

Shot 1: 大远景。黑夜暴雨,巨浪拍击悬崖,闪电照亮灯塔剪影,塔顶尚未亮灯。
Shot 2: 中景。灯室内油灯摇曳,@Image1 的雨衣还在滴水,双手奋力摇动黄铜曲柄,齿轮由慢到快转动。
Shot 3: 特写接广角。灯芯骤然点亮,巨大透镜放出雪亮光束;光束刺破雨幕,扫过咆哮的黑色海面。
Shot 4: 远景。光束扫到远处一艘小渔船,渔船在光的指引下调转船头,驶离暗礁方向。
Shot 5: 中近景收尾。@Image1 捧着一杯冒热气的茶,倚在灯室窗边望向海面,露出微笑,身后光束规律旋转。镜头静止,他保持微笑姿态。"""


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
    # 后端的 range-family 白名单只认 "seedance-2.0",2.5 会被判成旧枚举
    # 家族 → reference_images 通道被硬拒。实例级遮蔽只影响本 demo 进程,
    # 生产代码一字不动;若日后后端正式收编 2.5,删掉这行即可。
    vg._is_range_family = lambda mid: True
    return vg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--portrait", default=None,
                    help="复用已批准的肖像(不传则现场 t2i 生成)")
    ap.add_argument("--seg2-rolls", type=int, default=1,
                    help="后段掷几条(>1 时拼接用第 1 条,其余供人工挑)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt")
    a = ap.parse_args()

    if a.dry_run:
        print("═══ 肖像 t2i ═══\n" + PORTRAIT_PROMPT)
        print("\n═══ 前 30s(白日)═══\n" + SEG1)
        print("\n═══ 后 30s(风暴夜)═══\n" + SEG2)
        return

    _load_env()
    assert os.environ.get("WAVESPEED_API_KEY"), "缺 WAVESPEED_API_KEY(.env)"
    out_dir = Path("outputs") / f"demo_lighthouse_{time.strftime('%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print("输出目录:", out_dir, "| 模型:", MODEL_ID, "|", a.resolution)

    vg = _build_vg(out_dir, a.resolution)
    ledger: list = []

    # ① 肖像(一致性纪律①:两段共用这一份)
    if a.portrait:
        portrait = Path(a.portrait)
        assert portrait.exists(), f"肖像不存在: {portrait}"
        print("肖像(复用):", portrait)
    else:
        portrait = vg.text_to_image(PORTRAIT_PROMPT,
                                    out_dir / "portrait_keeper.png")
        print("肖像(新生成):", portrait,
              "—— 不满意可重跑,或下次 --portrait 固定它")
    ledger.append({"step": "portrait", "path": str(portrait)})

    # ② 两段 30s(vg.generate_audio 开:要海浪/风雨的原生环境音)
    vg.generate_audio = True
    segs: list = []
    for tag, prompt, seed in (("seg1_day", SEG1, a.seed),
                              ("seg2_storm", SEG2, a.seed + 500)):
        rolls = a.seg2_rolls if tag == "seg2_storm" else 1
        first = None
        for k in range(rolls):
            outp = out_dir / (f"{tag}.mp4" if k == 0
                              else f"{tag}_roll{k + 1}.mp4")
            print(f"[{tag}] roll {k + 1}/{rolls} 30s seed={seed + 101 * k} …")
            t0 = time.time()
            vg.generate(prompt, 30, outp, fps=24, seed=seed + 101 * k,
                        reference_images=[portrait])
            print(f"  完成 {time.time() - t0:.0f}s → {outp}")
            ledger.append({"step": tag, "roll": k + 1, "path": str(outp),
                           "seed": seed + 101 * k, "prompt": prompt})
            first = first or outp
        segs.append(first)
    if a.seg2_rolls > 1:
        print("⚠️ 后段掷了多条:拼接默认用第 1 条;换脸更像的那条请手动"
              "改名为 seg2_storm.mp4 后重跑拼接(或 ffmpeg 手拼)。")

    # ③ 拼接(硬切;管线同款音轨归一,防止 concat 出坏文件)
    from maestro.pipeline.audio_stage import any_audio, normalize_for_concat
    from maestro.tools.video_concat import VideoConcatTool
    clips = [p for p in segs if p]
    concat_in = (normalize_for_concat(clips, out_dir / "concat_norm")
                 if any_audio(clips) else clips)
    final = VideoConcatTool().run(concat_in, out_dir / "lighthouse_60s.mp4")
    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1))
    print("成片:", final)


if __name__ == "__main__":
    main()
