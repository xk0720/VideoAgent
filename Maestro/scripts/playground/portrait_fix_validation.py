#!/usr/bin/env python3
"""真 API 验证(2026-07-31 双大 bug 修复 + 裁决 1):

① 修好的肖像链:英文契约 + 全角兼容拆分 + 背景=影片场景 —— 用真
   t2i 重生成面包师/顾客两张官方肖像,肉眼核验背景是雨后清晨的
   街角面包店(不再是影棚白布);
② ViMax 肖像替换原语:拿 movie_20260731_144652 真实的 shot1 关键帧,
   用 seedream-v4/edit(多图:关键帧+新肖像)执行"换人保景",核验
   编辑端点的多图参考服从度 —— 这是 repair_keyframe_identity 的承重墙。

政策:真 API(不打桩)、输出进 outputs/ 时间戳目录、密钥来自 .env。
花费:t2i ×2 + image-edit ×1 ≈ $0.1 以内。
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import yaml  # noqa: E402

from maestro.memory.storyboard import StoryboardMemory  # noqa: E402
from maestro.models import build_video_gen  # noqa: E402
from maestro.models.image_edit import WaveSpeedImageEditClient  # noqa: E402
from maestro.pipeline import window_loop as wl  # noqa: E402
from maestro.types import AssetMemory  # noqa: E402

OLD_KEYFRAME = (REPO_ROOT / "outputs" / "movie_20260731_144652" /
                "keyframes" / "shot000_kf0_t2i.png")

# 英文契约(修复后 scene_write 的 LANGUAGE LAW 输出形态;名字保留中文)
CAST = {
    "年轻面包师": ("static: slender young man, short black hair, "
                   "rolled-sleeve white shirt, white apron, dark gray "
                   "trousers, brown leather shoes; dynamic: expression, "
                   "pose, tray or paper in hand"),
    "撑伞的顾客": ("static: middle-aged woman, shoulder-length brown "
                   "hair, dark green raincoat, black trousers, rain "
                   "boots, long dark-blue umbrella; dynamic: pose, "
                   "expression, umbrella open or closed"),
}
SETTING = ("a street-corner bakery interior on a rainy morning — black "
           "oven, wooden worktable, glass display counter, warm yellow "
           "shop lights mixing with wet morning light")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portrait", default="",
                    help="复用已生成的肖像(跳过 t2i,省钱重测编辑)")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / "outputs" /
               f"portrait_fix_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    if args.portrait:
        baker = args.portrait
        print(f"复用肖像: {baker}")
    else:
        models_cfg = yaml.safe_load(
            (REPO_ROOT / "configs" / "basic.yaml").read_text())["models"]
        vg_cfg = dict(models_cfg.get("video_gen") or {})
        vg_cfg.setdefault("call_log", str(out_dir / "wavespeed_calls.jsonl"))
        video_gen = build_video_gen(vg_cfg)

        # ① 修好的肖像链(真 t2i;走管线同一函数,不是复刻逻辑)
        sb = StoryboardMemory.from_outline(
            ["shot 1: placeholder"], path=out_dir / "storyboard.json")
        sb.cast = dict(CAST)
        sb.setting = SETTING
        notes = wl._ensure_cast_portraits(sb, AssetMemory(), video_gen,
                                          out_dir)
        for n in notes:
            print(f"portrait: {n.get('name')} via={n.get('via')} "
                  f"path={n.get('path', '-')}")
        baker = sb.portraits.get("年轻面包师")
        if not baker:
            print("❌ 面包师肖像生成失败 — 中止")
            return 1

    # ② ViMax 肖像替换原语(真 seedream-v4/edit,多图:关键帧+肖像)。
    # 指令 = 修复 handler 的同一单一事实源;size 由客户端按关键帧比例推导。
    from maestro.agents.orchestrator import identity_repair_instruction

    if not OLD_KEYFRAME.exists():
        print(f"⚠ 旧关键帧不存在({OLD_KEYFRAME})— 跳过编辑验证")
        return 0
    editor = WaveSpeedImageEditClient()
    edited = editor.edit(OLD_KEYFRAME,
                         identity_repair_instruction("the young baker"),
                         out_dir / "keyframe_identity_repaired.png",
                         references=[Path(baker)])
    print(f"edited keyframe: {edited}")
    print("完成 — 请肉眼核验:肖像背景应为面包店场景;编辑结果应换人保景。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
