#!/usr/bin/env python
"""vimax_benchmark 英译中(2026-08-13 用户令):EN 版当题库,ZH 版当
用户剧本喂框架。schema 原样保留,只翻三类文本字段:
story_overview / 每镜 first_frame / 每镜 video_prompt。

纪律:
  • 一致性关键描述(人物外观逐项、场景几何逐件)【直译不省略】——
    它们就是考题本身,漏一件框架就少考一件;
  • 断点续跑:已存在且非空的输出文件跳过(逐故事幂等);
  • 每镜独立翻译调用,失败重试 2 次,仍失败则该故事标记 FAILED 并
    继续(收尾汇总,绝不静默吞)。

用法(模型走 configs 同款 llm 配置;默认 bailian.yaml 的 brain 底座):
  python scripts/translate_benchmark.py                # 全量 35 个
  python scripts/translate_benchmark.py --only chef_international_kitchens_typeA
  python scripts/translate_benchmark.py --config rl/configs/server_bailian_qwen.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from maestro.config import load_dotenv, load_yaml      # noqa: E402

load_dotenv(REPO / ".env")

from maestro.models import build_llm                   # noqa: E402

_INSTR = (
    "把下面这段英文影视分镜文本翻译成中文。铁律:\n"
    "1. 忠实直译,不概括不省略不发挥——人物外观(肤色/发型/伤疤/眼镜/"
    "每件衣物及颜色)和场景几何(每件固定陈设及方位)必须逐项译全,"
    "这些是一致性考题本身;\n"
    "2. 影视术语用行业惯用语(wide shot=远景/全景,eye-level=平视,"
    "medium close-up=中近景,pan=摇镜,dolly=推轨);\n"
    "3. 只输出译文,不加任何前后缀、引号或解释。\n\n原文:\n")


def _translate(llm, text: str) -> str:
    for attempt in range(3):
        try:
            out = (llm.complete(_INSTR + text) or "").strip()
            # 拒收明显失败:空/太短/仍是纯英文
            if out and len(out) >= max(10, len(text) // 8) \
                    and any("一" <= c <= "鿿" for c in out):
                return out
        except Exception as exc:
            print(f"    translate retry {attempt + 1}: {exc}",
                  flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("translation failed after 3 attempts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "vimax_benchmark"))
    ap.add_argument("--dst", default=str(REPO / "vimax_benchmark_zh"))
    ap.add_argument("--config",
                    default=str(REPO / "configs/bailian.yaml"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="只翻这些故事(不带 .json 的文件名)")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    llm = build_llm(cfg.get("models", {}).get("llm"))
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    idx = json.loads((src / "benchmark_index.json").read_text())
    (dst / "benchmark_index.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2))

    failed = []
    stories = [s["file"] for s in idx["stories"]]
    if args.only:
        stories = [f for f in stories
                   if f.replace(".json", "") in set(args.only)]
    for fname in stories:
        outp = dst / fname
        if outp.exists() and outp.stat().st_size > 200:
            print(f"skip (done): {fname}", flush=True)
            continue
        print(f"translating: {fname}", flush=True)
        d = json.loads((src / fname).read_text())
        try:
            d["story_overview"] = _translate(llm, d["story_overview"])
            for sc in d.get("scenes", []):
                for sh in sc.get("shots", []):
                    sh["first_frame"] = _translate(llm,
                                                   sh["first_frame"])
                    sh["video_prompt"] = _translate(llm,
                                                    sh["video_prompt"])
                    print(f"  scene {sc['scene_num']} shot "
                          f"{sh['shot_id']} ok", flush=True)
        except RuntimeError as exc:
            print(f"  ❌ {fname}: {exc} — story SKIPPED", flush=True)
            failed.append(fname)
            continue
        outp.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    print(f"\nDONE. failed={len(failed)} {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
