#!/usr/bin/env python3
"""hook_remake 测试链路入口。

用法:
  python run_test.py --video 爆款.mp4 --hooks hooks.json --dry-run   # 只切镜+排产, 不花钱
  python run_test.py --video 爆款.mp4 --hooks hooks.json             # 默认只生成前 3 镜(控费)
  python run_test.py --video 爆款.mp4 --hooks hooks.json --limit 0   # 全量生成

hooks.json 就是需求里的字典原样:
  {"person_hook_1": "<url|本地路径>", "object_hook_1": "<url>", ...}
(object_hook_* 本版忽略; person_hook 建议用与原片同比例的全身照。)

前置: ffmpeg/ffprobe 在 PATH; 环境变量 DASHSCOPE_API_KEY(或仓库根 .env)。
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import run  # noqa: E402


def _load_dotenv() -> None:
    """极简 .env 读取(只为 DASHSCOPE_API_KEY, 不引第三方依赖):
    依次尝试 hook_remake/.env 和仓库根 .env, 不覆盖已有环境变量。"""
    for env in (HERE / ".env", HERE.parent / ".env"):
        if not env.exists():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser(description="爆款视频复刻 - 极简测试链路")
    ap.add_argument("--video", required=True, help="爆款原片(本地路径)")
    ap.add_argument("--hooks", required=True, help="hooks.json 路径")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--limit", type=int, default=None,
                    help="只生成前 N 个镜头, 0=全量(默认取 config)")
    ap.add_argument("--assign", choices=["sequential", "round_robin"], default=None)
    ap.add_argument("--mode", choices=["wan-std", "wan-pro"], default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="只切镜+分配+排产落台账, 不调用付费 API")
    ap.add_argument("--yes", action="store_true", help="跳过费用确认")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    _load_dotenv()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for k in ("limit", "assign", "mode", "workers"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    cfg["api_key"] = os.environ.get("DASHSCOPE_API_KEY", "")

    hooks = json.loads(Path(args.hooks).read_text(encoding="utf-8"))
    if not isinstance(hooks, dict):
        raise SystemExit("hooks.json 应为 {\"person_hook_1\": \"...\"} 形式的字典")

    run(video=args.video, hooks=hooks, cfg=cfg, out_root=Path(args.out),
        dry_run=args.dry_run, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
