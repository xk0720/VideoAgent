#!/usr/bin/env python
"""rl/local 的唯一 Python 入口(2026-08-21 用户裁决:一个运行文件)。

  python -m rl.local.main --role trainer          # 训练器(一张卡)
  python -m rl.local.main --role stream --worker i --workers N

平时不用直接敲它 —— `bash rl/local/run_local.sh` 会把训练器和 N 条流
一起拉起来。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "rl"))
sys.path.insert(0, str(REPO))

from env.config import load_dotenv                              # noqa: E402

load_dotenv(REPO / ".env")

from rl.local import config as C                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("trainer", "stream"), required=True)
    C.add_args(ap)
    a = ap.parse_args()
    hp = C.from_args(a)

    if not hp.base_model:
        print("缺 --base-model(或 .env 里的 BASE_MODEL)", flush=True)
        return 2

    if a.role == "trainer":
        from rl.local.trainer import run_trainer
        return run_trainer(hp, wandb_on=a.wandb) or 0

    from rl.local.stream import run_stream
    return run_stream(hp, wandb_on=a.wandb) or 0


if __name__ == "__main__":
    raise SystemExit(main())
