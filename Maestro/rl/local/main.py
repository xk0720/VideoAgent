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
    ap.add_argument("--role", choices=("trainer", "stream", "rollback"),
                    required=True)
    ap.add_argument("--to", default=None,
                    help="rollback 目标:版本号,或 best(全程 reward "
                         "最优版);历史见 rl/state/live_adapter/"
                         "reward_history.jsonl")
    C.add_args(ap)
    a = ap.parse_args()
    hp = C.from_args(a)

    if a.role == "rollback":
        # 峰值回滚:目标版本复制为 v_max+1 后原子换入。必须先
        # `run_local.sh --stop` —— 训练器还活着会立刻发布更高版盖掉它。
        from rl.local.broadcast import rollback_adapter
        if a.to is None:
            print("缺 --to <版本号|best>;峰值查 rl/state/live_adapter/"
                  "reward_history.jsonl", flush=True)
            return 2
        v = rollback_adapter(a.to if a.to == "best" else int(a.to))
        print(f"已把 v{a.to} 重新发布为 v{v};流将在下一个镜间安全点"
              f"换用。训练器如需继续,请以 --kl-coef 调整后重启。",
              flush=True)
        return 0

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
