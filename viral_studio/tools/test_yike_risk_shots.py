#!/usr/bin/env python3
"""高危镜头预检(std 档小成本): s02/s03 怼脸近景(FullFace 风险), s05 背面(质量风险)。
FullFace/NoHuman 是确定性拒绝且与档位无关 —— std 测过 = pro 也测过。"""
import logging
import os
import sys
from pathlib import Path

VS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VS))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

from dotenv import load_dotenv
load_dotenv(VS / ".env")
from studio.backends.bailian_animate import BailianAnimateClient

ref = str(VS / "examples/new/yike/person_hook1.jpg")
out = VS / "outputs/yike_risk_test"
out.mkdir(parents=True, exist_ok=True)
c = BailianAnimateClient(api_key=os.environ["DASHSCOPE_API_KEY"], mode="wan-std")
for sid in ("s02", "s03", "s05"):
    drv = str(VS / f"memory/assets/yike_pose/shots/{sid}.mp4")
    ok, tid, err = c.animate(ref, drv, str(out / f"{sid}_std.mp4"))
    print(f"{'✅' if ok else '❌'} {sid}: task={tid} {err[:180]}")
