#!/usr/bin/env python3
"""单跑 beat_pose_reel 开场段(真机, wan-pro x13): 用户先看效果, 再决定上全片。"""
import json
import logging
import os
import sys
from pathlib import Path

VS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VS))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
from dotenv import load_dotenv
load_dotenv(VS / ".env")

import yaml

from studio.skill_store import SkillStore
from studio.render import Renderer
from studio.executor import Executor

brief = yaml.safe_load((VS / "examples/product_yike.yaml").read_text(encoding="utf-8"))
card = SkillStore().get("beat_pose_reel")
hooks = [str(VS / "examples/new/yike/person_hook1.jpg")]
pipe = Renderer(card, hooks, person_count=1, t0=0.0, t1=8.566,
                ref_frames=brief["ref_frames"]).pipeline(
    {"spec_line": "白/蓝 两色可选"})
out = VS / "outputs/yike_opening"
ex = Executor(out, dashscope_key=os.environ.get("DASHSCOPE_API_KEY", ""),
              wavespeed_key=os.environ.get("WAVESPEED_API_KEY", ""),
              kling_mode="pro", resume=True)
ex._sigs_all = {}
seg = {"seg_id": "seg01", "skill_id": "beat_pose_reel",
       "t0": 0.0, "t1": 8.566, "calls": pipe}
res = ex.run_segment(seg)
ex._sig_file.write_text(json.dumps(ex._sigs_all, indent=1), encoding="utf-8")
(out / "execution.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
fails = [c["id"] for c in res["calls"] if not c.get("ok")]
print(f"\n★ 开场段: ok={res['ok']} 产物={res.get('output')}")
print(f"  失败步骤: {fails or '无'}")
