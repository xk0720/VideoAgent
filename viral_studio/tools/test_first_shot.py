#!/usr/bin/env python3
"""单测首镜 s01(用户更新参考图后的验证): 产物/签名按正式跑落位, 全量跑可复用。"""
import json
import logging
import os
import sys
from pathlib import Path

VS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VS))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
from dotenv import load_dotenv
load_dotenv(VS / ".env")
import yaml

from studio.skill_store import SkillStore
from studio.render import Renderer
from studio.executor import Executor

brief = yaml.safe_load((VS / "examples/product_yike.yaml").read_text(encoding="utf-8"))
card = SkillStore().get("beat_pose_reel")
pipe = Renderer(card, [str(VS / "examples/new/yike/person_hook1.jpg")], person_count=1,
                t0=0.0, t1=8.566, ref_frames=brief["ref_frames"]).pipeline(
    {"spec_line": "白/蓝 两色可选"})
out = VS / "outputs/yike_opening"
ex = Executor(out, dashscope_key=os.environ["DASHSCOPE_API_KEY"], kling_mode="pro")
ex._sigs_now = {}
sigs = {}
for call in pipe:
    sig = ex._signature(call)
    ex._sigs_now[call["id"]] = sig
    sigs[call["id"]] = sig

s01 = [c for c in pipe if c["id"] == "s01"][0]
print(f"s01: ref={Path(s01['params']['ref']).name} driving={Path(s01['params']['driving']).name}")
rec = ex._invoke_with_retry("seg01", s01, s01["params"])
print("✅" if rec.get("ok") else "❌", rec.get("path", ""), str(rec.get("error", ""))[:200])
if rec.get("ok"):
    try:
        prev = json.loads((out / "_resume.json").read_text())
    except Exception:
        prev = {}
    prev["seg01.s01"] = sigs["s01"]
    (out / "_resume.json").write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print("签名已记, 全量跑将复用该镜")
