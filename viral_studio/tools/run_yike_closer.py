#!/usr/bin/env python3
"""yike 第三段(收尾, 双人跳舞出场): 标题走真实 Planner LLM 链路。"""
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
from studio.agents.storyboard_planner import StoryboardPlanner

brief = yaml.safe_load((VS / "examples/product_yike.yaml").read_text(encoding="utf-8"))
store = SkillStore()
card = store.get("sequential_reveal")
_saved = VS / "outputs/yike_closer/texts.json"
if _saved.exists():          # 写稿只发生一次(Planner 阶段); 执行期重写会让语音与字幕分家
    texts = json.loads(_saved.read_text(encoding="utf-8"))
    print("使用存稿:", json.dumps(texts, ensure_ascii=False))
else:
    texts = StoryboardPlanner(store)._write(card, brief, 2)
print("── LLM 标题 ──", json.dumps(texts, ensure_ascii=False))

hooks = [str(VS / "examples/new/yike/person_hook1.jpg"),
         str(VS / "examples/new/yike/person_hook2.jpg")]
r = Renderer(card, hooks, person_count=2, t0=28.566, t1=32.566,
             ref_frames=brief.get("ref_frames"))
pipe = r.pipeline(texts, prompt=r.prompt(texts))
out = VS / "outputs/yike_closer"
out.mkdir(parents=True, exist_ok=True)
(out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
ex = Executor(out, dashscope_key=os.environ.get("DASHSCOPE_API_KEY", ""),
              wavespeed_key=os.environ.get("WAVESPEED_API_KEY", ""),
              kling_mode="pro", resume=True)
ex._sigs_all = {}
res = ex.run_segment({"seg_id": "seg03", "skill_id": "sequential_reveal",
                      "t0": 28.566, "t1": 32.566, "calls": pipe})
ex._sig_file.write_text(json.dumps(ex._sigs_all, indent=1), encoding="utf-8")
(out / "execution.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
print(f"\n★ 第三段: ok={res['ok']} 产物={res.get('output')}")
