#!/usr/bin/env python3
"""原样导出 Planner 两阶段的真实输入(system+user)与输出, 供人工审 prompt。
不调用 LLM(除非 --live)。用法: python tools/dump_planner_io.py examples/xxx.yaml
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.skill_store import SkillStore                      # noqa: E402
from studio.agents.storyboard_planner import (SELECT_PROMPT,   # noqa: E402
                                              FILL_PROMPT, StoryboardPlanner)

brief = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
n = len(brief.get("person_hooks", [])) or 1
cat = brief.get("category", "服装")
store = SkillStore()

bar = lambda t: print(f"\n{'█'*3} {t} {'█'*(70-len(t))}")

bar("阶段① SYSTEM (storyboard_select.md)")
print(SELECT_PROMPT)

bar("阶段① USER (真实发送)")
print(f"## 商品\n名称: {brief.get('name')}\n类目: {cat}\n"
      f"描述: {brief.get('description','')}\n"
      f"卖点: {'; '.join(brief.get('selling_points', []))}\n"
      f"人物参考图: {n} 张\n"
      f"目标总时长: {brief.get('duration_target_s','不限')} 秒\n\n"
      f"## 候选 skill(已按类目与人数筛过)\n{store.digest(cat, n, brief=True)}\n\n"
      f"## {store.rules}\n\n请输出选卡结果 JSON。")

# 阶段②以 outdoor_narration 段为例
card = store.get("outdoor_narration")
seg = {"seg_id": "seg02", "part": "body", "skill_id": "outdoor_narration", "hook_index": 1}
p = StoryboardPlanner(store)
auto = p._auto_slots(card, seg, brief)
spec = {k: v for k, v in (card.get("slots") or {}).items() if k not in auto}
skeleton = {k: f"<{v.get('lang','')}" + (f" {v['min_chars']}-{v['max_chars']}字"
            if v.get("min_chars") else "") + ">" for k, v in spec.items()}
hints = "\n".join(f"  · {k}: {v.get('desc','')}"
                  + (f" [{v['min_chars']}-{v['max_chars']}字]" if v.get("min_chars") else "")
                  for k, v in spec.items())
extras = {k: card[k] for k in ("action_library", "scene_by_color", "scene_default",
                               "background") if k in card}
colors = brief.get("hook_colors") or []

bar("阶段② SYSTEM (storyboard_fill.md)")
print(FILL_PROMPT)

bar("阶段② USER (真实发送 · 以 seg02 为例)")
print(f"## 商品\n{brief.get('name')} — {brief.get('description','')}\n"
      f"卖点: {'; '.join(brief.get('selling_points', []))}\n\n"
      f"## 本段\nskill: outdoor_narration「{card.get('name','')}」\n"
      f"段落位置: body  第 1 张人物图(配色 {colors[0] if colors else '?'})\n\n"
      f"## 只输出这个 JSON 对象(键固定, 值替换为你填的内容)\n"
      f"{json.dumps(skeleton, ensure_ascii=False, indent=2)}\n\n"
      f"## 各字段要求\n{hints}\n\n"
      f"## 该 skill 卡提供的素材(优先取用)\n"
      f"{json.dumps(extras, ensure_ascii=False, indent=2)[:1800]}\n\n"
      f"## 前面段落已经说过的内容(不要重复, 要递进)\n- (第一段, 无)\n\n"
      f"## {store.rules}")

bar("程序注入(不问模型)的 slots")
print(json.dumps(auto, ensure_ascii=False, indent=2))
