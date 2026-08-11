"""Reward 函数(step-level;设计见 rl/DESIGN.md)。

纯函数、零管线依赖:输入 = 一条 join 好的 step 记录(dict),输出 =
{"reward", "r_format", "r_task", "detail"}。format 判据刻意与管线闸门
同构(JSON/菜单/引用/语言/字段)—— 训练目标与生产闸门对齐,模型学会
的就是"过闸"的能力。
"""
from __future__ import annotations

import json
import re

W_FORMAT = 0.2
W_TASK = 0.8
ACCEPT_BAR = 0.75          # accept 决策的及格线(与管线 quality_bar 对齐)
VERIFIED_BONUS = 0.3
REPAIR_TURN_PENALTY = 0.05


def _extract_json(text: str):
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def format_reward(step: dict) -> tuple[float, dict]:
    """0..1:可解析 .4 + 工具/策略在菜单 .2 + 引用⊆清单 .2 + 语言 .1
    + 必填字段 .1。管线判 unusable → 直接 0(它已触发纠错重试)。"""
    detail: dict = {}
    if step.get("usable") is False:
        return 0.0, {"unusable": True}
    data = step.get("parsed")
    if not isinstance(data, dict):
        data = _extract_json(step.get("raw"))
    score = 0.0
    if isinstance(data, dict):
        score += 0.4
        detail["json"] = True
    else:
        return 0.0, {"json": False}

    menu = [m.get("name") for m in (step.get("menu") or [])
            if isinstance(m, dict)]
    picked = (data.get("strategy") or data.get("tool")
              or data.get("plan") or None)
    if not menu:
        score += 0.2                      # 无菜单类 step(缝合/挑图)满分
        detail["menu"] = "n/a"
    elif picked in menu:
        score += 0.2
        detail["menu"] = True
    else:
        detail["menu"] = False

    slots = {r.get("slot") for r in (step.get("slots") or [])
             if isinstance(r, dict)}
    used = set(re.findall(r"<<<image_\d+>>>|@Image\d+",
                          json.dumps(data, ensure_ascii=False)))
    if not used or not slots:
        score += 0.2
        detail["refs"] = "n/a"
    elif used <= slots:
        score += 0.2
        detail["refs"] = True
    else:
        detail["refs"] = sorted(used - slots)

    lang = step.get("prompt_language")
    text_out = str(data.get("video_prompt") or data.get("hint") or "")
    if lang != "zh" or not text_out:
        score += 0.1
        detail["lang"] = "n/a"
    else:
        han = len(re.findall(r"[一-鿿]", text_out))
        ok = han >= max(1, len(text_out) // 8)
        score += 0.1 if ok else 0.0
        detail["lang"] = ok

    required = {"generation-condition": ("strategy", "video_prompt"),
                "repair": ("tool",),
                "image-plan": ("plan",)}.get(step.get("kind"), ())
    if all(data.get(k) for k in required):
        score += 0.1
        detail["fields"] = True
    else:
        detail["fields"] = [k for k in required if not data.get(k)]
    return round(score, 4), detail


def task_reward(step: dict) -> tuple[float, dict]:
    """按 step 类型归因(0..~1.3,可为负):
    condition → weighted_total + 0.3·verifier/10
    repair    → Δ分;accept → ±(final−bar)
    image_plan→ 0.1·无降级 + 0.5·weighted_total
    附加:verified 加成、修复轮惩罚。"""
    kind = step.get("kind")
    o = step.get("outcome") or {}
    r = 0.0
    detail: dict = {"kind": kind}
    wt = o.get("weighted_total")
    vs = o.get("verifier_score")
    if kind == "generation-condition":
        r += float(wt or 0.0)
        if vs is not None:
            r += 0.3 * max(0.0, float(vs)) / 10.0
    elif kind == "repair":
        if o.get("tool") == "accept":
            final = float(wt or 0.0)
            r += final - ACCEPT_BAR
            detail["accept_margin"] = round(final - ACCEPT_BAR, 4)
        else:
            prev = o.get("prev_total")
            new = o.get("new_total")
            if prev is not None and new is not None:
                delta = max(-1.0, min(1.0, float(new) - float(prev)))
                r += delta
                detail["delta"] = round(delta, 4)
    elif kind == "image-plan":
        if not o.get("degraded_from"):
            r += 0.1
        r += 0.5 * float(wt or 0.0)
    else:                                  # junction 类:借整镜分的一半
        r += 0.5 * float(wt or 0.0)
    if o.get("converged"):
        r += VERIFIED_BONUS
        detail["verified"] = True
    turns = o.get("repair_turns")
    if kind == "generation-condition" and turns:
        r -= REPAIR_TURN_PENALTY * int(turns)
        detail["turn_penalty"] = REPAIR_TURN_PENALTY * int(turns)
    return round(r, 4), detail


def step_reward(step: dict) -> dict:
    rf, df = format_reward(step)
    rt, dt = (task_reward(step) if rf > 0 else (0.0, {"gated": True}))
    return {"reward": round(W_FORMAT * rf + W_TASK * rt, 4),
            "r_format": rf, "r_task": rt,
            "detail": {"format": df, "task": dt}}
