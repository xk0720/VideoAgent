"""RL 内部的策略 prompt 事实源(2026-08-19 用户令:rl/ 不调用文件夹
以外的包 —— 训练所需一切自包含)。

本模块是 rl/ 里唯一定义"发给策略模型的 prompt 怎么拼"的地方:
  • env 采样时用它拼 prompt;
  • 训练器重建 prompt 时用它 —— 同一份代码,训练分布 = 采样分布。

技能【文本】按数据文件从 src/maestro/skills/brain_skills/<name>/SKILL.md
原位读取(读文件不是调包):技能全文是策略的输入分布,与生产共用同
一份文件,谁也不会漂移。prompt 组装【代码】= window_loop.decision_prompt
的逐字拷贝(2026-08-19 起 rl/ 不再 import 它,以此文件为准)。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BRAIN_SKILLS = REPO / "src/maestro/skills/brain_skills"

_CACHE: dict[str, str] = {}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """(frontmatter dict, body);无 frontmatter → ({}, 全文)。"""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm: dict = {}
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip()
            return fm, text[end + 4:].lstrip("\n")
    return {}, text


def skill_body(name: str) -> str:
    """技能全文(缓存;缺文件返回 "" —— 调用方响亮记录后降级)。"""
    if name not in _CACHE:
        p = BRAIN_SKILLS / name / "SKILL.md"
        try:
            _fm, body = parse_frontmatter(p.read_text(encoding="utf-8"))
            _CACHE[name] = body if body.strip() else ""
        except Exception:
            _CACHE[name] = ""
    return _CACHE[name]


def decision_prompt(skill_text: str, menu: list, context: dict) -> str:
    """窗口决策喂给 brain 的完整 prompt(与生产 window_loop 同款拼法,
    逐字拷贝 —— 改这里必须同步改那边,反之亦然)。"""
    return (
        skill_text
        + "\n\nTHIS TURN (JSON):\n"
        + json.dumps({"menu": menu, **context}, ensure_ascii=False)
        + '\n\nSTRICT JSON only: {"strategy": "<name from menu>", '
          '"reason": "<one short sentence>", ... optional semantic fields '
          "per the skill above (images / video_prompt / use_prev_tail_video)}"
        + (" reason AND video_prompt in CHINESE (excerpt the screenplay's"
           " wording; tokens inline in Chinese sentences); ONLY image"
           " descriptions stay ENGLISH (English-biased image models)."
           if context.get("prompt_language") == "zh" else
           " ALL output text (reason / video_prompt / image descriptions)"
           " must be in ENGLISH, regardless of the user's language.")
    )


def extract_json(text):
    """容错 JSON 抽取:剥 ```json 围栏/前后闲话,取第一个对象或数组;
    解析不动返回 None(调用方走兜底,永不 crash)。"""
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i = text.find(open_c)
        j = text.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    return None
