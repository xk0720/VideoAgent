# 2026-08-19 用户令【训练=生产完全同构 + rl/ 自包含】:本文件为
# src/maestro/agents/junction_stitcher.py 的逐字拷贝,仅 import 行改指 rl/env 内部 shim。
# 改生产原件必须同步改这里(tests/unit/test_rl_env_parity.py 锁差异)。
"""JunctionStitcherAgent — 缝合师(2026-08-09 用户令,默认启用)。

职责:为交界派生(transition video)写两镜描述。分工哲学:agent 出
品味(第一镜以实拍片尾为准;第二镜提炼"切后第一眼"),机器保契约
(ViMax 骨架/槽位语义行/refs 装配顺序全部由执行器锁死,agent 只产
两段散文)。

纪律(与全家 brain 决策一致):
- 技能全文进 prompt;STRICT JSON 输出 {"first_shot_desc",
  "second_shot_desc"};
- 四级校验(格式/记号契约/内容纪律/语言),判死带因重问一轮,仍坏
  返回 None —— 调用方退化到确定性模板装配(台账留痕),缝合层永远
  不把管线弄坏;
- 每次调用 brain_log 落盘。
"""
from __future__ import annotations

import json
import re
from typing import Optional

from env.logging_utils import brain_log, get_logger
from env.skills import extract_json as _extract_json

log = get_logger(__name__)

_SKILL_CACHE: dict = {}


def _skill_body() -> str:
    if "junction_stitch" not in _SKILL_CACHE:
        try:
            from ..skills.loader import load_skill

            sk = load_skill("junction_stitch")
            _SKILL_CACHE["junction_stitch"] = (
                sk["body"] if sk and sk["body"].strip() else "")
        except Exception:
            _SKILL_CACHE["junction_stitch"] = ""
        if _SKILL_CACHE["junction_stitch"]:
            log.info("brain skill LOADED: junction_stitch (%d chars)",
                     len(_SKILL_CACHE["junction_stitch"]))
    return _SKILL_CACHE["junction_stitch"]


class JunctionStitcherAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def run(self, *, prev_end_state: str, tail_report,
            cur_opening: str, slot_table: list,
            prompt_language: str = "zh") -> Optional[dict]:
        """→ {"first_shot_desc", "second_shot_desc"} | None(退化模板)。

        slot_table 行:{"slot": "<<<image_N>>>", "content": 语义,
        "kind": "tail_frame"|"portrait"|"location"}。"""
        if self.llm is None:
            return None
        from env.window_core import (_is_mostly_chinese,
                                            _scene_text_for_prompt)
        task = {
            "prompt_language": prompt_language,
            "prev_end_state_script": prev_end_state,
            "prev_tail_report": tail_report,
            "current_shot_opening_script": cur_opening,
            "slot_table": slot_table,
        }
        prompt = (_skill_body()
                  + "\n\n## Task\n"
                  + json.dumps(task, ensure_ascii=False))
        valid_tokens = {r["slot"] for r in slot_table}
        cast_names = [r.get("name") for r in slot_table if r.get("name")]
        err: Optional[str] = None
        for _attempt in range(2):
            if err:
                prompt += (f"\n\nYOUR PREVIOUS REPLY WAS INVALID: {err}. "
                           "Output the corrected STRICT JSON.")
            raw = ""
            try:
                raw = self.llm.complete(prompt)
                data = _extract_json(raw) or {}
            except Exception as exc:
                data = {}
                err = f"LLM call failed: {str(exc)[:160]}"
                brain_log("window/junction_stitch", {
                    "raw": raw, "parsed": None, "usable": False,
                    "error": err, "context": task})
                continue
            first = _scene_text_for_prompt(
                str(data.get("first_shot_desc") or ""))
            second = _scene_text_for_prompt(
                str(data.get("second_shot_desc") or ""))
            err = self._validate(first, second, valid_tokens, cast_names,
                                 prompt_language)
            brain_log("window/junction_stitch", {
                "raw": raw,
                "parsed": ({"first_shot_desc": first,
                            "second_shot_desc": second}
                           if err is None else None),
                "usable": err is None, "error": err, "context": task})
            if err is None:
                return {"first_shot_desc": first,
                        "second_shot_desc": second}
            log.warning("junction stitch reply unusable (%s) — %s", err,
                        "corrective retry" if _attempt == 0
                        else "falling back to template assembly")
        return None

    @staticmethod
    def _validate(first: str, second: str, valid_tokens: set,
                  cast_names: list, prompt_language: str) -> Optional[str]:
        from env.window_core import _is_mostly_chinese
        if not first or not second:
            return "empty first_shot_desc or second_shot_desc"
        used = set(re.findall(r"<<<image_\d+>>>", first + second))
        unknown = sorted(used - valid_tokens)
        if unknown:
            return f"tokens not in the slot table: {unknown}"
        bare = [n for n in cast_names
                if n and (n in first or n in second)]
        if bare:
            return f"bare character names (use tokens): {bare}"
        if prompt_language == "zh" \
                and not _is_mostly_chinese(first + second):
            return "descriptions must be in Chinese on a zh project"
        return None
