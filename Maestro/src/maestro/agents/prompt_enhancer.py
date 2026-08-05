"""PromptEnhancerAgent — optional per-shot video-prompt polisher (2026-07-15).

用户需求:可选(超参数开关)的 prompt 润色 agent。输入 = shot 的文字描述 +
全部条件的文字描述(每张图的角色/描述、参考视频),它先想清楚"这个 shot
该怎么利用这些条件",再按自己的技能(官方 prompt guide 蒸馏成
skills/brain_skills/prompt_enhancer/SKILL.md)把最终的 video prompt 写好。

纪律(和其他 brain 决策一致):
- 技能全文进 prompt,模型输入输出一律英文;
- STRICT JSON 输出 {"video_prompt": ...},解析/校验失败返回 None ——
  调用方保留原 prompt,增强层永远不会把管线弄坏(开关关闭 = 行为零变化);
- 每次调用的原始输出经 brain_log 落 brain_calls.jsonl(debug 可核对)。
"""
from __future__ import annotations

import json
from typing import Optional

from ..logging_utils import brain_log, get_logger
from ..models.mllm_backends import _extract_json
from .base import BaseAgent

log = get_logger(__name__)

# 策略 → 模型家族(docs/CONDITION_MODEL_MAP.md §1 的只读投影)。
# 家族决定参考语法:seedance 用 @ImageN/@VideoN,kling 用 "reference image N"。
_STRATEGY_FAMILY = {
    "t2v": "seedance_t2v",
    "t2v_own_refs": "seedance_t2v",
    "ti2v_prev_plus_keyframe": "seedance_t2v",
    "tiv2v_window": "seedance_t2v",
    "extend_prev": "seedance_extend",
    "i2v_keyframe": "seedance_i2v",
    "ti2v_prev_last": "seedance_i2v",
    "flf2v_own_pair": "seedance_i2v_flf",
    "flf2v_bridge": "seedance_i2v_flf",
    "multi_image_fusion": "kling_reference",
}


class PromptEnhancerAgent(BaseAgent):
    def __init__(self, *args, llm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm

    _SKILL_CACHE: dict = {}

    @classmethod
    def _skill_body(cls) -> str:
        if "prompt_enhancer" not in cls._SKILL_CACHE:
            try:
                from ..skills.loader import load_skill

                sk = load_skill("prompt_enhancer")
                cls._SKILL_CACHE["prompt_enhancer"] = (
                    sk["body"] if sk and sk["body"].strip() else "")
            except Exception:
                cls._SKILL_CACHE["prompt_enhancer"] = ""
            body = cls._SKILL_CACHE["prompt_enhancer"]
            if body:
                log.info("brain skill LOADED: prompt_enhancer (%d chars)",
                         len(body))
            else:
                log.warning("brain skill MISSING/EMPTY: prompt_enhancer — "
                            "enhancement prompts carry NO technique text")
        return cls._SKILL_CACHE["prompt_enhancer"]

    def run(
        self,
        shot_description: str,
        strategy: str,
        conditions: list[dict],
        base_prompt: str = "",
        label: str = "",
    ) -> Optional[str]:
        """润色一条 video prompt。None = 增强不可用/失败(调用方保留原文)。

        `conditions` 逐条 {kind: image|video, role, description} —— 执行器从
        台账/窗口状态收集的【条件事实】,增强器只能利用、不能发明。"""
        if self.llm is None:
            return None
        family = _STRATEGY_FAMILY.get(strategy, "seedance_t2v")
        skill_text = self._skill_body()
        proof = {"skill": "prompt_enhancer", "skill_chars": len(skill_text),
                 "skill_loaded": bool(skill_text),
                 # 裁决 1.3:输入可审计(shot 描述 + 条件事实 + 原 prompt)
                 "context": {"shot_description": shot_description,
                             "strategy": strategy, "model_family": family,
                             "conditions": conditions,
                             "current_prompt": base_prompt or shot_description}}
        prompt = (
            skill_text
            + "\n\nTHIS TURN (JSON):\n"
            + json.dumps({
                "shot_description": shot_description,
                "strategy": strategy,
                "model_family": family,
                "conditions": conditions,
                "current_prompt": base_prompt or shot_description,
            }, ensure_ascii=False)
            + ('\n\nSTRICT JSON only: {"video_prompt": "<润色后的最终 '
               'prompt,必须全中文(记号 <<<image_N>>> 原样保留,台词逐字'
               '照抄),30-100 词>"}'
               if any(c.get("role") == "prompt_language"
                      and c.get("description") == "zh"
                      for c in conditions) else
               '\n\nSTRICT JSON only: {"video_prompt": "<the final polished '
               'prompt, English, 30-100 words>"}')
        )
        # 方案 A(2026-07-16):可引用槽位来自 conditions(执行器的槽位
        # 清单投影)——输出里的编号必须 ⊆ 它;引用了不存在的编号,带着
        # 错误反馈重试一次;仍错 → None(调用方保留原 prompt,主循环的
        # 出口闸再兜一层)。
        slots = [{"slot": c.get("slot", ""), "content": c.get("description", ""),
                  "referenceable": bool(c.get("referenceable"))}
                 for c in conditions if c.get("slot")]
        attempt_prompt = prompt
        for attempt in range(2):
            raw = ""
            try:
                raw = self.llm.complete(attempt_prompt)
                data = _extract_json(raw)
            except Exception as exc:
                brain_log("window/prompt_enhance", {
                    "label": label, "strategy": strategy, "family": family,
                    "raw": raw or f"<complete() raised: {exc}>",
                    "parsed": None, "usable": False, **proof})
                return None
            out = (data or {}).get("video_prompt") \
                if isinstance(data, dict) else None
            usable = isinstance(out, str) and 10 <= len(out.strip()) <= 2000
            audit = {"ok": True, "unknown": [], "appended": []}
            if usable:
                from ..pipeline.ref_slots import validate_references

                fixed, audit = validate_references(out.strip(), slots)
                usable = audit["ok"]
                if usable:
                    out = fixed
            brain_log("window/prompt_enhance", {
                "label": label, "strategy": strategy, "family": family,
                "raw": raw,
                "parsed": ({"video_prompt": out} if usable else None),
                "usable": bool(usable), "ref_audit": audit,
                "attempt": attempt, **proof})
            if usable:
                self._log("enhance", {"label": label, "strategy": strategy},
                          {"chars": len(out.strip()), "attempt": attempt})
                return out.strip()
            if audit["unknown"] and attempt == 0:
                # 重试一次:告诉它错在哪、只许用哪些编号
                attempt_prompt = (
                    prompt
                    + "\n\nYOUR PREVIOUS ANSWER WAS REJECTED: it referenced "
                    + ", ".join(audit["unknown"])
                    + ", which the executor will NOT assemble. Use ONLY these "
                      "reference IDs, exactly as written: "
                    + (", ".join(r["slot"] for r in slots
                                 if r["referenceable"]) or "(none — no "
                       "reference syntax on this route)")
                    + ". Rewrite the prompt.")
                continue
            return None
        return None
