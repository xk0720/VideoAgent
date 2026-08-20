"""分镜剧本 Planner Agent —— 两阶段: ① 选卡排期 ② 逐段填空。

为什么分两阶段(实测教训): 一次输出带嵌套 slots 的完整脚本, qwen-max 会把
slots 写成字符串或结构错乱, 连修三轮都回不来。拆开后每次输出都是扁平小对象:
阶段①只输出 skill_id 列表, 阶段②每段只输出该卡声明的几个键 —— 一次就对。

校验仍然**按卡进行**: slots 的键集合、语言、字数区间全部从 skill 卡读取,
加新 skill 不用改这里。
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from ..llm import chat_json
from ..render import Renderer
from ..skill_store import SkillStore
from ..storyboard import Issue, Report, SegmentSpec, Storyboard

log = logging.getLogger("viral_studio")
PROMPTS = Path(__file__).parents[1] / "prompts"
SELECT_PROMPT = (PROMPTS / "storyboard_select.md").read_text(encoding="utf-8")
FILL_PROMPT = (PROMPTS / "storyboard_fill.md").read_text(encoding="utf-8")
CJK = re.compile(r"[一-鿿]")
BANNED = re.compile(r"\b(360|full turn|spin|rotate quickly)\b", re.I)
MAX_FILL_RETRY = 2


class StoryboardPlanner:
    def __init__(self, store: SkillStore):
        self.store = store

    # ── 阶段①: 选卡排期 ──────────────────────────────────
    def _select(self, brief: dict, cat: str, n: int) -> List[dict]:
        user = (f"## 商品\n名称: {brief.get('name')}\n类目: {cat}\n"
                f"描述: {brief.get('description','')}\n"
                f"卖点: {'; '.join(brief.get('selling_points', []))}\n"
                f"人物参考图: {n} 张\n"
                f"目标总时长: {brief.get('duration_target_s','不限')} 秒\n\n"
                f"## 候选 skill(已按类目与人数筛过)\n"
                f"{self.store.digest(cat, n, brief=True)}\n\n"
                f"## {self.store.rules}\n\n请输出选卡结果 JSON。")
        raw = chat_json(SELECT_PROMPT, user, temperature=0.5)
        segs = raw.get("segments", [])
        log.info("阶段①选卡: %s", " → ".join(
            f"{s.get('part')}/{s.get('skill_id')}" for s in segs))
        return segs, raw.get("overall_reason", "")

    # ── 阶段②: 逐段填空 ──────────────────────────────────
    @staticmethod
    def _auto_slots(card: dict, seg: dict, brief: dict) -> Dict[str, str]:
        """确定性字段由程序注入, 不问模型 —— 实测模型会把配色填错位。"""
        out: Dict[str, str] = {}
        colors = brief.get("hook_colors") or []
        idx = (seg.get("hook_index") or 1) - 1
        color = colors[idx] if idx < len(colors) else None
        for k, v in (card.get("slots") or {}).items():
            src = v.get("auto_from")
            if not src:
                continue
            if src == "hook_color" and color:
                out[k] = color
            elif src == "scene_default":
                out[k] = str(card.get("scene_default", "")).strip()
            elif src == "scene_by_color" and color:
                out[k] = str((card.get("scene_by_color") or {}).get(color, "")).strip()
        return {k: v for k, v in out.items() if v}

    def _fill(self, seg: dict, card: dict, brief: dict, n: int,
              prior: List[str]) -> Tuple[Dict[str, str], str]:
        spec = card.get("slots") or {}
        if not spec:
            return {}, seg.get("reason", "")
        auto = self._auto_slots(card, seg, brief)
        spec = {k: v for k, v in spec.items() if k not in auto}   # 只问模型剩下的
        if not spec:
            return auto, seg.get("reason", "")
        skeleton = {k: f"<{v.get('lang','')}"
                       + (f" {v['min_chars']}-{v['max_chars']}字"
                          if v.get("min_chars") else "") + ">"
                    for k, v in spec.items()}
        hints = "\n".join(
            f"  · {k}: {v.get('desc','')}"
            + (f" [{v['min_chars']}-{v['max_chars']}字]" if v.get("min_chars") else "")
            for k, v in spec.items())
        extras = {k: card[k] for k in ("action_library", "scene_by_color",
                                       "scene_default", "background")
                  if k in card}
        colors = brief.get("hook_colors") or []
        idx = (seg.get("hook_index") or 1) - 1
        user = (f"## 商品\n{brief.get('name')} — {brief.get('description','')}\n"
                f"卖点: {'; '.join(brief.get('selling_points', []))}\n\n"
                f"## 本段\nskill: {seg['skill_id']}「{card.get('name','')}」\n"
                f"段落位置: {seg.get('part')}  第 {seg.get('hook_index')} 张人物图"
                + (f"(配色 {colors[idx]})" if idx < len(colors) else "") + "\n\n"
                f"## 只输出这个 JSON 对象(键固定, 值替换为你填的内容)\n"
                f"{json.dumps(skeleton, ensure_ascii=False, indent=2)}\n\n"
                f"## 各字段要求\n{hints}\n\n"
                f"## 该 skill 卡提供的素材(优先取用)\n"
                f"{json.dumps(extras, ensure_ascii=False, indent=2)[:1800]}\n\n"
                + (f"## 前面段落已经说过的内容(不要重复, 要递进)\n- "
                   + "\n- ".join(prior) + "\n\n" if prior else "")
                + f"## {self.store.rules}\n")
        last, prev_json, slots = "", "", {}
        for attempt in range(MAX_FILL_RETRY + 2):
            q = user if not last else (
                user + f"\n## 你上一版的输出\n{prev_json}\n"
                f"\n## 这一版必须修复的问题(其余字段保持不变)\n{last}\n"
                f"中文字数按汉字个数算, 逐字数清楚再输出。\n")
            raw = chat_json(FILL_PROMPT, q, temperature=0.6 if not last else 0.2)
            slots = {k: str(v) for k, v in raw.items() if k in spec}
            issues = self._check_slots(slots, spec)
            if not issues:
                return {**auto, **slots}, seg.get("reason", "")
            last = "; ".join(issues)
            prev_json = json.dumps(slots, ensure_ascii=False, indent=2)
            log.info("  %s 填空第%d次: %s", seg["seg_id"], attempt + 1, last[:110])
        return {**auto, **slots}, seg.get("reason", "")   # 交给总校验兜底报错

    @staticmethod
    def _check_slots(slots: dict, spec: dict) -> List[str]:
        out = []
        for k in spec:
            v = (slots.get(k) or "").strip()
            if not v:
                out.append(f"{k} 缺失或为空"); continue
            c, has_cjk = spec[k], bool(CJK.search(v))
            if c.get("lang") == "zh" and not has_cjk:
                out.append(f"{k} 应为中文")
            if c.get("lang") == "en" and has_cjk:
                out.append(f"{k} 应为英文")
            if c.get("lang") == "zh" and c.get("min_chars"):
                ln = len(CJK.findall(v)) + len(re.findall(r"[A-Za-z0-9]+", v))
                if not (c["min_chars"] <= ln <= c["max_chars"]):
                    out.append(f"{k} 现{ln}字, 需{c['min_chars']}-{c['max_chars']}字")
            if BANNED.search(v):
                out.append(f"{k} 含高危动作词(360/spin)")
        return out

    # ── 主流程 ───────────────────────────────────────────
    def plan(self, brief: dict, bgm_source=None) -> Tuple[Storyboard, Report]:
        cat = brief.get("category", "服装")
        n = len(brief.get("person_hooks", [])) or 1
        hooks = list(brief.get("person_hooks") or [])
        picked, overall = self._select(brief, cat, n)

        segs: List[SegmentSpec] = []
        prior: List[str] = []
        t = 0.0
        for i, seg in enumerate(picked, 1):
            seg.setdefault("seg_id", f"seg{i:02d}")
            card = self.store.get(seg.get("skill_id", ""))
            if not card:                                  # 选了不存在的卡 → 总校验会报
                segs.append(SegmentSpec(seg_id=seg["seg_id"], part=seg.get("part", "body"),
                                        skill_id=seg.get("skill_id", "?"),
                                        reason=seg.get("reason", "")))
                continue
            fills, reason = self._fill(seg, card, brief, n, prior)
            prior += [v for k, v in fills.items()
                      if (card.get("slots", {}).get(k, {}) or {}).get("lang") == "zh"]

            p = card.get("produces", {})
            variants = p.get("variants")
            variant = str(n) if variants else None
            dur = float(p.get("duration_s")
                        or (variants or {}).get(variant, {}).get("duration_s") or 0)
            tail = float(card.get("tail_s", 0.5)) if variants else 0.0

            # ★ 填空之后立刻渲染成完整 pipeline —— 分镜脚本即可执行形态
            r = Renderer(card, hooks, person_count=n,
                         hook_index=seg.get("hook_index"), bgm_source=bgm_source,
                         t0=round(t, 3), t1=round(t + dur, 3))
            pipeline = r.pipeline(fills)

            segs.append(SegmentSpec(
                seg_id=seg["seg_id"], part=seg.get("part", "body"),
                skill_id=seg["skill_id"], variant=variant,
                hook_index=seg.get("hook_index"), duration_s=dur,
                t0=round(t, 3), t1=round(t + dur, 3),
                pipeline=pipeline, fills=fills, reason=reason))
            t += dur + tail
            log.info("  %s [%s] %s → %d 步: %s", seg["seg_id"], seg.get("part"),
                     seg["skill_id"], len(pipeline),
                     " → ".join(c["tool"] for c in pipeline))

        sb = Storyboard(product_name=brief.get("name", ""), category=cat,
                        person_count=n, segments=segs, overall_reason=overall)
        return sb, self.validate(sb, brief)

    # ── 总校验(按卡) ─────────────────────────────────────
    def validate(self, sb: Storyboard, brief: dict) -> Report:
        errs: List[Issue] = []
        warns: List[Issue] = []
        n = len(brief.get("person_hooks", [])) or 1
        cat = brief.get("category", "服装")
        seen, body_skills = set(), set()
        rank = {"opening": 0, "body": 1, "ending": 2}
        last = -1

        for seg in sb.segments:
            sid = seg.seg_id
            if sid in seen:
                errs.append(Issue(seg_id=sid, field="seg_id", msg="重复"))
            seen.add(sid)
            if rank[seg.part] < last:
                errs.append(Issue(seg_id=sid, field="part", msg="段落顺序错乱"))
            last = max(last, rank[seg.part])

            card = self.store.get(seg.skill_id)
            if not card:
                errs.append(Issue(seg_id=sid, field="skill_id",
                                  msg=f"'{seg.skill_id}' 不在 skill 库")); continue
            if card not in self.store.candidates(cat, n, seg.part):
                errs.append(Issue(seg_id=sid, field="skill_id",
                                  msg=f"'{seg.skill_id}' 不在 {seg.part} 段候选内")); continue
            if seg.part == "body":
                body_skills.add(seg.skill_id)
            if seg.hook_index is not None and not (1 <= seg.hook_index <= n):
                errs.append(Issue(seg_id=sid, field="hook_index", msg=f"越界(1..{n})"))

            spec = card.get("slots") or {}
            for k in sorted(set(seg.fills) - set(spec)):
                errs.append(Issue(seg_id=sid, field=f"slots.{k}", msg="该 skill 没有这个 slot"))
            for k in sorted(set(spec) - set(seg.fills)):
                errs.append(Issue(seg_id=sid, field=f"slots.{k}", msg="缺少必填 slot"))
            for m in self._check_slots(seg.fills, {k: v for k, v in spec.items()
                                                   if k in seg.fills}):
                errs.append(Issue(seg_id=sid, field="slots", msg=m))

        if len(body_skills) > 1:
            errs.append(Issue(field="body", msg=f"body 混用了 {body_skills}, 风格须统一"))
        if not any(s.part == "body" for s in sb.segments):
            errs.append(Issue(field="segments", msg="缺少 body 段"))
        if n >= 2 and not any(s.part == "ending" for s in sb.segments):
            warns.append(Issue(field="segments", msg=f"{n} 张人物图但无收尾段"))
        if n < 2 and any(s.part == "ending" for s in sb.segments):
            errs.append(Issue(field="segments", msg="人物图<2, 收尾段必须省略"))
        return Report(ok=not errs, errors=errs, warnings=warns)
