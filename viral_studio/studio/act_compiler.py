"""Act 编译器 —— 分镜脚本 + skill 卡 → 精确的工具调用计划(call plan)。

为什么是编译器而不是 LLM: 解码之后已经没有不确定性了 —— prompt 正文由卡写死、
插槽由 Planner 填好、流水线由卡声明、素材路径由输入给定。此时再让 LLM "翻译"
一遍, 只会引入幻觉(错的工具名、漏的参数、编的路径)。这里全部确定性求值,
LLM 只在**执行失败**时才被唤起做决策(见 agents/act_agent.py)。

支持的占位符(取自三张卡的实际用法):
  $hook / $hook_N        本段(或指定序号)的人物参考图
  $background            背景图: 卡里给了现成图就用它, 否则插入一步 image_generation
  $bgm_slice             源 BGM 在本段时间轴上的切片(需要成片时间轴 → 见 timeline)
  $skill.a.b             skill 卡自身的字段
  $variant.x [+ n]       变体字段, 支持简单加法(如 duration_s + 2)
  @call_id               本段前面某次调用的产物 → executor 拓扑执行时替换成实际路径
  {slot}                 Planner 填的插槽值(prompt/文本类参数里)
"""
import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PROJECT_ROOT

log = logging.getLogger("viral_studio")

# 计费口径: 视频按输出秒, 音乐按生成秒, TTS 按字符, 图像按张
BILLED = {"seedance_t2v": "video_s", "animate_move": "video_s",
          "sonilo_text_to_music": "music_s", "minimax_tts": "tts_chars",
          "image_generation": "image_calls"}


class ActCompiler:
    def __init__(self, store, brief: dict, bgm_source: Optional[str] = None):
        self.store = store
        self.brief = brief
        self.hooks: List[str] = list(brief.get("person_hooks") or [])
        self.bgm_source = bgm_source          # 整条成片的源 BGM(可空)

    # ── 时间轴: 先算, 因为 $bgm_slice 依赖它 ──────────────
    def timeline(self, sb) -> List[dict]:
        t, out = 0.0, []
        for seg in sb.segments:
            card = self.store.get(seg.skill_id) or {}
            p = card.get("produces", {})
            dur = p.get("duration_s")
            if dur is None and seg.variant:
                dur = (p.get("variants", {}).get(seg.variant, {}) or {}).get("duration_s")
            dur = float(dur or 0)
            tail = float(card.get("tail_s", 0.5)) if p.get("variants") else 0.0
            out.append({"seg_id": seg.seg_id, "t0": round(t, 3),
                        "t1": round(t + dur, 3), "duration_s": dur,
                        "tail_s": tail})       # tail: 收尾段留给重击衰减的余量
            t += dur + tail
        return out

    # ── 求值器 ───────────────────────────────────────────
    def _resolve(self, val: Any, ctx: dict) -> Any:
        if isinstance(val, dict):
            return {k: self._resolve(v, ctx) for k, v in val.items()}
        if isinstance(val, list):
            return [self._resolve(v, ctx) for v in val]
        if not isinstance(val, str):
            return val
        s = val.strip()

        # {slot} 文本插值(prompt/台词类)
        if "{" in s:
            for k, v in ctx["slots"].items():
                s = s.replace("{" + k + "}", str(v))
            for k, v in ctx["derived"].items():
                s = s.replace("{" + k + "}", str(v))
            if "{" not in s:
                return s

        if not s.startswith(("$", "@")):
            return s
        if s.startswith("@"):                  # 产物引用, 留给 executor
            return s

        expr, add = s, 0.0                     # $x + n
        m = re.match(r"^(.*?)\s*\+\s*([\d.]+)$", s)
        if m:
            expr, add = m.group(1).strip(), float(m.group(2))

        out = self._lookup(expr, ctx)
        if add and isinstance(out, (int, float)):
            out = round(float(out) + add, 3)
        return out

    def _lookup(self, expr: str, ctx: dict) -> Any:
        card, seg, tl = ctx["card"], ctx["seg"], ctx["tl"]
        if expr == "$hook":
            i = (seg.hook_index or 1) - 1
            return self.hooks[i] if i < len(self.hooks) else None
        m = re.match(r"^\$hook_(\d+)$", expr)
        if m:
            i = int(m.group(1)) - 1
            return self.hooks[i] if i < len(self.hooks) else None
        if expr == "$background":
            bg = _abs(card.get("background", {}).get("default_image"))
            return bg or "@bgimg"              # 没有现成图 → 引用即将插入的生成步
        if expr == "$bgm_slice":
            if not self.bgm_source:
                return None
            return {"source": self.bgm_source, "t0": tl["t0"], "t1": tl["t1"]}
        if expr.startswith("$skill."):
            cur: Any = card
            for part in expr[len("$skill."):].split("."):
                cur = (cur or {}).get(part)
            return _abs(cur) or cur            # 素材路径转绝对, 其他原样
        if expr.startswith("$variant."):
            p = card.get("produces", {})
            v = (p.get("variants") or {}).get(seg.variant or "", {})
            return v.get(expr[len("$variant."):])
        log.warning("未知占位符 %s(原样保留)", expr)
        return expr

    # ── 单段编译 ─────────────────────────────────────────
    def compile_segment(self, seg, tl: dict) -> dict:
        card = self.store.get(seg.skill_id)
        if not card:
            return {"seg_id": seg.seg_id, "error": f"skill '{seg.skill_id}' 不存在", "calls": []}

        # 变体参数 + 派生量(供 {beats_text} 之类插值)
        p = card.get("produces", {})
        var = (p.get("variants") or {}).get(seg.variant or "", {})
        beats = var.get("beats") or []
        derived = {}
        if beats:
            derived["beats_text"] = (", ".join(f"{b}s" for b in beats[:-1])
                                     + f" and {beats[-1]}s" if len(beats) > 1
                                     else f"{beats[0]}s")
        ctx = {"card": card, "seg": seg, "tl": tl,
               "slots": seg.slots, "derived": derived}

        # 渲染 prompt 正文(变体卡用 prompt_Np)
        tpl = card.get("prompt_template") or ""
        if not tpl.strip() and seg.variant:
            tpl = card.get(f"prompt_{seg.variant}p", "")
        prompt = self._resolve(tpl, ctx).strip() if tpl.strip() else ""

        calls: List[dict] = []
        # 需要背景图但卡里没有现成图 → 前置一步 image_generation
        if card.get("needs_background"):
            if not _abs(card.get("background", {}).get("default_image")):
                bg_prompt = self._resolve(
                    (card.get("background") or {}).get("background_prompt", ""), ctx)
                calls.append({"id": "bgimg", "tool": "image_generation",
                              "params": {"prompt": bg_prompt.strip(),
                                         "size": "720x1280"}})

        for step in card.get("pipeline") or []:
            params = self._resolve(copy.deepcopy(step.get("params") or {}), ctx)
            # prompt 类参数由卡正文补齐(pipeline 里通常不重复写)
            if step["tool"] in ("seedance_t2v", "animate_move") and prompt:
                params.setdefault("prompt", prompt)
            if step["tool"] == "sonilo_text_to_music" and "prompt" in params:
                params["prompt"] = self._resolve(card.get("music_prompt", ""), ctx).strip()
            calls.append({"id": step["id"], "tool": step["tool"],
                          "local": bool(step.get("local")), "params": params})

        # 后期烧字(标题/字幕)——卡里声明了就补一步, 中文由我们自己渲染
        title = card.get("title_overlay") or {}
        if title.get("enabled"):
            calls.append({"id": "title", "tool": "burn_text", "local": True,
                          "params": {"video": f"@{calls[-1]['id']}",
                                     "text": seg.slots.get("title",
                                                           title.get("default_text", "")),
                                     "y_frac": title.get("y_frac", 0.16),
                                     "size": title.get("size", 76)}})

        return {"seg_id": seg.seg_id, "part": seg.part, "skill_id": seg.skill_id,
                "variant": seg.variant, "t0": tl["t0"], "t1": tl["t1"],
                "duration_s": tl["duration_s"], "tail_s": tl["tail_s"],
                "calls": calls}

    # ── 整片编译 ─────────────────────────────────────────
    def compile(self, sb) -> dict:
        tls = self.timeline(sb)
        segs = [self.compile_segment(s, tl) for s, tl in zip(sb.segments, tls)]
        cost = {"video_s": 0.0, "music_s": 0.0, "tts_chars": 0, "image_calls": 0}
        for s in segs:
            for c in s["calls"]:
                kind = BILLED.get(c["tool"])
                if kind == "video_s":
                    cost["video_s"] += float(c["params"].get("duration") or s["duration_s"])
                elif kind == "music_s":
                    cost["music_s"] += float(c["params"].get("duration") or 0)
                elif kind == "tts_chars":
                    cost["tts_chars"] += len(str(c["params"].get("text", "")))
                elif kind == "image_calls":
                    cost["image_calls"] += 1
        return {"product_name": sb.product_name, "person_count": sb.person_count,
                "total_duration_s": round(sum(s["duration_s"] + s["tail_s"] for s in segs), 2),
                "segments": segs, "cost_estimate": cost}


def _abs(rel) -> Optional[str]:
    """卡里的相对路径按 PROJECT_ROOT 解析; 存在才返回绝对路径, 否则 None。
    executor 可能从任意 cwd 启动, 计划里必须是绝对路径。"""
    if not rel or not isinstance(rel, str):
        return None
    p = Path(rel)
    if p.is_absolute():
        return str(p) if p.exists() else None
    q = PROJECT_ROOT / rel
    return str(q) if q.exists() else None
