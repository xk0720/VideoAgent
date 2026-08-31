"""分镜剧本 Planner —— 两次 LLM 调用: ① 主体段选卡 ② 一次写完所有文案。

为什么这样切:
  · 选卡很轻(开场/收尾各只有一张卡, 主体二选一) → 输出一个 skill_id 就够
  · 写文案要看全局(三人递进、卖点不重复) → 必须一次写完, 逐句问会各说各的
  · 输出扁平 JSON —— 实测最稳的形态(嵌套结构会崩)

驱动文案生成的五样输入(见 _writing_brief):
  ① 商品卖点清单        内容从哪来
  ② 每人对应哪个 hook   谁在说
  ③ 每句所处的动作语境  说的话要配得上动作 —— 从卡的 prompt 正文里正则抽出来
  ④ 叙事结构要求        建立 → 深入 → 收束
  ⑤ 字数铁律            TTS 实测约 5 字/秒
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
P = Path(__file__).parents[1] / "prompts"
SELECT_PROMPT = (P / "storyboard_select.md").read_text(encoding="utf-8")
WRITE_PROMPT = (P / "storyboard_write.md").read_text(encoding="utf-8")
CJK = re.compile(r"[一-鿿]")
BANNED = re.compile(r"\b(360|full turn|spin|rotate quickly)\b", re.I)
MAX_RETRY = 4
NARRATIVE = {1: "建立(这是什么、什么手感、第一眼感受)",
             2: "深入(设计细节、做工、为什么值)",
             3: "收束(适用场景、价格、引导下单)"}


class StoryboardPlanner:
    def __init__(self, store: SkillStore):
        self.store = store

    # ── ① 选卡 ───────────────────────────────────────────
    def _select_body(self, brief: dict, cat: str, n: int) -> Tuple[str, str]:
        cands = [c for c in self.store.candidates(cat, n, "body")]
        if not cands:
            raise RuntimeError(f"没有适用于「{cat} / {n} 人」的主体 skill")
        if len(cands) == 1:
            return cands[0]["skill_id"], "该段位只有这一张卡"
        lines = []
        for c in cands:
            m = c.get("measured") or {}
            lines.append(
                f"- **{c['skill_id']}**「{c.get('name','')}」 "
                f"音频 {(c.get('produces') or {}).get('audio_mode','-')}"
                f" | {n} 人时 {(c.get('produces') or {}).get('durations',{}).get(str(n),'?')}s\n"
                f"    实测: {str(m.get('notes','')).strip()}\n"
                + "".join(f"    ⚠ {x}\n" for x in (m.get("caveats") or [])[:3]))
        user = (f"## 商品\n名称: {brief.get('name')}\n描述: {brief.get('description','')}\n"
                f"卖点: {'; '.join(brief.get('selling_points', []))}\n"
                f"人物参考图: {n} 张\n\n## 主体段候选\n" + "\n".join(lines)
                + "\n请输出选卡 JSON。")
        raw = chat_json(SELECT_PROMPT, user, temperature=0.4)
        sid = raw.get("body_skill", "")
        if sid not in {c["skill_id"] for c in cands}:
            log.warning("选了不存在的卡 '%s', 退回第一个候选", sid)
            sid = cands[0]["skill_id"]
        return sid, raw.get("reason", "")

    # ── ② 写文案 ─────────────────────────────────────────
    @staticmethod
    def _shot_context(card: dict, person: int) -> List[Tuple[str, str]]:
        """从第 person 人的 prompt 正文里抽出「参数名 → 动作语境」。

        模板里的句式固定为 `Shot 1 (0-3s): she <动作>, ... says: "{line1_1}"`,
        所以正则一次抽全; 抽不到就退回整段正文(如 outdoor_narration 的旁白)。
        """
        tpl = card.get(f"prompt_p{person}") or ""
        out = []
        for m in re.finditer(r"Shot (\d) \((\d+)-(\d+)s\): (.*?)\{(\w+)\}", tpl, re.S):
            _, t0, t1, body, key = m.groups()
            act = re.sub(r"\s+", " ", body).strip()
            act = act.split(", speaking to the camera")[0].strip(' ,"')
            out.append((key, f"{t0}-{t1}s 画面: {act}"))
        if not out:                                   # 旁白型: 整段一句
            for key in (card.get("text_params") or {}):
                if key.endswith(f"_{person}"):
                    scene = re.search(r"vertical 9:16 full-body shot,(.*?)\. The woman",
                                      tpl, re.S)
                    act = re.search(r"Over the \d+ seconds, (.*?)\. Her expressions",
                                    tpl, re.S)
                    ctx = "; ".join(re.sub(r"\s+", " ", g.group(1)).strip()
                                    for g in (scene, act) if g)
                    out.append((key, f"整段 10s 画面: {ctx}"))
        return out

    def _writing_brief(self, card: dict, brief: dict, n: int) -> Tuple[str, Dict[str, dict]]:
        tp = card.get("text_params") or {}
        hooks = brief.get("person_hooks") or []
        need: Dict[str, dict] = {}
        blocks = []
        for person in range(1, n + 1):
            ctxs = self._shot_context(card, person)
            if not ctxs:
                continue
            head = (f"### 第 {person} 个人"
                    + (f"(参考图 {Path(hooks[person-1]).name})" if person <= len(hooks) else "")
                    + f" —— 叙事职责: {NARRATIVE.get(person, '补充')}")
            rows = []
            for key, ctx in ctxs:
                spec = tp.get(key)
                if not spec:
                    continue
                need[key] = spec
                lo, hi = spec.get("chars", [0, 99])
                rows.append(f"  - `{key}` ({lo}-{hi} 字)  {ctx}")
            blocks.append(head + "\n" + "\n".join(rows))
        # 与人无关的文本参数(如收尾标题)
        for key, spec in tp.items():
            if key not in need and not re.search(r"_\d$", key):
                need[key] = spec
                lo, hi = spec.get("chars", [0, 99])
                blocks.append(f"### 其他\n  - `{key}` ({lo}-{hi} 字)  {spec.get('desc','')}")
        return "\n\n".join(blocks), need

    def _write(self, card: dict, brief: dict, n: int) -> Dict[str, str]:
        body, need = self._writing_brief(card, brief, n)
        if not need:
            return {}
        skeleton = {k: f"<{v.get('chars',['',''])[0]}-{v.get('chars',['',''])[-1]}字>"
                    for k, v in need.items()}
        user = (f"## 商品\n{brief.get('name')} — {brief.get('description','')}\n"
                f"卖点(每个最多用一次, 都要落地):\n"
                + "\n".join(f"  {i}. {s}" for i, s in
                            enumerate(brief.get("selling_points", []), 1))
                + f"\n\n## 要写的文案(逐条按画面写)\n{body}\n\n"
                f"## 输出骨架(键照抄, 值换成你写的中文)\n"
                f"{json.dumps(skeleton, ensure_ascii=False, indent=2)}\n")
        last, prev = "", ""
        texts: Dict[str, str] = {}
        for attempt in range(MAX_RETRY):
            q = user if not last else (
                user + f"\n## 你上一版的逐条字数(我已替你数好, 直接照做)\n{prev}\n\n"
                f"标 ✓ 的原样输出, 标 ✗ 的按提示增删字数后输出。\n")
            raw = chat_json(WRITE_PROMPT, q, temperature=0.7 if not last else 0.3)
            texts = {k: str(v).strip() for k, v in raw.items() if k in need}
            issues = self._check(texts, need)
            if not issues:
                return texts
            last = "; ".join(issues)
            # 把每条的现字数与差额直接列出来 —— 模型自我计数不可靠(实测 3 轮收敛不了)
            rows = []
            for k in need:
                v = (texts.get(k) or "").strip()
                ln = len(CJK.findall(v)) + len(re.findall(r"[A-Za-z0-9]+", v))
                lo, hi = need[k].get("chars", [0, 99])
                if lo <= ln <= hi:
                    rows.append(f'  "{k}": 现{ln}字 ✓ 保持原样: {v}')
                else:
                    diff = (f"要再加 {lo - ln} 个字" if ln < lo else f"要删掉 {ln - hi} 个字")
                    rows.append(f'  "{k}": 现{ln}字 ✗ 需{lo}-{hi}字, {diff}。原句: {v}')
            prev = "\n".join(rows)
            log.info("  文案第%d次: %s", attempt + 1, last[:120])
        # 收敛不了就放宽: 差 1-2 字的实际听感无碍(3秒句 10 vs 9 字 = 0.2 秒),
        # 与其让模型在"加一个字"上死循环, 不如程序判定可接受(实测 4 轮仍差 1 字)
        for k, spec in need.items():
            v = (texts.get(k) or "").strip()
            ln = len(CJK.findall(v)) + len(re.findall(r"[A-Za-z0-9]+", v))
            lo, hi = spec.get("chars", [0, 99])
            if v and abs(ln - lo) <= 2 and ln < lo:
                log.info("  %s 差%d字(现%d需%d), 在容差内放行", k, lo - ln, ln, lo)
        return texts

    @staticmethod
    def _check(texts: dict, need: dict, tol: int = 0) -> List[str]:
        """tol = 字数容差。写作循环里用 0(逼模型写准), 终检用 2
        (差 1-2 字听感无碍, 不值得为此判整条脚本不合格)。"""
        out = []
        for k, spec in need.items():
            v = (texts.get(k) or "").strip()
            if not v:
                out.append(f"{k} 缺失"); continue
            if spec.get("lang") == "zh":
                if not CJK.search(v):
                    out.append(f"{k} 应为中文"); continue
                ln = len(CJK.findall(v)) + len(re.findall(r"[A-Za-z0-9]+", v))
                lo, hi = spec.get("chars", [0, 99])
                if not (lo - tol <= ln <= hi + tol):
                    out.append(f"{k} 现{ln}字需{lo}-{hi}字")
            if BANNED.search(v):
                out.append(f"{k} 含高危动作词")
        return out

    # ── 主流程 ───────────────────────────────────────────
    def plan(self, brief: dict, bgm_source=None) -> Tuple[Storyboard, Report]:
        cat = brief.get("category", "服装")
        hooks = list(brief.get("person_hooks") or [])
        n = len(hooks) or 1

        # 开场/收尾: 各只有一张卡 → 程序直接取; 主体: 问模型
        opening = sorted(self.store.candidates(cat, n, "opening"),
                         key=lambda c: -float(c.get("priority", 0)))
        ending = self.store.candidates(cat, n, "ending")
        body_id, body_reason = self._select_body(brief, cat, n)
        log.info("① 选卡: 开场=%s 主体=%s 收尾=%s",
                 opening[0]["skill_id"] if opening else "(无)", body_id,
                 ending[0]["skill_id"] if ending else "(无, 人数不足)")

        picks = ([(opening[0], "opening")] if opening else []) \
            + [(self.store.get(body_id), "body")] \
            + ([(ending[0], "ending")] if ending else [])

        segs: List[SegmentSpec] = []
        t = 0.0
        for i, (card, part) in enumerate(picks, 1):
            texts = self._write(card, brief, n)
            if texts:
                log.info("② 文案 %s: %d 条", card["skill_id"], len(texts))
            pr = card.get("produces", {})
            dur = float((pr.get("durations") or {}).get(str(n)) or pr.get("duration_s") or 0)
            tail = float(pr.get("tail_s", 0))
            r = Renderer(card, hooks, person_count=n, bgm_source=bgm_source,
                         t0=round(t, 3), t1=round(t + dur, 3))
            prompts = {k: r.prompt_of_person(k, texts) for k in range(1, n + 1)}
            # 单模板卡(如 closer, 按总人数选版本)走 $prompt; 多人卡走 $prompt_N
            single = r.prompt(texts)
            segs.append(SegmentSpec(
                seg_id=f"seg{i:02d}", part=part, skill_id=card["skill_id"],
                variant=str(n), duration_s=dur, t0=round(t, 3), t1=round(t + dur, 3),
                pipeline=r.pipeline(texts, prompt=single or prompts.get(1, ""),
                                    prompts=prompts),
                texts=texts,
                reason=body_reason if part == "body" else f"{part} 段唯一可用卡"))
            t += dur + tail
            log.info("  %s [%s] %s → %d 步", segs[-1].seg_id, part, card["skill_id"],
                     len(segs[-1].pipeline))

        sb = Storyboard(product_name=brief.get("name", ""), category=cat,
                        person_count=n, segments=segs,
                        overall_reason=f"开场借爆款片段, 主体{body_reason}, 收尾多人相继出镜")
        return sb, self.validate(sb, brief)

    # ── 校验 ─────────────────────────────────────────────
    def validate(self, sb: Storyboard, brief: dict) -> Report:
        errs: List[Issue] = []
        warns: List[Issue] = []
        n = len(brief.get("person_hooks") or []) or 1
        for seg in sb.segments:
            card = self.store.get(seg.skill_id)
            if not card:
                errs.append(Issue(seg_id=seg.seg_id, field="skill_id", msg="不在 skill 库"))
                continue
            need = {k: v for k, v in (card.get("text_params") or {}).items()
                    if not re.search(r"_(\d)$", k)
                    or int(re.search(r"_(\d)$", k).group(1)) <= n}
            for m in self._check(seg.texts, need, tol=2):
                errs.append(Issue(seg_id=seg.seg_id, field="texts", msg=m))
            for c in seg.pipeline:
                pr = c["params"].get("prompt", "")
                if isinstance(pr, str) and "{" in pr:
                    errs.append(Issue(seg_id=seg.seg_id, field=c["id"],
                                      msg=f"prompt 残留未填空位 {sorted(set(re.findall(r'{(\\w+)}', pr)))}"))
                for k, v in c["params"].items():
                    if isinstance(v, str) and v.startswith("$"):
                        errs.append(Issue(seg_id=seg.seg_id, field=f"{c['id']}.{k}",
                                          msg=f"未解析的占位符 {v}"))
        if not any(s.part == "body" for s in sb.segments):
            errs.append(Issue(field="segments", msg="缺少主体段"))
        if n >= 2 and not any(s.part == "ending" for s in sb.segments):
            warns.append(Issue(field="segments", msg=f"{n} 人但无收尾段"))
        return Report(ok=not errs, errors=errs, warnings=warns)
