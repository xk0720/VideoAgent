"""EpisodeMemory — 跨任务长期记忆:台账蒸馏成【技能轨迹】(2026-08-13 用户设计).

一部片 = 一条轨迹:
    轨迹头 header(全片共享上下文:人物正典/场景切分/引用解读表/参考库配方)
    + 步序列 trajectory(一镜一步,每步三格):
        context  该镜决策时真正看到的东西(裁剪:字段以 brain 决策输入为准)
        action   工具/策略选择 + 完整 prompt(草稿+终稿,不截断)+ 引用图例
        feedback VLM 评语头条 + 分数(未评审 → null,诚实留空)

引用自解释法(用户令):轨迹里出现的一切引用(image_N 图例值、挑中的
space_view)都必须能在 header.reference_registry 查到身份与图注 ——
"new_0 是什么、图注是什么"在记录内闭环,不需要回台账。

一份格式三个用途:检索当范文(few-shot)/ 高分步直接进训练语料
(与 RL 的 (context, action, reward) 同构)/ 人工复盘。

good / bad / ungraded 判定(全部客观信号,无 LLM 自评):
  good     全镜 Verifier 收敛;
  bad      开了评审但有镜未收敛;
  ungraded 全程无实证评审(--no-review 的占位评审行不算)——
           完成的片是"最佳可得蓝图",不是失败案例。

检索(确定性可复现):关键词 Jaccard;中文按字二元组切分(2026-08-13
修:整段汉字当一个词会让检索退化成全句精确匹配)。
持久化:JSONL 追加 + 原子重写;旧版字段(replay/avoid/shot_plans 等
2026-08-13 前的三表制)load 时静默忽略 —— 轨迹制取代表制。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

_WORD_RE = re.compile(r"[a-zA-Z一-鿿]+")
_STOP = {"a", "an", "the", "of", "in", "on", "and", "to", "with", "is",
         "then", "into", "onto", "at", "for", "by", "从", "的", "了", "在",
         "和", "与", "一个", "上", "下"}


def _keywords(text: str) -> set[str]:
    """英文按词;中文按字二元组(整词也保留,精确命中权重自然更高)。"""
    out: set[str] = set()
    for w in _WORD_RE.findall(text or ""):
        lw = w.lower()
        if lw in _STOP or len(lw) < 2:
            continue
        out.add(lw)
        if re.search(r"[一-鿿]", lw):
            out.update(lw[i:i + 2] for i in range(len(lw) - 1)
                       if lw[i:i + 2] not in _STOP)
    return out


def _graded(rv: dict) -> bool:
    """一条评审行算不算实证:--no-review 的占位行(review_disabled、
    零证据零分)不算 —— 否则完成片被误判 bad(2026-08-13 实锤)。"""
    return rv.get("stop_reason") != "review_disabled" and bool(
        (rv.get("review_evidence") or {}).get("checklist_items", 0)
        or rv.get("n_failed") or rv.get("weighted_total"))


@dataclass
class EpisodeRecord:
    """一部片的技能轨迹(header + steps)。"""

    episode_id: str
    user_prompt: str
    keywords: list[str]                 # 检索键(片名+全部分镜描述)
    outcome: str                        # "good" | "bad" | "ungraded"
    n_shots: int
    final_video: str = ""
    # 轨迹头:全片共享上下文 + 引用解读表(轨迹内一切引用在此闭环)
    header: dict = field(default_factory=dict)
    # 步序列:[{step, label, context{}, action{}, feedback{}}]
    trajectory: list = field(default_factory=list)
    # 修复工具接受/拒绝台账(聚合;修复线关闭时自然为空)
    repair_tool_stats: dict = field(default_factory=dict)
    created_at: float = 0.0
    uses: int = 0


class EpisodeMemory:
    """长期轨迹库:distill(写入)+ retrieve/guidance(读出)。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.episodes: list[EpisodeRecord] = []
        if self.path and self.path.exists():
            self._load()

    # ── 写入:任务结束时由窗口循环调用 ─────────────────────────────────
    def distill_episode(self, user_prompt: str, storyboard,
                        final_video: str = "") -> EpisodeRecord:
        """台账 → 技能轨迹。裁剪原则:留"决策与结局",丢"过程与文件"
        (路径/种子/重试流水/评审逐条/图注全文只在 registry 留必要项)。"""
        entries = list(storyboard.entries)
        any_review = any(_graded(r) for e in entries
                         for r in (e.reviews or []))
        cast = dict(getattr(storyboard, "cast", {}) or {})
        spaces = getattr(storyboard, "spaces", {}) or {}
        backgrounds = getattr(storyboard, "backgrounds", {}) or {}

        # ── 引用解读表:轨迹里出现的每个引用键在此有身份+图注 ──
        registry: dict = {}
        for name in (getattr(storyboard, "portraits", {}) or {}):
            registry[f"portrait:{name}"] = {
                "kind": "portrait",
                "desc": cast.get(name, "")}
        for bg, rec_ in backgrounds.items():
            registry[f"bg_plate:{bg}"] = {
                "kind": "bg_plate", "src": (rec_ or {}).get("src", "")}
        for bg, views in spaces.items():
            for v, it in views.items():
                registry[f"space_view:{bg}/{v}"] = {
                    "kind": "space_view",
                    "src": (it or {}).get("src", ""),
                    "caption": (it or {}).get("caption", "")}
        registry["derived_junction_frame"] = {
            "kind": "derived_frame",
            "desc": "过渡视频切点后的首帧(由上镜末帧+肖像+空间视图派生);"
                    "机器承接句把它钉为本镜开场画面"}
        registry["prev_shot_tail_frame"] = {
            "kind": "tail_frame", "desc": "上一镜真实视频的最后一帧"}

        # 路径 → registry 键(图例判读用)
        _by_name: dict[str, str] = {}
        for name, pth in (getattr(storyboard, "portraits", {})
                          or {}).items():
            _by_name[Path(str(pth)).name] = f"portrait:{name}"
        for bg, rec_ in backgrounds.items():
            _by_name[Path(str((rec_ or {}).get("path", ""))).name] = \
                f"bg_plate:{bg}"
        for bg, views in spaces.items():
            for v, it in views.items():
                _by_name[Path(str((it or {}).get("path", ""))).name] = \
                    f"space_view:{bg}/{v}"

        def _ref_key(path_str: str) -> str:
            n = Path(str(path_str)).name
            if n in _by_name:
                return _by_name[n]
            if "junction_derived" in n:
                return "derived_junction_frame"
            if "tail" in n:
                return "prev_shot_tail_frame"
            return f"other:{n}"

        # ── 步序列 ──
        steps, tool_stats = [], {}
        all_verified = True
        prev = None
        for i, e in enumerate(entries):
            if e.status != "verified":
                all_verified = False
            if e.video_path is None:
                prev = e
                continue
            cond = e.condition or {}
            jm = getattr(e, "junction_meta", None) or {}
            sv = jm.get("space_view")
            sv_key = None
            if isinstance(sv, dict) and sv.get("view"):
                sv_key = f"space_view:{getattr(e, 'bg_id', '')}/" \
                         f"{sv['view']}"
            elif isinstance(sv, str) and sv:
                sv_key = f"space_view:{getattr(e, 'bg_id', '')}/{sv}"
            last = next((r for r in reversed(e.reviews or [])
                         if _graded(r)), None)
            steps.append({
                "step": i + 1,
                "label": e.label,
                "context": {
                    "shot": e.description,
                    "camera_facing": getattr(e, "camera_facing", ""),
                    "bg_id": getattr(e, "bg_id", ""),
                    "prev_end_state": (prev.end_state if prev else ""),
                    "junction": {
                        "kind": jm.get("kind"),
                        "fallback_to": jm.get("fallback_to"),
                        "space_view": sv_key,     # ← registry 键,可查图注
                        "stitcher_via": (jm.get("stitcher")
                                         or {}).get("via"),
                    },
                },
                "action": {
                    "strategy": cond.get("strategy", "t2v"),
                    "decided_strategy": cond.get(
                        "decided_strategy", cond.get("strategy", "t2v")),
                    "degraded_from": cond.get("degraded_from"),
                    "image_plan": getattr(e, "image_plan", "") or "",
                    "prompt": str(cond.get("final_prompt")
                                  or cond.get("brain_prompt") or ""),
                    "prompt_draft": getattr(e, "draft_prompt", "") or "",
                    "refs": {f"image_{k + 1}": _ref_key(rp)
                             for k, rp in enumerate(
                                 cond.get("reference_images") or [])},
                },
                "feedback": {
                    "vlm_headline": (last or {}).get("brief_headline")
                                    or None,
                    "score": (last or {}).get("weighted_total"),
                    "converged": (True if e.status == "verified"
                                  else (None if not any_review
                                        else False)),
                },
            })
            for a in e.repair_actions:
                t_ = str(a.get("tool", "?"))
                ok = a.get("outcome") == "accepted"
                tool_stats.setdefault(t_, [0, 0])[0 if ok else 1] += 1
            prev = e

        bgs_map: dict[str, list] = {}
        for e in entries:
            bgs_map.setdefault(getattr(e, "bg_id", "") or "?",
                               []).append(e.shot_idx)
        recipe = {bg: {} for bg in spaces}
        for bg, views in spaces.items():
            for it in views.values():
                s = (it or {}).get("src", "?")
                recipe[bg][s] = recipe[bg].get(s, 0) + 1

        rec = EpisodeRecord(
            episode_id="ep_" + hashlib.md5(
                (user_prompt + str(len(self.episodes))).encode()
            ).hexdigest()[:10],
            user_prompt=user_prompt,
            keywords=sorted(_keywords(
                user_prompt + " " + " ".join(
                    e.description or "" for e in entries))),
            outcome=("good" if all_verified and entries
                     else "ungraded" if not any_review and entries
                     else "bad"),
            n_shots=len(entries),
            final_video=str(final_video or ""),
            header={
                "task": (user_prompt or "")[:200],
                "cast": cast,
                "setting": str(getattr(storyboard, "setting", ""))[:300],
                "scene_layout": {
                    "n_scenes": len({e.scene_idx for e in entries}),
                    "bgs": bgs_map},
                "reference_registry": registry,
                "asset_recipe": {"spaces": recipe},
            },
            trajectory=steps,
            repair_tool_stats=tool_stats,
            created_at=time.time(),
        )
        self.episodes.append(rec)
        self._save()
        return rec

    # ── 读出:任务开始时由窗口循环调用 ─────────────────────────────────
    def retrieve(self, user_prompt: str, k: int = 3) -> list[EpisodeRecord]:
        """Jaccard(关键词) 相似度 top-k;0 分不返回(不硬凑无关经验)。"""
        q = _keywords(user_prompt)
        if not q or not self.episodes:
            return []
        scored = []
        for ep in self.episodes:
            s = set(ep.keywords)
            j = len(q & s) / max(1, len(q | s))
            if j > 0:
                scored.append((j, ep))
        scored.sort(key=lambda t: (t[0], t[1].created_at), reverse=True)
        return [ep for _, ep in scored[:k]]

    def guidance_for(self, user_prompt: str, k: int = 3) -> dict:
        """开工简报(从轨迹现场推导,读侧契约不变):
        replay_hints  good/ungraded 片的步 + bad 片里收敛的步(策略先验)
        avoid         实证失败的步(策略 + VLM 评语)
        past_task_shapes  相似任务当年拆几镜、成没成(供 playwriting)"""
        hits = self.retrieve(user_prompt, k)
        replay_hints, avoid, shapes = [], [], []
        for ep in hits:
            ep.uses += 1
            for st in ep.trajectory:
                fb = st.get("feedback") or {}
                act = st.get("action") or {}
                row = {"label": st.get("label"),
                       "description": (st.get("context")
                                       or {}).get("shot", ""),
                       "image_plan": act.get("image_plan", ""),
                       "condition_strategy": act.get("strategy"),
                       "decided_strategy": act.get("decided_strategy"),
                       "degraded_from": act.get("degraded_from"),
                       "converged": fb.get("converged"),
                       "final_score": fb.get("score")}
                if fb.get("converged") is False:
                    avoid.append({
                        "label": st.get("label"),
                        "condition_strategy": act.get("strategy"),
                        "decided_strategy": act.get("decided_strategy"),
                        "degraded_from": act.get("degraded_from"),
                        "reason": fb.get("vlm_headline")
                                  or "unconverged"})
                elif ep.outcome in ("good", "ungraded") \
                        or fb.get("converged"):
                    replay_hints.append(row)
            shapes.append({"n_shots": ep.n_shots, "outcome": ep.outcome,
                           "user_prompt": ep.user_prompt[:80]})
        if hits:
            self._save()
        return {"replay_hints": replay_hints[:8], "avoid": avoid[:8],
                "past_task_shapes": shapes[:5],
                "n_episodes_matched": len(hits)}

    # ── 持久化(JSONL,原子重写;旧字段静默忽略)────────────────────────
    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for ep in self.episodes:
                f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def _load(self) -> None:
        known = {f.name for f in fields(EpisodeRecord)}
        self.episodes = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = {k: v for k, v in json.loads(line).items() if k in known}
            self.episodes.append(EpisodeRecord(**d))
