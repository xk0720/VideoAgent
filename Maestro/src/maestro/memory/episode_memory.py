"""EpisodeMemory — 跨任务长期记忆:good/bad 案例的【可执行】蒸馏 (需求 R2 / 用户 3.(2)).

用户需求原文的标准化:
    任务结束后,把"生成一条完整视频"的整条轨迹蒸馏进长期记忆,分 good case
    和 bad case;并回答"记忆的可执行化到底怎么实现"。

"可执行化"的落地(这就是实现方式,不再是口号):
  一条 episode 存的不是文本回忆,而是两张可以直接驱动决策的表——
  • replay 表(good 面):每个 shot 实际被 Verifier 接受的
    {keyframe 策略, 条件策略}。检索命中后,窗口 brain 可以直接【采纳】同款
    策略(决策记录 via="episode"),跳过重新推理 —— 检索即执行。
  • avoid 表(bad 面):被拒/未收敛的 {策略/工具, 场景特征, 失败原因}。
    检索命中后注入 brain prompt 的 do_not_repeat 区 —— 检索即禁止。
  修复层的可执行记忆已经存在(skill_library.distill_repair:缺陷签名→可重放
  工具序列);EpisodeMemory 把同一思想抬到【整条任务】层:
      修复技能 = "这类缺陷怎么修好"; episode = "这类任务怎么排产"。

good / bad 的判定(必须客观、来自 artifact,沿用信号诚实原则):
  • good episode:全部 shot 的内层评审循环收敛(status=verified)。
  • bad episode:任一 shot 未收敛。注意 bad episode 里收敛了的 shot 的策略
    仍进 replay 表(好的局部经验不陪葬),未收敛 shot 的策略进 avoid 表。
  没有任何字段来自 LLM 的自我评价 —— 全部由 Verifier/评审的客观结果推导。

检索(训练自由、确定性):prompt 关键词集合的 Jaccard 重叠打分,同
lesson_library 的哲学 —— 不引 embedding 依赖,分数可复现可解释。
持久化:JSONL 追加 + 原子重写,和 skill_library 同款。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_WORD_RE = re.compile(r"[a-zA-Z一-鿿]+")
# 检索时忽略的高频虚词(中英),避免 "a/the/的" 撑高相似度
_STOP = {"a", "an", "the", "of", "in", "on", "and", "to", "with", "is",
         "then", "into", "onto", "at", "for", "by", "从", "的", "了", "在",
         "和", "与", "一个", "上", "下"}


def _keywords(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")
            if w.lower() not in _STOP and len(w) > 1}


@dataclass
class EpisodeRecord:
    """一次完整视频任务的可执行蒸馏。"""

    episode_id: str
    user_prompt: str
    keywords: list[str]                 # 检索键(确定性关键词集合)
    outcome: str                        # "good" | "bad"
    n_shots: int
    final_video: str = ""
    # replay 表:Verifier 接受的 per-shot 策略(可直接采纳执行)
    #   [{label, description, keyframe_strategy, condition_strategy,
    #     converged, final_score}]
    replay: list = field(default_factory=list)
    # avoid 表:失败模式(检索命中 → 注入 do_not_repeat)
    #   [{label, condition_strategy, keyframe_strategy, reason}]
    avoid: list = field(default_factory=list)
    # 修复工具的接受/拒绝台账(聚合自内层循环 actions):tool → [n_ok, n_rej]
    repair_tool_stats: dict = field(default_factory=dict)
    created_at: float = 0.0
    uses: int = 0                       # 被检索采纳的次数(EMA 淘汰的输入)


class EpisodeMemory:
    """长期 episode 库:distill(写入)+ retrieve/guidance(读出)。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.episodes: list[EpisodeRecord] = []
        if self.path and self.path.exists():
            self._load()

    # ── 写入:任务结束时由窗口循环调用 ─────────────────────────────────────
    def distill_episode(self, user_prompt: str, storyboard,
                        final_video: str = "") -> EpisodeRecord:
        """从跑完的 StoryboardMemory 蒸馏一条 episode(good/bad 由客观状态定)。

        replay/avoid 的取材规则(精确定义,不含主观判断):
          shot.status == "verified"  → 该 shot 的 (keyframe_source,
            condition.strategy) 进 replay 表,附最终分;
          其他已生成状态             → 进 avoid 表,reason 带上最后一条评审的
            头条(为什么没修好),供未来 brain 阅读。
        repair_tool_stats 聚合每个 shot 的 repair_actions 的 outcome。
        """
        replay, avoid = [], []
        tool_stats: dict[str, list[int]] = {}
        all_verified = True
        for e in storyboard.entries:
            if e.video_path is None:
                all_verified = False
                continue
            last = e.reviews[-1] if e.reviews else {}
            row = {
                "label": e.label,
                "description": e.description,
                # Image Plan(数量+角色):可执行重放的主键;老台账没有该
                # 字段就留空("" → 不重放,诚实降级到 LLM/兜底层)。
                "image_plan": getattr(e, "image_plan", "") or "",
                "keyframe_strategy": e.keyframe_source or "none",
                # "strategy" 是【实际执行】的条件(胜出 seed 用的那个);brain 的
                # 原始决定和降级痕迹一并带上 —— 未来 brain 读 avoid/replay 时能
                # 分清"策略本身不行"和"策略没跑成、降级顶上"两种情况。
                "condition_strategy": e.condition.get("strategy", "t2v"),
                "decided_strategy": e.condition.get("decided_strategy",
                                                    e.condition.get("strategy", "t2v")),
                "degraded_from": e.condition.get("degraded_from"),
                "converged": e.status == "verified",
                "final_score": last.get("weighted_total"),
            }
            if e.status == "verified":
                replay.append(row)
            else:
                all_verified = False
                avoid.append({
                    "label": e.label,
                    "condition_strategy": row["condition_strategy"],
                    "decided_strategy": row["decided_strategy"],
                    "degraded_from": row["degraded_from"],
                    "keyframe_strategy": row["keyframe_strategy"],
                    "reason": last.get("brief_headline")
                              or f"{last.get('n_failed', '?')} defects unresolved",
                })
            for a in e.repair_actions:
                t = str(a.get("tool", "?"))
                ok = a.get("outcome") == "accepted"
                tool_stats.setdefault(t, [0, 0])[0 if ok else 1] += 1

        rec = EpisodeRecord(
            episode_id="ep_" + hashlib.md5(
                (user_prompt + str(len(self.episodes))).encode()
            ).hexdigest()[:10],
            user_prompt=user_prompt,
            keywords=sorted(_keywords(user_prompt)),
            outcome="good" if all_verified and storyboard.entries else "bad",
            n_shots=len(storyboard.entries),
            final_video=str(final_video or ""),
            replay=replay,
            avoid=avoid,
            repair_tool_stats=tool_stats,
            created_at=time.time(),
        )
        self.episodes.append(rec)
        self._save()
        return rec

    # ── 读出:任务开始时由窗口循环调用 ─────────────────────────────────────
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
        """窗口 brain 的开工简报:可采纳的 replay 提示 + 必须规避的 avoid 表。

        good episode 的 replay 行原样给出(brain 可以直接照抄策略,记
        via="episode");bad episode 只贡献 avoid(它的教训,不是它的做法)。
        命中的 episode 记一次 uses(使用台账,同 skill mark_used 哲学)。"""
        hits = self.retrieve(user_prompt, k)
        replay_hints, avoid, shapes = [], [], []
        for ep in hits:
            ep.uses += 1
            if ep.outcome == "good":
                replay_hints.extend(ep.replay)
            else:
                # bad episode 里收敛的局部经验也值得借(见模块 docstring)
                replay_hints.extend(r for r in ep.replay if r.get("converged"))
            avoid.extend(ep.avoid)
            # 任务形状经验:相似任务当年拆了几镜、成没成 —— 供 playwriting
            # 的 brain 决定本次 shot 数量(用户裁决:数量绝不预设,由 brain
            # 结合这些历史经验自己定)。
            shapes.append({"n_shots": ep.n_shots, "outcome": ep.outcome,
                           "user_prompt": ep.user_prompt[:80]})
        if hits:
            self._save()
        return {"replay_hints": replay_hints[:8], "avoid": avoid[:8],
                "past_task_shapes": shapes[:5],
                "n_episodes_matched": len(hits)}

    # ── 持久化(JSONL,原子重写)─────────────────────────────────────────────
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
        self.episodes = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                self.episodes.append(EpisodeRecord(**json.loads(line)))
