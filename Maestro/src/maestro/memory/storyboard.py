"""StoryboardMemory — the brain's TASK-LEVEL working memory (需求 R1 / 用户 3.(1)).

用户需求原文的标准化:
    brain 在生成一条完整视频的任务中,维护一个【按时间顺序】的条目列表——
    每个条目 = 一个 shot:场景/镜头号、文本描述、keyframe(路径+来源)、
    视频(路径,如果已生成)、评审轨迹、物理轨迹——并且【持续更新】。

这就是 brain 的"眼前有什么"的唯一事实来源(single source of truth):
  • 窗口式生成(pipeline/window_loop.py)每走一步都读它(上一个 shot 的
    视频在哪、下一个待生成的是谁)并写它(生成结果、评审意见、修复动作);
  • to_brain_json() 是喂给 brain prompt 的紧凑视图 —— brain 依据它决定
    keyframe 策略和条件策略;
  • save()/load() 落盘为 run 目录下的 storyboard.json,人工可查、可续跑。

与 EpisodeMemory(长期记忆,memory/episode_memory.py)的分工:
  StoryboardMemory 只活在【一次任务】里(任务结束→蒸馏进 EpisodeMemory);
  EpisodeMemory 跨任务持久化 good/bad 案例。两层合起来 = 用户 3.(1)+3.(2)。

设计决定(用户没有明说、但必须定下来的点,在此精确解释):
  • 条目状态机 status ∈ pending → keyframed → generated → verified |
    generated_with_defects。"verified" 仅指内层评审循环收敛(review 全过);
    未收敛但已尽力的 shot 诚实地标 generated_with_defects,合成时照样拼入
    (交付最优可得),但状态永不谎报。
  • reviews 是【追加式】列表(每次评审一条),绝不覆盖 —— 这就是用户说的
    "评审意见要嵌入到轨迹中":轨迹 = 条目的完整演化史。
  • physics_trajectory 存测量链输出的逐实体轨迹(tracks.json 同构),没有
    测量时为 None —— 绝不放伪造的轨迹。
  • scene_idx 从 outline 文本解析("scene 2"/"场景2"),解析不到全归 scene 1
    (诚实降级:剧本没写分场就是单场)。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_SCENE_RE = re.compile(r"scene\s*(\d+)|场景\s*(\d+)", re.IGNORECASE)


def _parse_scene_idx(description: str) -> int:
    """outline 文本里带 "scene N"/"场景N" 就用 N,否则归 1。"""
    m = _SCENE_RE.search(description or "")
    if m:
        return int(m.group(1) or m.group(2))
    return 1


@dataclass
class ShotEntry:
    """一个 shot 的完整台账(用户示例的 '[k] scene i shot j: ...' 条目)。"""

    shot_idx: int                       # 全局顺序号(时间顺序的主键)
    scene_idx: int                      # 场号
    label: str                          # "scene 1 shot 2"(人读的键)
    description: str                    # 文本描述(playwriting 产物)
    # 交接棒(2026-07-15 需求 ②-①):brain 写剧本时逐镜声明"结束瞬间谁在哪、
    # 是动是停、朝什么方向"。下一镜的开头必须从这里继续;评审拿它当镜尾
    # 验收标准。brain 没输出 = 空串,不编造。
    end_state: str = ""
    # ViMax 借鉴(2026-07-17 P2):首尾帧变化幅度 large|medium|small(策略
    # 提示:小变 → extend/i2v 足够;大变 → flf 双锚/t2v);空 = 未声明。
    variation: str = ""
    # 换场/首镜的【开场静态快照】(纯静态,无进行中动作)—— 图计划的 t2i
    # prompt 底稿;续接镜留空(开场 = 上镜 end_state,不重复声明)。
    opening_frame: str = ""
    # keyframe:路径 + 来源(t2i 生成 | 素材库图片 | 素材视频抽帧 | 无)。
    # Image Plan 升级后仍保留:= images 里第一张 first/first_frame 角色图
    # (老策略 i2v_keyframe / flf2v_bridge 读它,兼容不破)。
    keyframe_path: Optional[str] = None
    keyframe_source: str = ""           # "t2i" | "asset_image" | "video_extract" | ""
    # Image Plan(升级版关键帧):brain 决定的 0-2 张图,每张带角色。
    #   plan  ∈ none | single_first_frame | single_reference |
    #           pair_first_last | pair_reference
    #   images = [{path, role: first_frame|reference|first|last,
    #              source: t2i|asset_image|video_extract, description}]
    # 执行降级(比如第二张图产不出来,pair→single)必须在 plan_degraded_from
    # 留痕 —— 台账绝不把降级伪装成 brain 的原始决定。
    image_plan: str = ""
    images: list = field(default_factory=list)
    plan_degraded_from: str = ""
    # 视频:生成后才有
    video_path: Optional[str] = None
    status: str = "pending"             # pending|keyframed|generated|verified|generated_with_defects
    # 这个 shot 用了什么生成条件(窗口策略 + 具体输入),brain 的决策记录
    condition: dict = field(default_factory=dict)
    # 评审轨迹:每次评审追加一条 {revision, weighted_total, failed_items,
    # physics_verdicts, brief_headline} —— 用户要求"评审意见嵌入轨迹"
    reviews: list = field(default_factory=list)
    # 内层修复循环的逐回合动作(res.actions 原样嵌入)
    repair_actions: list = field(default_factory=list)
    # 物理测量轨迹(实体 → 逐帧归一化 (x,y));无测量 = None,绝不伪造
    physics_trajectory: Optional[dict] = None

    def images_by_role(self, *roles: str) -> list:
        """按角色取图(路径存在的才算数 —— 检索命中但文件丢了 = 没有)。"""
        out = []
        for im in self.images:
            if im.get("role") in roles and im.get("path") \
                    and Path(im["path"]).exists():
                out.append(im)
        return out

    def to_brain_line(self) -> dict:
        """喂给 brain prompt 的单行紧凑视图(不带大对象)。

        图片 description 全文保留(2026-07-15 裁决 1.1:80 字截断让 brain
        只看到半截话,引用素材内容时必然写偏)。"""
        last_review = self.reviews[-1] if self.reviews else None
        return {
            "label": self.label,
            "description": self.description,
            "end_state": self.end_state,
            "variation": self.variation,
            "opening_frame": self.opening_frame,
            "status": self.status,
            "image_plan": self.image_plan,
            "images": [{"role": im.get("role"), "source": im.get("source"),
                        "description": im.get("description", "")}
                       for im in self.images],
            "keyframe": self.keyframe_path or "",
            "keyframe_source": self.keyframe_source,
            "video": self.video_path or "",
            "condition_strategy": self.condition.get("strategy", ""),
            "last_score": (last_review or {}).get("weighted_total"),
            "open_defects": (last_review or {}).get("n_failed", None),
        }


class StoryboardMemory:
    """按时间顺序的 shot 台账;持续更新;JSON 落盘。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.entries: list[ShotEntry] = []
        # 跨镜一致性载体(2026-07-17 审计):视频模型跨调用零记忆,生成
        # 角色/场景的一致性只能靠【每条 prompt 逐字复述同一段官方描述】。
        # cast = {实体名: 10-20 词外观描述符};setting = 场景陈设+光线一句。
        # 剧本层一次定稿,所有写 prompt 的人照抄,评审拿它当一致性标尺。
        self.cast: dict = {}
        self.setting: str = ""
        self._rev = 0                    # 单调更新计数(每次写 +1,可审计)

    # ── 构建 ──────────────────────────────────────────────────────────────
    @classmethod
    def from_outline(cls, outline: list[str], path: Optional[Path] = None
                     ) -> "StoryboardMemory":
        """playwriting 的 outline(每 shot 一句)→ 初始台账(全 pending)。
        场号从文本解析;同场内镜头号按出现顺序 1..n。"""
        sb = cls(path=path)
        per_scene: dict[int, int] = {}
        for i, desc in enumerate(outline):
            scene = _parse_scene_idx(desc)
            per_scene[scene] = per_scene.get(scene, 0) + 1
            sb.entries.append(ShotEntry(
                shot_idx=i, scene_idx=scene,
                label=f"scene {scene} shot {per_scene[scene]}",
                description=desc,
            ))
        sb._save()
        return sb

    # ── 查询(窗口循环的读端)───────────────────────────────────────────────
    def get(self, shot_idx: int) -> Optional[ShotEntry]:
        for e in self.entries:
            if e.shot_idx == shot_idx:
                return e
        return None

    def next_pending(self) -> Optional[ShotEntry]:
        """时间顺序上第一个还没生成视频的 shot(窗口的"下一个目标")。"""
        for e in self.entries:
            if e.video_path is None:
                return e
        return None

    def prev_generated(self, shot_idx: int) -> Optional[ShotEntry]:
        """shot_idx 之前【最近的】已有视频的 shot —— 窗口条件的锚。
        注意用"已生成"而非"已 verified":上一镜哪怕带遗留缺陷,它的尾帧仍是
        时间上唯一正确的续接点;用更早的完美镜头续接反而必然跳变。"""
        best = None
        for e in self.entries:
            if e.shot_idx < shot_idx and e.video_path is not None:
                best = e
        return best

    def all_generated(self) -> bool:
        return all(e.video_path is not None for e in self.entries)

    # ── 更新(窗口循环的写端;每次写都落盘)─────────────────────────────────
    def set_keyframe(self, shot_idx: int, path, source: str) -> None:
        e = self.get(shot_idx)
        if e is None:
            return
        e.keyframe_path = str(path)
        e.keyframe_source = source
        if e.status == "pending":
            e.status = "keyframed"
        self._save()

    def set_image_plan(self, shot_idx: int, plan: str, images: list,
                       degraded_from: str = "") -> None:
        """记录 brain 的 Image Plan 及其实际产出的图(升级版 set_keyframe)。

        兼容:第一张 first/first_frame 角色图同步进 keyframe_path/source,
        老的读取方(i2v_keyframe / flf2v_bridge / cand.keyframes)照常工作。"""
        e = self.get(shot_idx)
        if e is None:
            return
        e.image_plan = plan
        e.images = [dict(im) for im in images]
        e.plan_degraded_from = degraded_from
        first = next((im for im in e.images
                      if im.get("role") in ("first_frame", "first")
                      and im.get("path")), None)
        if first is not None:
            e.keyframe_path = str(first["path"])
            e.keyframe_source = first.get("source", "")
        if e.status == "pending" and (e.images or plan != "none"):
            e.status = "keyframed"
        self._save()

    def set_condition(self, shot_idx: int, condition: dict) -> None:
        e = self.get(shot_idx)
        if e is None:
            return
        e.condition = dict(condition)
        self._save()

    def add_review(self, shot_idx: int, review: dict) -> None:
        """追加一条评审记录(绝不覆盖 —— 评审意见嵌入轨迹)。"""
        e = self.get(shot_idx)
        if e is None:
            return
        e.reviews.append(dict(review))
        self._save()

    def set_result(self, shot_idx: int, video_path, converged: bool,
                   repair_actions: Optional[list] = None,
                   physics_trajectory: Optional[dict] = None) -> None:
        e = self.get(shot_idx)
        if e is None:
            return
        e.video_path = str(video_path)
        e.status = "verified" if converged else "generated_with_defects"
        if repair_actions:
            e.repair_actions = list(repair_actions)
        if physics_trajectory is not None:
            e.physics_trajectory = physics_trajectory
        self._save()

    # ── 视图 / 持久化 ─────────────────────────────────────────────────────
    def to_brain_json(self) -> list[dict]:
        """brain prompt 用的紧凑台账(时间顺序,一行一 shot)。"""
        return [e.to_brain_line() for e in self.entries]

    def summary(self) -> str:
        done = sum(1 for e in self.entries if e.video_path)
        ver = sum(1 for e in self.entries if e.status == "verified")
        return (f"{len(self.entries)} shots planned; {done} generated "
                f"({ver} verified)")

    def _save(self) -> None:
        self._rev += 1
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rev": self._rev,
                   "cast": self.cast, "setting": self.setting,
                   "entries": [asdict(e) for e in self.entries]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)          # 原子写,和 skill_library 同款

    @classmethod
    def load(cls, path: Path) -> "StoryboardMemory":
        sb = cls(path=Path(path))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sb._rev = int(data.get("rev", 0))
        sb.cast = dict(data.get("cast", {}) or {})
        sb.setting = str(data.get("setting", "") or "")
        sb.entries = [ShotEntry(**e) for e in data.get("entries", [])]
        return sb
