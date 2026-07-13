"""Window-based movie generation — the OUTER brain loop (需求 R3 / 用户第 4 点).

═══════════════════════════════════════════════════════════════════════════
用户需求的标准化(每一条在代码里的落点都标了 §号,一条不落):

§A  playwriting(用户 4.(1) 前半)
    用户 prompt → 按时间顺序的全部 shot 文本描述(Screenwriter + Director,
    复用现有 agent)→ 建 StoryboardMemory 台账(需求 R1:brain 维护的
    按时间顺序、可持续更新的 keyframe/video+描述 列表)。

§B  keyframe 阶段(用户 4.(1) 的 (1)(2)(3) 三种方式,brain 逐 shot 选择)
    B1 "t2i"           按 shot 描述构建 prompt → 文生图(WaveSpeed t2i 能力)
    B2 "asset_image"   从用户素材库检索图片当 keyframe(identity/style 锚)
    B3 "video_extract" 从用户视频素材库检索片段 → 抽帧当 keyframe
    B4 "none"          不要 keyframe(纯 t2v 路线)—— 用户没列但必须存在:
                       三种方式都无可用输入时的诚实降级,而非硬造。
    brain 用严格 JSON 从【能力+素材双重门控】后的菜单里选;选择理由进台账。

§C  窗口条件策略(用户 4.(2.1) 的 (1)(2),外加补全的同族策略)
    生成"下一个未生成 shot"时,brain 从菜单选一种给生成器搭条件:
    C1 "t2v"            纯文本(没有任何可用锚时的兜底)
    C2 "i2v_keyframe"   本 shot 的 keyframe 当首帧(§B 的产物)
    C3 "ti2v_prev_last" 用户的 (1):上一镜【尾帧】+ 文本 → i2v
    C4 "flf2v_bridge"   上一镜尾帧 → 本 shot keyframe 的首尾帧桥接
                       (用户 (1) 的强化版:两端同时锚定,续接最稳)
    C5 "tiv2v_window"   用户的 (2):上一镜【尾段视频】(+keyframe)+文本 →
                        参考视频通道(seedance-2.0 reference_videos)
    门控:C3-C5 需要上一镜已生成;C4 需要 flf2v 能力 + keyframe;
    C5 需要 ref_video 能力。brain 的 JSON 选择失败 → 确定性兜底
    (可用性优先级 C4 > C5 > C3 > C2 > C1,和修复循环的 INVALID→Router 同款
    "brain 提议失效则确定性接管"模式)。

§D  每镜小循环(用户 4.(2.0)):按条件生成首批候选 → 交给【现有的】
    generate_shot_orchestrated(initial_candidates=...)——评审(VLM 按其
    skill 文件的维度出意见)、缺陷定位(哪几帧/哪段失败:DefectReport +
    物理 verdict 的 frame_range)、Verifier 闸门、brain 修复工具调用,
    全部原样复用,不重写。评审意见+修复动作嵌入台账(reviews / repair_actions
    追加式,构成用户要的"轨迹")。
    ※ 用户把"评审汇总+闸门"合称 verifier;我们架构里是 Summarizer(汇总)+
      Verifier(裁决)两角色,依据见 survey_review_summarizer_2026_07.md。

§E  合成(用户 Final):全部 shot 按时间顺序 ffmpeg concat → 最终视频。
    未收敛的 shot 照样拼入(交付最优可得)但台账诚实标注。

§M  记忆闭环(用户 3.(1)+3.(2)):台账全程更新(R1);收工后
    EpisodeMemory.distill_episode 蒸馏 good/bad 案例(R2);开工时
    guidance_for(prompt) 取历史经验 —— replay 提示可被 brain 直接采纳
    (决策记 via="episode",这就是"记忆可执行化":检索即执行),avoid 表
    注入 prompt 当硬约束(检索即禁止)。

设计决定(用户没说、但必须定并说明白的点):
  • "上一镜"用【最近已生成】而非【已 verified】:哪怕上一镜带遗留缺陷,
    它的尾帧也是时间上唯一正确的续接点(见 storyboard.prev_generated)。
  • 尾段截取:ffmpeg -sseof 取末尾 N 秒;ffmpeg 缺失 → 整段视频当参考
   (诚实降级,记录在 condition 里)。
  • brain 决策三层回退:episode replay 命中(via="episode")→ LLM 严格
    JSON(via="llm")→ 确定性优先级(via="fallback")。每层都记录在台账。
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agents.director import DirectorAgent
from ..agents.screenwriter import ScreenwriterAgent
from ..logging_utils import get_logger
from ..memory.episode_memory import EpisodeMemory
from ..memory.storyboard import StoryboardMemory
from ..models.mllm_backends import _extract_json
from ..pipeline.generate_loop import generate_shot_orchestrated
from ..pipeline.timeline import extract_frame
from ..types import AssetMemory, CandidateClip, ShotSpec

log = get_logger(__name__)

# §C 确定性兜底的优先级(强锚优先;仅在菜单里可用的策略间比较)。
# Q1 多图调研落地后新增两个多图策略:
#   ti2v_prev_plus_keyframe — 上镜尾帧当首帧 + 本镜 keyframe 进
#     reference_images(@Image1 提及)——用户 4.(2.1)(1) "尾帧以及 keyframe"
#     的字面实现:两张图一次调用(seedance-2.0,连续性锚定 + 目标画面引导);
#   multi_image_fusion — [上镜尾帧, 本镜 keyframe(, 身份锚)] 作 images 数组
#     一次融合生成(kling multi-i2v):无指定首帧,画面按全部图片融合。
#   排序依据:硬锚(像素级续接)优先于软锚 —— flf2v_bridge(双端硬锚)>
#   tiv2v_window(尾段运动参考+可选首帧)> ti2v_prev_last(首帧硬锚)>
#   ti2v_prev_plus_keyframe(t2v+refs 软锚)> multi_image_fusion(融合)。
_CONDITION_PRIORITY = ["flf2v_bridge", "tiv2v_window", "ti2v_prev_last",
                       "ti2v_prev_plus_keyframe",
                       "multi_image_fusion", "i2v_keyframe", "t2v"]
# §B 确定性兜底的优先级(用户素材优先于生成 —— 真材实料的外观赢过再生成)
_KEYFRAME_PRIORITY = ["asset_image", "video_extract", "t2i", "none"]


@dataclass
class MovieResult:
    """generate_movie_windowed 的完整回执(全部可审计)。"""

    final_video: Optional[Path]
    storyboard: StoryboardMemory
    shot_results: list = field(default_factory=list)   # SelfImproveResult per shot
    episode_id: str = ""
    decisions: list = field(default_factory=list)      # brain 的 §B/§C 决策流水


# ─────────────────────────────────────────────────────────────────────────
# 小工具
# ─────────────────────────────────────────────────────────────────────────
def _cut_tail(video: Path, seconds: float, out_path: Path) -> Optional[Path]:
    """截取视频末尾 `seconds` 秒(§C5 的窗口素材)。ffmpeg 缺失/失败 → None
    (调用方降级为整段视频当参考,并如实记录)。"""
    if not shutil.which("ffmpeg"):
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-sseof", f"-{max(0.5, seconds):.2f}",
             "-i", str(video), "-c", "copy", str(out_path)],
            capture_output=True, timeout=120,
        )
    except Exception:
        return None
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        return None
    return out_path


def _last_frame(video: Path, out_path: Path) -> Optional[Path]:
    """上一镜尾帧(§C3/§C4 的锚)。复用 timeline.extract_frame(同一解码栈);
    不可解码(mock 文本桩)→ None → 依赖它的策略自动从菜单消失。"""
    return extract_frame(Path(video), 10 ** 9, Path(out_path))


def _brain_pick(llm, kind: str, menu: list[dict], context: dict) -> dict:
    """让 brain 用严格 JSON 从菜单选一项;失败返回 {}(调用方走兜底)。

    和 OrchestratorAgent.decide 同款纪律:只能选菜单里的 name,越界即无效。
    MockLLM 回 "ack:..." 必然解析失败 → 测试/mock 模式全程走确定性兜底,
    不会伪造"brain 决策"。"""
    if llm is None:
        return {}
    prompt = (
        f"You are the window-generation brain. Pick EXACTLY ONE {kind} "
        "strategy from `menu` for the CURRENT shot. Consider the storyboard "
        "(what exists so far), the episode guidance (replay_hints = strategies "
        "that WORKED on similar past tasks — prefer them; avoid = strategies "
        "that FAILED — never pick them for a similar shot).\n\n"
        + json.dumps({"menu": menu, **context}, ensure_ascii=False)
        + '\n\nSTRICT JSON only: {"strategy": "<name from menu>", '
          '"reason": "<one short sentence>"}'
    )
    try:
        data = _extract_json(llm.complete(prompt))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    valid = {m["name"] for m in menu}
    if str(data.get("strategy", "")) not in valid:
        return {}
    return {"strategy": str(data["strategy"]),
            "reason": str(data.get("reason", ""))}


def _decide(llm, kind: str, menu: list[dict], context: dict,
            replay_hint: Optional[str], priority: list[str]) -> dict:
    """三层决策(§M 的可执行记忆就落在这):
    1) episode replay 命中且策略仍在菜单 → 直接采纳,via="episode"
       (长期记忆的检索即执行 —— 不再消耗一次 LLM 推理);
    2) brain LLM 严格 JSON → via="llm";
    3) 确定性优先级兜底 → via="fallback"(菜单非空必有解,循环永不卡死)。"""
    names = {m["name"] for m in menu}
    if replay_hint and replay_hint in names:
        return {"strategy": replay_hint, "via": "episode",
                "reason": f"replaying a verified strategy from a similar past episode"}
    picked = _brain_pick(llm, kind, menu, context)
    if picked:
        return {**picked, "via": "llm"}
    for name in priority:
        if name in names:
            return {"strategy": name, "via": "fallback",
                    "reason": "deterministic priority (brain reply unusable)"}
    return {"strategy": "t2v", "via": "fallback", "reason": "empty menu guard"}


# ─────────────────────────────────────────────────────────────────────────
# §B keyframe 阶段
# ─────────────────────────────────────────────────────────────────────────
def _keyframe_menu(video_gen, asset_memory: Optional[AssetMemory]) -> list[dict]:
    """能力+素材双重门控的 keyframe 策略菜单(和 orchestrator 的工具菜单
    同一哲学:brain 只能选真正可执行的)。"""
    caps = video_gen.capabilities() if video_gen is not None else set()
    menu: list[dict] = []
    if "t2i" in caps and hasattr(video_gen, "text_to_image"):
        menu.append({"name": "t2i",
                     "description": "Generate the keyframe from the shot "
                                    "description (text-to-image)."})
    if asset_memory is not None and (asset_memory.identity_anchors
                                     or asset_memory.style_anchors):
        menu.append({"name": "asset_image",
                     "description": "Use a user-provided image from the asset "
                                    "library as the keyframe (real appearance)."})
    if asset_memory is not None and asset_memory.video_shots:
        menu.append({"name": "video_extract",
                     "description": "Retrieve a user-provided source video and "
                                    "extract a frame as the keyframe."})
    menu.append({"name": "none",
                 "description": "No keyframe — go text-to-video (fallback when "
                                "no material fits)."})
    return menu


def _make_keyframe(strategy: str, entry, video_gen,
                   asset_memory: Optional[AssetMemory], retrieval,
                   out_dir: Path, seed: int) -> Optional[Path]:
    """执行 §B 选中的策略;产不出真图就返回 None(台账保持无 keyframe,
    绝不放占位图冒充)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if strategy == "t2i":
        out = out_dir / f"shot{entry.shot_idx:03d}_kf_t2i.png"
        return Path(video_gen.text_to_image(entry.description, out, seed=seed))
    if strategy == "asset_image" and asset_memory is not None:
        # identity 锚优先(角色长相是一致性的命门),style 锚兜底
        for anchor in list(asset_memory.identity_anchors.values()):
            p = Path(anchor.source)
            if anchor.source and p.exists():
                return p
        for s in asset_memory.style_anchors:
            p = Path(getattr(s, "source", "") or "")
            if p and p.exists():
                return p
        return None
    if strategy == "video_extract" and retrieval is not None \
            and asset_memory is not None:
        shot_ids = retrieval.retrieve_source_shots(query=entry.description)
        for sid in shot_ids:
            shot = asset_memory.video_shots.get(sid)
            if shot is None or not shot.source_video:
                continue
            src = Path(shot.source_video)
            if not src.exists():
                continue
            out = out_dir / f"shot{entry.shot_idx:03d}_kf_extract.png"
            # 取源片段的中间帧(比首帧更能代表片段内容)
            got = extract_frame(src, 10 ** 6, out)
            if got is not None:
                return got
        return None
    return None   # "none"


# ─────────────────────────────────────────────────────────────────────────
# §C 窗口条件策略
# ─────────────────────────────────────────────────────────────────────────
def _condition_menu(entry, prev, video_gen) -> list[dict]:
    """当前 shot 可用的条件策略(存在性+能力双重门控)。"""
    caps = video_gen.capabilities() if video_gen is not None else set()
    has_kf = bool(entry.keyframe_path and Path(entry.keyframe_path).exists())
    has_prev = prev is not None and prev.video_path is not None
    menu = [{"name": "t2v", "description": "Text only — no visual anchor. Use "
             "when nothing else is available or the shot is a hard scene cut."}]
    if has_kf:
        menu.append({"name": "i2v_keyframe",
                     "description": "This shot's own keyframe as the first "
                                    "frame (strong look anchor, no continuity "
                                    "with the previous shot)."})
    if has_prev:
        menu.append({"name": "ti2v_prev_last",
                     "description": "Previous shot's LAST frame as the first "
                                    "frame + text (strongest temporal "
                                    "continuity into this shot)."})
        if has_kf and "flf2v" in caps and hasattr(video_gen, "frame_to_frame"):
            menu.append({"name": "flf2v_bridge",
                         "description": "Bridge: previous shot's last frame → "
                                        "this shot's keyframe (both ends "
                                        "anchored — continuity AND target "
                                        "look)."})
        if "ref_video" in caps:
            menu.append({"name": "tiv2v_window",
                         "description": "Previous shot's TAIL video segment as "
                                        "a motion reference (+ keyframe as "
                                        "first frame if present) + text — the "
                                        "generator SEES the ongoing motion."})
        if has_kf and "ref_images" in caps:
            menu.append({"name": "ti2v_prev_plus_keyframe",
                         "description": "TWO images in ONE call via the t2v "
                                        "reference channel: previous shot's "
                                        "last frame as @Image1 (the moment to "
                                        "continue from) + this shot's keyframe "
                                        "as @Image2 (target composition). SOFT "
                                        "anchoring — composition-level "
                                        "continuity without locking any exact "
                                        "frame; for pixel-exact continuity use "
                                        "ti2v_prev_last or flf2v_bridge."})
        if has_kf and "multi_i2v" in caps and hasattr(video_gen,
                                                      "multi_image_to_video"):
            menu.append({"name": "multi_image_fusion",
                         "description": "FUSE multiple images (previous "
                                        "shot's last frame + this shot's "
                                        "keyframe, ≤4) into one video — no "
                                        "designated first frame; the scene is "
                                        "composed to stay consistent with ALL "
                                        "of them. Use when the shot should "
                                        "blend elements rather than continue "
                                        "pixel-exactly."})
    return menu


def _generate_with_condition(strategy: str, entry, prev, spec: ShotSpec,
                             video_gen, cache_dir: Path, seed: int,
                             fps: int, window_tail_s: float) -> tuple[Path, dict]:
    """执行 §C 策略 → (视频路径, 实际用到的条件记录)。条件记录进台账,
    保证"这镜是怎么搭条件生成的"可审计。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"shot{spec.shot_idx:03d}_w_s{seed}.mp4"
    cond: dict = {"strategy": strategy}
    kf = Path(entry.keyframe_path) if entry.keyframe_path else None

    if strategy == "flf2v_bridge":
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        if last is not None and kf is not None:
            cond.update(first_anchor=str(last), last_anchor=str(kf))
            return Path(video_gen.frame_to_frame(
                prompt=spec.prompt, first_frame=last, last_frame=kf,
                out_path=out, duration=spec.duration, seed=seed)), cond
        strategy = "ti2v_prev_last"      # 尾帧抽不出来 → 逐级降级(如实改写)
        cond = {"strategy": strategy, "degraded_from": "flf2v_bridge"}

    if strategy == "ti2v_prev_plus_keyframe":
        # 用户 4.(2.1)(1) 字面版:上镜尾帧 + 本镜 keyframe,一次调用两张图。
        # 走【t2v + reference_images】通道(官方 schema 只在 t2v 端点验证过
        # refs;i2v 端点 schema 无此字段——不按未验证 schema 硬编码,见
        # docs/research/wavespeed_multi_image_2026_07.md §4)。语义是【软锚】:
        # @Image1 = 续接起点的画面状态,@Image2 = 目标构图/外观——构图级
        # 连续,不保证首帧像素级一致(要像素级用 ti2v_prev_last / flf2v_bridge)。
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        if last is not None and kf is not None:
            cond.update(reference_images=[str(last), str(kf)],
                        anchoring="soft_t2v_refs")
            prompt2 = (spec.prompt + ". Open on the exact scene state shown in "
                       "@Image1 (the previous moment) and move toward the "
                       "composition and look of @Image2 (this shot's target "
                       "keyframe).")
            return Path(video_gen.generate(
                prompt=prompt2, duration=spec.duration, out_path=out, fps=fps,
                reference_images=[last, kf], seed=seed)), cond
        # 尾帧抽不出来 → 还有 keyframe 可锚(门控保证 kf 存在)
        strategy = "i2v_keyframe" if kf is not None else "t2v"
        cond = {"strategy": strategy,
                "degraded_from": "ti2v_prev_plus_keyframe"}

    if strategy == "multi_image_fusion":
        # 多图融合(kling multi-i2v):无指定首帧,images 数组共同约束画面。
        # 图片顺序 = [上镜尾帧, 本镜 keyframe];都拿不到才逐级降级。
        imgs: list = []
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        if last is not None:
            imgs.append(last)
        if kf is not None:
            imgs.append(kf)
        if len(imgs) >= 2:
            cond.update(images=[str(p) for p in imgs])
            return Path(video_gen.multi_image_to_video(
                prompt=spec.prompt, images=imgs, out_path=out,
                duration=spec.duration, seed=seed)), cond
        # 不足两张 → 逐级降级(落到下方对应策略块,degraded_from 保留)
        strategy = ("i2v_keyframe" if kf is not None
                    else "ti2v_prev_last" if last is not None else "t2v")
        cond = {"strategy": strategy, "degraded_from": "multi_image_fusion"}

    if strategy == "tiv2v_window":
        tail = _cut_tail(Path(prev.video_path), window_tail_s,
                         cache_dir / f"shot{spec.shot_idx:03d}_prev_tail.mp4")
        ref = tail if tail is not None else Path(prev.video_path)
        cond.update(reference_video=str(ref),
                    tail_seconds=(window_tail_s if tail else None),
                    first_frame=str(kf) if kf else None)
        return Path(video_gen.generate(
            prompt=spec.prompt, duration=spec.duration, out_path=out, fps=fps,
            first_frame=kf, seed=seed, reference_video=ref)), cond

    if strategy == "ti2v_prev_last":
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        if last is not None:
            cond.update(first_frame=str(last))
            return Path(video_gen.generate(
                prompt=spec.prompt, duration=spec.duration, out_path=out,
                fps=fps, first_frame=last, seed=seed)), cond
        strategy = "i2v_keyframe" if kf is not None else "t2v"
        cond = {"strategy": strategy, "degraded_from": "ti2v_prev_last"}

    if strategy == "i2v_keyframe" and kf is not None:
        cond.update(first_frame=str(kf))
        return Path(video_gen.generate(
            prompt=spec.prompt, duration=spec.duration, out_path=out,
            fps=fps, first_frame=kf, seed=seed)), cond

    cond = {"strategy": "t2v", **({"degraded_from": cond.get("degraded_from")}
                                  if cond.get("degraded_from") else {})}
    return Path(video_gen.generate(
        prompt=spec.prompt, duration=spec.duration, out_path=out,
        fps=fps, seed=seed)), cond


# ─────────────────────────────────────────────────────────────────────────
# 主入口 —— 大循环
# ─────────────────────────────────────────────────────────────────────────
def generate_movie_windowed(
    user_prompt: str,
    *,
    board,                              # ReviewBoard(§D 复用)
    generator,                          # GeneratorAgent(内层修复循环用)
    refiner,
    verifier,
    orchestrator,                       # OrchestratorAgent(内层 brain;其 llm 兼任窗口 brain)
    cache_dir: Path,
    asset_memory: Optional[AssetMemory] = None,
    retrieval=None,
    screenwriter: Optional[ScreenwriterAgent] = None,
    director: Optional[DirectorAgent] = None,
    tournament=None,
    skill_library=None,
    lesson_library=None,
    image_edit=None,
    episode_memory: Optional[EpisodeMemory] = None,
    summarizer=None,
    llm=None,                           # 窗口 brain 的 LLM;缺省用 orchestrator.llm
    fps: int = 8,
    n_candidates: int = 2,
    max_turns: int = 4,
    window_tail_s: float = 2.0,         # §C5 尾段窗口长度(秒)
) -> MovieResult:
    """窗口式全片生成:§A playwriting → §B keyframe → §C+§D 逐镜窗口循环
    → §E 合成 → §M episode 蒸馏。全程读写 StoryboardMemory(R1)。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    asset_memory = asset_memory or AssetMemory()
    llm = llm or getattr(orchestrator, "llm", None)
    video_gen = generator.video_gen
    decisions: list[dict] = []

    # §M 开工:检索长期记忆的开工简报(没有 episode_memory 就是空简报)
    guidance = (episode_memory.guidance_for(user_prompt)
                if episode_memory is not None
                else {"replay_hints": [], "avoid": [], "n_episodes_matched": 0})
    # label → 历史上被 Verifier 接受过的策略(可直接采纳,via="episode")
    replay_kf = {h["label"]: h["keyframe_strategy"]
                 for h in guidance["replay_hints"] if h.get("converged")}
    replay_cond = {h["label"]: h["condition_strategy"]
                   for h in guidance["replay_hints"] if h.get("converged")}
    if guidance["n_episodes_matched"]:
        log.info("window: episode guidance matched %d past episode(s); "
                 "%d replay hints, %d avoid entries",
                 guidance["n_episodes_matched"],
                 len(guidance["replay_hints"]), len(guidance["avoid"]))

    # ── §A playwriting:prompt → outline → specs → 台账 ────────────────────
    screenwriter = screenwriter or ScreenwriterAgent()
    director = director or DirectorAgent()
    outline = screenwriter.run(user_prompt, asset_memory)
    specs = director.run(outline, asset_memory, lesson_library)
    storyboard = StoryboardMemory.from_outline(
        outline, path=cache_dir / "storyboard.json")
    log.info("window: playwriting done — %s", storyboard.summary())

    # ── §B keyframe 阶段(逐 shot:brain 选策略 → 执行 → 台账)──────────────
    kf_dir = cache_dir / "keyframes"
    for entry, spec in zip(storyboard.entries, specs):
        menu = _keyframe_menu(video_gen, asset_memory)
        d = _decide(
            llm, "keyframe", menu,
            {"shot": entry.to_brain_line(),
             "storyboard": storyboard.to_brain_json(),
             "episode_guidance": guidance},
            replay_hint=replay_kf.get(entry.label),
            priority=_KEYFRAME_PRIORITY,
        )
        decisions.append({"stage": "keyframe", "label": entry.label, **d})
        kf = None
        if d["strategy"] != "none":
            try:
                kf = _make_keyframe(d["strategy"], entry, video_gen,
                                    asset_memory, retrieval, kf_dir,
                                    seed=entry.shot_idx)
            except Exception as exc:      # t2i API 失败等 → 无 keyframe,如实记录
                log.info("window: keyframe strategy %s failed for %s: %s",
                         d["strategy"], entry.label, exc)
        if kf is not None:
            storyboard.set_keyframe(entry.shot_idx, kf, d["strategy"])
        log.info("window: %s keyframe → %s (via=%s%s)", entry.label,
                 d["strategy"], d["via"], "" if kf else "; no image produced")

    # ── §C+§D 大循环:逐镜窗口生成 + 小循环评审修复 ─────────────────────────
    shot_results = []
    while True:
        entry = storyboard.next_pending()
        if entry is None:
            break
        spec = specs[entry.shot_idx]
        prev = storyboard.prev_generated(entry.shot_idx)

        # §C brain 选条件策略(episode → llm → 兜底 三层)
        menu = _condition_menu(entry, prev, video_gen)
        d = _decide(
            llm, "generation-condition", menu,
            {"shot": entry.to_brain_line(),
             "prev_shot": prev.to_brain_line() if prev else None,
             "storyboard": storyboard.to_brain_json(),
             "episode_guidance": guidance},
            replay_hint=replay_cond.get(entry.label),
            priority=_CONDITION_PRIORITY,
        )
        decisions.append({"stage": "condition", "label": entry.label, **d})
        log.info("window: %s condition → %s (via=%s) %s",
                 entry.label, d["strategy"], d["via"], d.get("reason", ""))

        # 按条件生成首批候选(不同 seed;条件相同)。每个 seed 的实际条件
        # 单独记账(per_seed):策略在执行中降级/崩溃时,那个 seed 的记录
        # 必须如实写 degraded_from —— 台账绝不把降级伪装成 brain 的决定。
        shot_dir = cache_dir / f"shot{entry.shot_idx:03d}"
        initial: list[CandidateClip] = []
        seed_conds: list[dict] = []
        # 子循环里 keyframe_edit 工具需要 clip.keyframes;窗口候选挂上本 shot
        # 真实存在的关键帧(比生成器的占位帧更真),没有就空列表(该工具在
        # 菜单中仍在,执行时诚实 no-op)。
        cand_keyframes = ([Path(entry.keyframe_path)]
                          if entry.keyframe_path
                          and Path(entry.keyframe_path).exists() else [])
        for s in range(max(1, n_candidates)):
            try:
                video_path, cond = _generate_with_condition(
                    d["strategy"], entry, prev, spec, video_gen,
                    shot_dir, seed=s, fps=fps, window_tail_s=window_tail_s)
            except Exception as exc:
                log.info("window: conditioned generation failed (%s): %s — "
                         "falling back to plain t2v for this seed",
                         d["strategy"], exc)
                video_path, cond = _generate_with_condition(
                    "t2v", entry, prev, spec, video_gen, shot_dir,
                    seed=s, fps=fps, window_tail_s=window_tail_s)
                # 异常降级必须留痕:没有这两行,台账会谎称 brain 主动选了 t2v
                cond["degraded_from"] = d["strategy"]
                cond["degraded_reason"] = f"exception: {exc}"[:200]
            cond["seed"] = s
            seed_conds.append(cond)
            clip = CandidateClip(shot_idx=spec.shot_idx,
                                 video_path=Path(video_path), revision=0)
            clip.keyframes = list(cand_keyframes)
            initial.append(clip)

        # §D 小循环:评审(VLM skill 维度)→ 汇总 → 定位(帧/段)→ brain 修复
        # → Verifier 闸门 —— 全部在现有 generate_shot_orchestrated 内完成。
        res = generate_shot_orchestrated(
            spec, board=board, generator=generator, refiner=refiner,
            verifier=verifier, cache_dir=shot_dir, orchestrator=orchestrator,
            asset_memory=asset_memory, lesson_library=lesson_library,
            image_edit=image_edit, tournament=tournament, retrieval=retrieval,
            skill_library=skill_library, fps=fps, n_candidates=n_candidates,
            max_turns=max_turns, summarizer=summarizer,
            initial_candidates=initial,
        )
        shot_results.append(res)
        best = res.clip

        # 台账条件按【初选胜出者】归因(res.initial_winner):n_candidates>1 时
        # 各 seed 的条件可能不同(某个 seed 异常降级了),最终出镜的是锦标赛
        # 赢家 —— 记它实际用的条件,而不是"最后一个 seed 恰好用的条件"。
        winner_cond = next(
            (c for clip_, c in zip(initial, seed_conds)
             if str(clip_.video_path) == res.initial_winner),
            seed_conds[-1] if seed_conds else {"strategy": d["strategy"]},
        )
        cond_used = dict(winner_cond)
        cond_used["decided_strategy"] = d["strategy"]   # brain 的原始决定
        cond_used["decided_via"] = d["via"]
        distinct = {json.dumps({k: v for k, v in c.items() if k != "seed"},
                               sort_keys=True) for c in seed_conds}
        if len(distinct) > 1:
            cond_used["per_seed"] = seed_conds          # 有分歧才展开全量流水
        storyboard.set_condition(entry.shot_idx, cond_used)

        # 评审轨迹 + 修复动作嵌入台账(§D "意见嵌入轨迹")
        storyboard.add_review(entry.shot_idx, {
            "revision": best.revision,
            "weighted_total": best.metric_scores.get("weighted_total", 0.0),
            "n_failed": len(best.checklist.failed_items),
            "physics_verdicts": [
                {"entity": v.entity, "mode": v.mode.value,
                 "frame_range": list(v.frame_range),
                 "severity": round(float(v.severity), 3), "source": v.source}
                for v in best.physics_verdicts
            ],
            "brief_headline": next(
                (a.get("brief_headline", "") for a in reversed(res.actions)
                 if a.get("brief_headline")), ""),
            "converged": res.converged,
        })
        storyboard.set_result(entry.shot_idx, best.video_path,
                              converged=res.converged,
                              repair_actions=res.actions)
        log.info("window: %s done — %s (score=%.4f, %d repair turns)",
                 entry.label,
                 "verified" if res.converged else "generated_with_defects",
                 best.metric_scores.get("weighted_total", 0.0),
                 len(res.actions))

    # ── §E 合成:时间顺序 concat ────────────────────────────────────────────
    final: Optional[Path] = None
    clips = [e.video_path for e in storyboard.entries if e.video_path]
    if clips:
        try:
            from ..tools.video_concat import VideoConcatTool

            final = VideoConcatTool().run(clips, cache_dir / "movie.mp4")
        except Exception as exc:          # ffmpeg 缺失等 → 不合成,单镜可用
            log.info("window: merge degraded (%s) — per-shot clips remain", exc)

    # ── §M 收工:蒸馏 episode(good/bad 由客观收敛状态判定)────────────────
    episode_id = ""
    if episode_memory is not None:
        rec = episode_memory.distill_episode(
            user_prompt, storyboard, final_video=str(final or ""))
        episode_id = rec.episode_id
        log.info("window: episode distilled — %s (%s, %d replay rows, "
                 "%d avoid rows)", rec.episode_id, rec.outcome,
                 len(rec.replay), len(rec.avoid))

    return MovieResult(final_video=final, storyboard=storyboard,
                       shot_results=shot_results, episode_id=episode_id,
                       decisions=decisions)
