"""Window-based movie generation — the OUTER brain loop (需求 R3 / 用户第 4 点).

═══════════════════════════════════════════════════════════════════════════
用户需求的标准化(每一条在代码里的落点都标了 §号,一条不落):

§A  playwriting(用户 4.(1) 前半)
    用户 prompt → 按时间顺序的全部 shot 文本描述(Screenwriter + Director,
    复用现有 agent)→ 建 StoryboardMemory 台账(需求 R1:brain 维护的
    按时间顺序、可持续更新的 keyframe/video+描述 列表)。

§B' Image Plan 阶段(升级版 keyframe;用户追加需求 2026-07-13)
    brain 逐 shot 一次决定【数量(0/1/2)+ 角色 + 来源】:
      单图角色 = first_frame(→ i2v)或 reference(→ 参考通道模型);
      双图角色 = first_last(→ 首尾帧模型)或 reference_pair(→ kling-o1)。
    角色锁死后续视频模型族(_condition_menu 按角色门控,杜绝错配);
    来源逐张选 t2i / asset_image / video_extract,允许混搭(Q-B);
    素材检索按描述关键词重叠打分(Q-D:用户描述 > VLM caption > 文件名,
    ensure_asset_descriptions 负责 VLM 回填);产不出的图丢弃并把计划
    如实降级(pair→single→none,plan_degraded_from 留痕)。

§C  窗口条件策略(9 个;菜单 = Image Plan 角色 × 上镜存在性 × 能力)
    自有图策略:i2v_keyframe(首帧角色图)/ flf2v_own_pair(自有首尾双图)
    / t2v_own_refs(参考角色图,seedance @refs,无需上镜);
    上镜锚定策略:ti2v_prev_last(尾帧当首帧)/ flf2v_bridge(尾帧→自有图,
    图被改用作收场锚,技能里讲明)/ tiv2v_window(尾段视频参考)/
    ti2v_prev_plus_keyframe(尾帧+自有图全走 t2v refs 软锚)/
    multi_image_fusion(kling-o1:[尾帧?]+自有图,可再带尾段 video);
    兜底:t2v。brain 输出语义字段(strategy + 角色化 video_prompt +
    use_prev_tail_video,Q-A 裁决),机械字段执行器补齐;选择失败走确定性
    优先级(硬锚 > 软锚),循环永不卡死。

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
_CONDITION_PRIORITY = ["flf2v_own_pair", "flf2v_bridge", "tiv2v_window",
                       "ti2v_prev_last", "ti2v_prev_plus_keyframe",
                       "t2v_own_refs", "multi_image_fusion",
                       "i2v_keyframe", "t2v"]
# §B 确定性兜底的优先级(用户素材优先于生成 —— 真材实料的外观赢过再生成)
_KEYFRAME_PRIORITY = ["asset_image", "video_extract", "t2i", "none"]
# §B' Image Plan 兜底优先级:有上镜时单首帧最稳;素材/能力不足逐级退到 none。
_PLAN_PRIORITY = ["single_first_frame", "pair_first_last", "single_reference",
                  "pair_reference", "none"]


# ─────────────────────────────────────────────────────────────────────────
# 素材检索(Q-D 裁决:用户描述 > VLM caption > 文件名,逐级诚实降级)
# ─────────────────────────────────────────────────────────────────────────
def _asset_label(kind: str, name: str, description: str, path: str) -> str:
    """一个素材的检索文本:优先用户给的 description,没有就用文件名
    (VLM caption 由 ensure_asset_descriptions 在入库后回填 description)。"""
    desc = (description or "").strip()
    if desc:
        return f"{kind}: {name or ''} {desc}"
    stem = Path(path).stem.replace("_", " ").replace("-", " ") if path else ""
    return f"{kind}: {name or ''} {stem}"


def _asset_catalog(asset_memory: Optional[AssetMemory]) -> list[dict]:
    """全部图片素材的目录 [{kind, name, label, path}](路径存在的才算)。
    brain 的 Image Plan 决策会看到它 —— 素材长什么样、有多少,一目了然。"""
    out: list[dict] = []
    if asset_memory is None:
        return out
    for a in asset_memory.identity_anchors.values():
        p = Path(a.source or "")
        if a.source and p.exists():
            out.append({"kind": "identity", "name": a.name or a.identity_id,
                        "label": _asset_label("identity", a.name or "",
                                              a.description, a.source),
                        "path": str(p)})
    for s_ in asset_memory.style_anchors:
        p = Path(getattr(s_, "source", "") or "")
        if p and p.exists():
            out.append({"kind": "style", "name": getattr(s_, "style_id", ""),
                        "label": _asset_label("style",
                                              getattr(s_, "style_id", ""),
                                              getattr(s_, "description", ""),
                                              str(p)),
                        "path": str(p)})
    return out


def _retrieve_asset_image(query: str, asset_memory: Optional[AssetMemory]
                          ) -> Optional[Path]:
    """按关键词重叠给【全部】图片素材打分,取最高分(替代旧的"拿第一张")。
    确定性、可复现;0 重叠时退回第一张存在的图(单素材场景保持旧行为)。
    CLIP 向量检索登记在 TOOL_LIBRARY 缺口台账,本轮不做。"""
    catalog = _asset_catalog(asset_memory)
    if not catalog:
        return None
    q = {w for w in re_words(query) if len(w) > 1}
    best, best_score = None, -1.0
    for item in catalog:
        toks = {w for w in re_words(item["label"]) if len(w) > 1}
        score = len(q & toks) / max(1, len(q | toks)) if q else 0.0
        if score > best_score:
            best, best_score = item, score
    return Path(best["path"]) if best else None


def re_words(text: str) -> list[str]:
    import re

    return [w.lower() for w in re.findall(r"[a-zA-Z一-鿿0-9]+", text or "")]


def ensure_asset_descriptions(asset_memory: Optional[AssetMemory],
                              mllm=None) -> int:
    """Q-D 打标链的 VLM 中环:给【没有用户描述】的图片素材补一句 VLM
    caption(写回 description 字段)。mock/无 VLM → caption_image 返回 ""
    → 不写(诚实降级到文件名);返回补标数量。"""
    if asset_memory is None or mllm is None:
        return 0
    n = 0
    targets = list(asset_memory.identity_anchors.values()) \
        + list(asset_memory.style_anchors)
    for a in targets:
        desc = (getattr(a, "description", "") or "").strip()
        src = getattr(a, "source", "") or ""
        if desc or not src or not Path(src).exists():
            continue
        cap = ""
        try:
            cap = (mllm.caption_image(src) or "").strip()
        except Exception as exc:
            log.warning("asset caption failed for %s: %r", Path(src).name, exc)
        if cap:
            try:
                a.description = cap
                n += 1
            except AttributeError:
                pass                      # StyleRef 若无该字段则跳过(不硬塞)
    if n:
        log.info("asset labeling: %d asset(s) captioned by the VLM "
                 "(user description > VLM caption > filename)", n)
    return n


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


# brain 决策 JSON 的【语义附加字段】白名单(Q-A 裁决:brain 只出语义字段,
# aspect_ratio/duration/keep_original_sound/上传 URL 等机械字段一律由执行器
# 确定性补齐 —— LLM 永远不碰机械字段,payload 格式不会错)。
#   images              — Image Plan 里每张图的 {source, description}
#   video_prompt        — 按图片角色写好的视频生成 prompt(kling 用
#                         "reference image N",seedance 用 "@ImageN")
#   use_prev_tail_video — 参考类策略是否同请求带上镜尾段视频(kling-o1)
_EXTRA_FIELDS = ("images", "video_prompt", "use_prev_tail_video")


def _brain_pick(llm, kind: str, menu: list[dict], context: dict) -> dict:
    """让 brain 用严格 JSON 从菜单选一项;失败返回 {}(调用方走兜底)。

    和 OrchestratorAgent.decide 同款纪律:只能选菜单里的 name,越界即无效。
    语义附加字段(_EXTRA_FIELDS)做轻校验后透传;机械字段即使 brain 多嘴
    也被丢弃。MockLLM 回 "ack:..." 必然解析失败 → 测试/mock 模式全程走
    确定性兜底,不会伪造"brain 决策"。"""
    if llm is None:
        return {}
    prompt = (
        f"You are the window-generation brain. Pick EXACTLY ONE {kind} "
        "strategy from `menu` for the CURRENT shot. Consider the storyboard "
        "(what exists so far), the asset_catalog (user-provided images you "
        "can retrieve), and the episode guidance (replay_hints = strategies "
        "that WORKED on similar past tasks — prefer them; avoid = strategies "
        "that FAILED — never pick them for a similar shot).\n\n"
        + json.dumps({"menu": menu, **context}, ensure_ascii=False)
        + '\n\nSTRICT JSON only: {"strategy": "<name from menu>", '
          '"reason": "<one short sentence>", ... optional fields the menu '
          "entry's description asks for (images / video_prompt / "
          "use_prev_tail_video)}"
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
    out = {"strategy": str(data["strategy"]),
           "reason": str(data.get("reason", ""))}
    # 语义附加字段:轻校验透传(类型不对就丢 —— 执行器有确定性默认)。
    if isinstance(data.get("images"), list):
        imgs = []
        for im in data["images"][:2]:            # 暂定最多两张(用户设定)
            if isinstance(im, dict):
                imgs.append({"source": str(im.get("source", "")),
                             "description": str(im.get("description", ""))})
        if imgs:
            out["images"] = imgs
    if isinstance(data.get("video_prompt"), str) and data["video_prompt"].strip():
        out["video_prompt"] = data["video_prompt"].strip()
    if isinstance(data.get("use_prev_tail_video"), bool):
        out["use_prev_tail_video"] = data["use_prev_tail_video"]
    return out


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
# §B' Image Plan 阶段(升级版 keyframe:数量 + 角色 + 来源 一次决策)
# 用户设定(锁死的角色→模型族映射):
#   single_first_frame → 该图当首帧 → i2v(ti2v)
#   single_reference   → 该图当参考 → seedance t2v+refs / kling-video-o1
#   pair_first_last    → 首尾帧    → flf2v 族(seedance i2v image+last_image)
#   pair_reference     → 双参考    → kling-video-o1(images 数组)
# 角色决定后续条件菜单(_condition_menu 按角色门控),杜绝
# "图按首尾帧生成、却被当参考用"的错配。
# ─────────────────────────────────────────────────────────────────────────
_PLAN_ROLES = {
    "none": [],
    "single_first_frame": ["first_frame"],
    "single_reference": ["reference"],
    "pair_first_last": ["first", "last"],
    "pair_reference": ["reference", "reference"],
}


def _image_plan_menu(video_gen, asset_memory: Optional[AssetMemory]) -> list[dict]:
    """Image Plan 菜单(能力+素材双重门控)。来源(t2i/素材/抽帧)在计划的
    images 字段里逐张选(Q-B:允许混搭);这里只门控"计划本身可执行":
    任何一种来源可用 → 单图/双图计划可选;参考类计划还需要参考通道能力。"""
    caps = video_gen.capabilities() if video_gen is not None else set()
    has_t2i = "t2i" in caps and hasattr(video_gen, "text_to_image")
    has_assets = bool(_asset_catalog(asset_memory))
    has_src_videos = bool(asset_memory is not None and asset_memory.video_shots)
    any_source = has_t2i or has_assets or has_src_videos
    menu = [{"name": "none",
             "description": "No images — text-only / previous-shot-anchored "
                            "generation. Use when no source fits or the shot "
                            "needs no visual anchor of its own."}]
    if not any_source:
        return menu
    src_note = ("Per-image `images` field: [{source: t2i|asset_image|"
                "video_extract, description: <t2i prompt or retrieval "
                "query>}] — sources MAY mix (e.g. one user asset + one t2i).")
    menu.append({"name": "single_first_frame",
                 "description": "ONE image used as the FIRST FRAME (video via "
                                "i2v). The shot opens pixel-exactly on it. "
                                + src_note})
    if "ref_images" in caps or "multi_i2v" in caps:
        menu.append({"name": "single_reference",
                     "description": "ONE image used as a REFERENCE (identity/"
                                    "object/scene consistency; video via a "
                                    "reference-capable model). The shot is NOT "
                                    "forced to open on it. " + src_note})
    if "flf2v" in caps and hasattr(video_gen, "frame_to_frame"):
        menu.append({"name": "pair_first_last",
                     "description": "TWO images used as FIRST + LAST frame "
                                    "(video via a first/last-frame model). "
                                    "Give TWO `images` entries: opening frame "
                                    "then closing frame. " + src_note})
    if "multi_i2v" in caps or "ref_images" in caps:
        menu.append({"name": "pair_reference",
                     "description": "TWO images used as REFERENCES (e.g. two "
                                    "characters / character + scene; video via "
                                    "kling-video-o1 images array or seedance "
                                    "t2v @refs). " + src_note})
    return menu


def _execute_image_plan(decision: dict, entry, video_gen,
                        asset_memory: Optional[AssetMemory], retrieval,
                        out_dir: Path) -> tuple[str, list, str]:
    """执行 Image Plan → (最终 plan, images 列表, degraded_from)。

    每张图独立按来源产出(Q-B 混搭);产不出的图【丢弃并降级计划】——
    pair 剩一张 → 对应的 single 计划;全没 → none。降级必写 degraded_from
    (台账诚实:brain 的原始决定和实际产物分开记)。"""
    plan = decision["strategy"]
    roles = _PLAN_ROLES.get(plan, [])
    specs = list(decision.get("images") or [])
    # brain 没给逐张 spec(fallback/episode 层)→ 确定性默认:来源按
    # 素材>抽帧>t2i 优先,描述用分镜描述(尾帧槽位加收尾措辞)。
    while len(specs) < len(roles):
        idx = len(specs)
        specs.append({"source": "", "description":
                      entry.description + (" — the closing frame of this shot"
                                           if roles[idx] == "last" else "")})
    produced: list = []
    for i, role in enumerate(roles):
        spec_i = specs[i] if i < len(specs) else {}
        src = spec_i.get("source") or _default_source(video_gen, asset_memory)
        query = spec_i.get("description") or entry.description
        img = None
        try:
            img = _make_keyframe(src, entry, video_gen, asset_memory,
                                 retrieval, out_dir, seed=entry.shot_idx * 2 + i,
                                 query=query, slot=i)
        except Exception as exc:
            log.info("image plan: slot %d (%s via %s) failed: %s",
                     i, role, src, exc)
        if img is not None:
            produced.append({"path": str(img), "role": role, "source": src,
                             "description": query})
        else:
            log.info("image plan: slot %d (%s) produced no image — dropped",
                     i, role)
    if len(produced) == len(roles):
        return plan, produced, ""
    # 诚实降级:按剩余图的角色改写计划
    if not produced:
        return "none", [], plan
    only = produced[0]
    if only["role"] in ("first", "first_frame"):
        only = {**only, "role": "first_frame"}
        return "single_first_frame", [only], plan
    if only["role"] == "last":
        # 只剩收尾帧:当首帧用是撒谎;如实转参考(参考通道在才有意义,
        # 条件菜单会按角色门控,没有参考路线时它自然不被消费)。
        only = {**only, "role": "reference"}
        return "single_reference", [only], plan
    return "single_reference", [only], plan


def _default_source(video_gen, asset_memory) -> str:
    """确定性来源兜底:真材实料优先(素材 > 抽帧 > t2i)。"""
    if _asset_catalog(asset_memory):
        return "asset_image"
    if asset_memory is not None and asset_memory.video_shots:
        return "video_extract"
    caps = video_gen.capabilities() if video_gen is not None else set()
    if "t2i" in caps and hasattr(video_gen, "text_to_image"):
        return "t2i"
    return "t2i"


# ─────────────────────────────────────────────────────────────────────────
# §B keyframe 阶段(旧接口;Image Plan 的来源分发复用 _make_keyframe)
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
                   out_dir: Path, seed: int, query: str = "",
                   slot: int = 0) -> Optional[Path]:
    """按来源产出一张图;产不出真图就返回 None(绝不放占位图冒充)。
    `query`(默认分镜描述)驱动检索类来源:素材图按关键词重叠打分选
    (_retrieve_asset_image,替代旧的"拿第一张"),源视频同理。
    `slot` 区分同一 shot 的多张图(Image Plan 双图时文件不互撞)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    query = query or entry.description
    if strategy == "t2i":
        out = out_dir / f"shot{entry.shot_idx:03d}_kf{slot}_t2i.png"
        return Path(video_gen.text_to_image(query, out, seed=seed))
    if strategy == "asset_image":
        # 按 shot 描述/检索词给全部图片素材打分取最优(Q-D 标签链)。
        return _retrieve_asset_image(query, asset_memory)
    if strategy == "video_extract" and retrieval is not None \
            and asset_memory is not None:
        shot_ids = retrieval.retrieve_source_shots(query=query)
        for sid in shot_ids:
            shot = asset_memory.video_shots.get(sid)
            if shot is None or not shot.source_video:
                continue
            src = Path(shot.source_video)
            if not src.exists():
                continue
            out = out_dir / f"shot{entry.shot_idx:03d}_kf{slot}_extract.png"
            # 取源片段的中间帧(比首帧更能代表片段内容)
            got = extract_frame(src, 10 ** 6, out)
            if got is not None:
                return got
        return None
    return None   # "none"


# ─────────────────────────────────────────────────────────────────────────
# §C 窗口条件策略(菜单由 Image Plan 的角色 + 上镜存在性 + 能力共同门控)
# ─────────────────────────────────────────────────────────────────────────
def _entry_images(entry) -> tuple[Optional[Path], list[Path],
                                  Optional[Path], Optional[Path]]:
    """从台账条目解出四类图:(首帧图, 参考图列表, 首帧槽, 尾帧槽)。

    兼容旧数据:entry.images 为空但 keyframe_path 存在 → keyframe 同时充当
    "可当首帧的图"和"可当参考的图"(Image Plan 之前的隐式语义,原有测试
    与 episode 依赖它)。"""
    first_frame = None
    refs: list[Path] = []
    pair_first = pair_last = None
    getter = getattr(entry, "images_by_role", None)
    if getter is not None and entry.images:
        ff = getter("first_frame")
        first_frame = Path(ff[0]["path"]) if ff else None
        refs = [Path(im["path"]) for im in getter("reference")]
        f_ = getter("first")
        l_ = getter("last")
        pair_first = Path(f_[0]["path"]) if f_ else None
        pair_last = Path(l_[0]["path"]) if l_ else None
    elif entry.keyframe_path and Path(entry.keyframe_path).exists():
        kf = Path(entry.keyframe_path)
        first_frame = kf
        refs = [kf]
    return first_frame, refs, pair_first, pair_last


def _condition_menu(entry, prev, video_gen) -> list[dict]:
    """当前 shot 可用的条件策略(Image Plan 角色 + 存在性 + 能力三重门控)。"""
    caps = video_gen.capabilities() if video_gen is not None else set()
    ff, refs, pf, pl = _entry_images(entry)
    has_kf = ff is not None
    has_prev = prev is not None and prev.video_path is not None
    menu = [{"name": "t2v", "description": "Text only — no visual anchor. Use "
             "when nothing else is available or the shot is a hard scene cut."}]
    if has_kf:
        menu.append({"name": "i2v_keyframe",
                     "description": "This shot's own keyframe as the first "
                                    "frame (strong look anchor, no continuity "
                                    "with the previous shot)."})
    # pair_first_last 计划:自己的首尾双图 → 专属 flf2v 路线(最强自锚)
    if (pf is not None and pl is not None and "flf2v" in caps
            and hasattr(video_gen, "frame_to_frame")):
        menu.append({"name": "flf2v_own_pair",
                     "description": "This shot's OWN planned first+last frame "
                                    "pair drives a first/last-frame model — "
                                    "the shot opens on image 1 and closes on "
                                    "image 2 exactly. Provide `video_prompt` "
                                    "describing the motion BETWEEN the two "
                                    "frames."})
    # reference 角色图(无需上镜也能用)→ seedance t2v @refs 路线
    if refs and "ref_images" in caps:
        menu.append({"name": "t2v_own_refs",
                     "description": "This shot's planned REFERENCE image(s) "
                                    "ride the seedance t2v reference channel. "
                                    "Write `video_prompt` mentioning them as "
                                    "@Image1(, @Image2) with their roles (e.g. "
                                    "'@Image1 is the female character…'). Soft "
                                    "conditioning; no frame is pixel-locked."})
    if has_prev:
        menu.append({"name": "ti2v_prev_last",
                     "description": "Previous shot's LAST frame as the first "
                                    "frame + text (strongest temporal "
                                    "continuity into this shot)."})
        if has_kf and "flf2v" in caps and hasattr(video_gen, "frame_to_frame"):
            menu.append({"name": "flf2v_bridge",
                         "description": "Bridge: previous shot's last frame → "
                                        "this shot's first-frame image "
                                        "REPURPOSED as the CLOSING anchor "
                                        "(continuity AND the shot ARRIVES at "
                                        "your image). Pick only when arriving "
                                        "at the image is the intent."})
        if "ref_video" in caps:
            menu.append({"name": "tiv2v_window",
                         "description": "Previous shot's TAIL video segment as "
                                        "a motion reference (+ own first-frame "
                                        "image as the first frame if planned) "
                                        "+ text — the generator SEES the "
                                        "ongoing motion."})
        if (has_kf or refs) and "ref_images" in caps:
            menu.append({"name": "ti2v_prev_plus_keyframe",
                         "description": "t2v reference channel with the "
                                        "previous shot's last frame as @Image1 "
                                        "(the moment to continue from) + this "
                                        "shot's image(s) as @Image2(…) (target "
                                        "look). SOFT anchoring — for "
                                        "pixel-exact continuity use "
                                        "ti2v_prev_last or flf2v_bridge. Write "
                                        "`video_prompt` with the @ImageN "
                                        "mentions."})
        if (has_kf or refs) and "multi_i2v" in caps \
                and hasattr(video_gen, "multi_image_to_video"):
            menu.append({"name": "multi_image_fusion",
                         "description": "kling-video-o1 reference route: FUSE "
                                        "[previous shot's last frame + this "
                                        "shot's image(s)] (≤7) into one video "
                                        "— no designated first frame. Write "
                                        "`video_prompt` referring to them as "
                                        "'reference image 1/2…' with roles; "
                                        "set use_prev_tail_video=true to ALSO "
                                        "ride the previous shot's tail video "
                                        "(image cap drops to 4)."})
    elif refs and "multi_i2v" in caps \
            and hasattr(video_gen, "multi_image_to_video") and len(refs) >= 2:
        # 无上镜(如第一镜)但计划了双参考 → kling 融合仍可用
        menu.append({"name": "multi_image_fusion",
                     "description": "kling-video-o1 reference route over this "
                                    "shot's OWN reference pair — compose one "
                                    "video consistent with both images. Write "
                                    "`video_prompt` as 'reference image 1 is "
                                    "…, reference image 2 is …'."})
    return menu


def _generate_with_condition(strategy: str, entry, prev, spec: ShotSpec,
                             video_gen, cache_dir: Path, seed: int,
                             fps: int, window_tail_s: float,
                             brain_prompt: str = "",
                             use_prev_tail_video: bool = False
                             ) -> tuple[Path, dict]:
    """执行 §C 策略 → (视频路径, 实际用到的条件记录)。条件记录进台账,
    保证"这镜是怎么搭条件生成的"可审计。

    Q-A 分工:`brain_prompt` 是 brain 按图片角色写好的视频 prompt(kling 用
    "reference image N",seedance 用 "@ImageN"),优先使用;没给(fallback/
    episode 层)则用确定性模板。机械字段(时长/比例/上传 URL/keep_original_
    sound)全部由执行器与后端补齐,LLM 不碰。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"shot{spec.shot_idx:03d}_w_s{seed}.mp4"
    cond: dict = {"strategy": strategy}
    ff, refs, pf, pl = _entry_images(entry)
    kf = ff                                   # 首帧角色图(兼容旧 keyframe)
    if brain_prompt:
        cond["brain_prompt"] = True           # prompt 出自 brain(语义字段)

    if strategy == "flf2v_own_pair":
        # Image Plan pair_first_last 的专属路线:自己的首尾双图,像素级
        # 锁定开场和收场;prompt 描述两帧之间的运动。
        if pf is not None and pl is not None:
            cond.update(first_anchor=str(pf), last_anchor=str(pl))
            return Path(video_gen.frame_to_frame(
                prompt=brain_prompt or (spec.prompt + " — one continuous "
                                        "motion from the first frame to the "
                                        "last frame"),
                first_frame=pf, last_frame=pl,
                out_path=out, duration=spec.duration, seed=seed)), cond
        # 双图缺角(执行期文件丢失等)→ 剩哪张用哪张
        strategy = "i2v_keyframe" if (pf or kf) else "t2v"
        kf = pf or kf
        cond = {"strategy": strategy, "degraded_from": "flf2v_own_pair"}

    if strategy == "t2v_own_refs":
        # Image Plan reference 角色图 → seedance t2v @refs(无需上镜)。
        if refs:
            cond.update(reference_images=[str(p) for p in refs],
                        anchoring="soft_t2v_refs")
            fallback_prompt = spec.prompt + ". " + " ".join(
                f"@Image{i + 1} is a reference for this shot's "
                f"{'subject' if i == 0 else 'setting'} — keep it consistent."
                for i in range(len(refs)))
            return Path(video_gen.generate(
                prompt=brain_prompt or fallback_prompt,
                duration=spec.duration, out_path=out, fps=fps,
                reference_images=refs, seed=seed)), cond
        strategy = "t2v"
        cond = {"strategy": "t2v", "degraded_from": "t2v_own_refs"}

    if strategy == "flf2v_bridge":
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        anchor_img = kf or (refs[0] if refs else None)
        if last is not None and anchor_img is not None:
            cond.update(first_anchor=str(last), last_anchor=str(anchor_img))
            return Path(video_gen.frame_to_frame(
                prompt=brain_prompt or spec.prompt, first_frame=last,
                last_frame=anchor_img,
                out_path=out, duration=spec.duration, seed=seed)), cond
        strategy = "ti2v_prev_last"      # 尾帧抽不出来 → 逐级降级(如实改写)
        cond = {"strategy": strategy, "degraded_from": "flf2v_bridge"}

    if strategy == "ti2v_prev_plus_keyframe":
        # 上镜尾帧 + 本镜图(可多张参考),一次调用。走【t2v +
        # reference_images】通道(refs 仅在 t2v 端点验证过;软锚 —— 构图级
        # 连续,不锁任何帧;要像素级用 ti2v_prev_last / flf2v_bridge)。
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        own = refs if refs else ([kf] if kf is not None else [])
        if last is not None and own:
            all_refs = [last] + own
            cond.update(reference_images=[str(p) for p in all_refs],
                        anchoring="soft_t2v_refs")
            fallback_prompt = (
                spec.prompt + ". Open on the exact scene state shown in "
                "@Image1 (the previous moment) and stay consistent with "
                + ", ".join(f"@Image{i + 2}" for i in range(len(own)))
                + " (this shot's planned image(s)).")
            return Path(video_gen.generate(
                prompt=brain_prompt or fallback_prompt,
                duration=spec.duration, out_path=out, fps=fps,
                reference_images=all_refs, seed=seed)), cond
        # 尾帧抽不出来 → 还有自己的图可用。降级取向:有首帧角色图(含兼容
        # 模式的 keyframe)优先硬锚 i2v;纯参考角色图(从未打算当首帧)才
        # 降到 t2v_own_refs —— 角色语义在降级里也不许错配。
        strategy = ("i2v_keyframe" if kf is not None
                    else "t2v_own_refs" if refs else "t2v")
        cond = {"strategy": strategy,
                "degraded_from": "ti2v_prev_plus_keyframe"}
        if strategy == "t2v_own_refs":
            cond.update(reference_images=[str(p) for p in refs],
                        anchoring="soft_t2v_refs")
            return Path(video_gen.generate(
                prompt=brain_prompt or spec.prompt, duration=spec.duration,
                out_path=out, fps=fps, reference_images=refs, seed=seed)), cond

    if strategy == "multi_image_fusion":
        # kling-video-o1 参考路线:[上镜尾帧?] + 本镜图(≤7;带 video 时后端
        # 自动收缩到 4)。brain 可要求同请求带上镜尾段视频(use_prev_tail_video)。
        imgs: list = []
        if prev is not None and prev.video_path is not None:
            last = _last_frame(Path(prev.video_path),
                               cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
            if last is not None:
                imgs.append(last)
        imgs.extend(refs if refs else ([kf] if kf is not None else []))
        tail_video = None
        if use_prev_tail_video and prev is not None \
                and prev.video_path is not None:
            tail_video = _cut_tail(
                Path(prev.video_path), window_tail_s,
                cache_dir / f"shot{spec.shot_idx:03d}_prev_tail.mp4")
        if len(imgs) >= 2 or (imgs and tail_video is not None):
            cond.update(images=[str(p) for p in imgs],
                        video=str(tail_video) if tail_video else None)
            fallback_prompt = spec.prompt + ". " + " ".join(
                f"Reference image {i + 1} defines "
                f"{'the continuing scene state' if i == 0 and prev else 'a subject/setting to keep consistent'}."
                for i in range(len(imgs)))
            return Path(video_gen.multi_image_to_video(
                prompt=brain_prompt or fallback_prompt, images=imgs,
                out_path=out, duration=spec.duration, seed=seed,
                video=tail_video)), cond
        # 不足 → 逐级降级(落到下方对应策略块,degraded_from 保留)
        strategy = ("i2v_keyframe" if kf is not None
                    else "ti2v_prev_last" if prev is not None
                    and prev.video_path else "t2v")
        cond = {"strategy": strategy, "degraded_from": "multi_image_fusion"}

    if strategy == "tiv2v_window":
        tail = _cut_tail(Path(prev.video_path), window_tail_s,
                         cache_dir / f"shot{spec.shot_idx:03d}_prev_tail.mp4")
        ref = tail if tail is not None else Path(prev.video_path)
        cond.update(reference_video=str(ref),
                    tail_seconds=(window_tail_s if tail else None),
                    first_frame=str(kf) if kf else None)
        return Path(video_gen.generate(
            prompt=brain_prompt or spec.prompt, duration=spec.duration,
            out_path=out, fps=fps,
            first_frame=kf, seed=seed, reference_video=ref)), cond

    if strategy == "ti2v_prev_last":
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        if last is not None:
            cond.update(first_frame=str(last))
            return Path(video_gen.generate(
                prompt=brain_prompt or spec.prompt, duration=spec.duration,
                out_path=out,
                fps=fps, first_frame=last, seed=seed)), cond
        strategy = "i2v_keyframe" if kf is not None else "t2v"
        cond = {"strategy": strategy, "degraded_from": "ti2v_prev_last"}

    if strategy == "i2v_keyframe" and kf is not None:
        cond.update(first_frame=str(kf))
        return Path(video_gen.generate(
            prompt=brain_prompt or spec.prompt, duration=spec.duration,
            out_path=out,
            fps=fps, first_frame=kf, seed=seed)), cond

    cond = {"strategy": "t2v", **({"degraded_from": cond.get("degraded_from")}
                                  if cond.get("degraded_from") else {})}
    return Path(video_gen.generate(
        prompt=brain_prompt or spec.prompt, duration=spec.duration,
        out_path=out,
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
    patience: int = 2,                  # 小循环:连续 N 轮被拒即止损(≤0 关闭)
    quality_bar: Optional[float] = None,  # 小循环:达标即停(None 关闭)
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
    replay_plan = {h["label"]: h["image_plan"]
                   for h in guidance["replay_hints"]
                   if h.get("converged") and h.get("image_plan")}
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

    # ── §B' Image Plan 阶段(逐 shot:brain 定【数量+角色+来源】→ 产图 →
    #     台账)。用户设定:单图 = 首帧或参考;双图 = 首尾帧或双参考;角色
    #     锁死后续的生成模型族。素材目录(asset_catalog)进决策上下文,brain
    #     看得见用户给了什么(Q-C:靠完整技能让 brain 对任意素材场景做对
    #     决策,不写死"背景图=首帧"这类规则)。──────────────────────────────
    kf_dir = cache_dir / "keyframes"
    asset_catalog = _asset_catalog(asset_memory)
    for entry, spec in zip(storyboard.entries, specs):
        menu = _image_plan_menu(video_gen, asset_memory)
        d = _decide(
            llm, "image-plan", menu,
            {"shot": entry.to_brain_line(),
             "storyboard": storyboard.to_brain_json(),
             "asset_catalog": asset_catalog,
             "episode_guidance": guidance},
            replay_hint=replay_plan.get(entry.label),
            priority=_PLAN_PRIORITY,
        )
        decisions.append({"stage": "image_plan", "label": entry.label, **d})
        plan_final, images, degraded_from = _execute_image_plan(
            d, entry, video_gen, asset_memory, retrieval, kf_dir)
        storyboard.set_image_plan(entry.shot_idx, plan_final, images,
                                  degraded_from=degraded_from)
        log.info("window: %s image-plan → %s (via=%s, %d image(s)%s)",
                 entry.label, plan_final, d["via"], len(images),
                 f", degraded from {degraded_from}" if degraded_from else "")

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
        brain_prompt = d.get("video_prompt", "")
        use_tail = bool(d.get("use_prev_tail_video", False))
        for s in range(max(1, n_candidates)):
            try:
                video_path, cond = _generate_with_condition(
                    d["strategy"], entry, prev, spec, video_gen,
                    shot_dir, seed=s, fps=fps, window_tail_s=window_tail_s,
                    brain_prompt=brain_prompt, use_prev_tail_video=use_tail)
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
            patience=patience, quality_bar=quality_bar,
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
            "stop_reason": res.stop_reason,   # 小循环为何停(自动轮数控制留痕)
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
