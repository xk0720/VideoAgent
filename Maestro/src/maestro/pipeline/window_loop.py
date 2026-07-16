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
from ..logging_utils import brain_log, get_logger
from .ref_slots import validate_references
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
#   tiv2v_window(尾段运动参考+可选软图,全走 t2v)> ti2v_prev_last(首帧硬锚)>
#   ti2v_prev_plus_keyframe(t2v+refs 软锚)> multi_image_fusion(融合)。
_CONDITION_PRIORITY = ["flf2v_own_pair", "flf2v_bridge", "extend_prev",
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
                        # desc = 干净语义(进 prompt 用;label 带 kind 前缀
                        # 只用于检索打分)。Q-D 链:用户描述 > caption > 文件名
                        "desc": (a.description or a.name or p.name),
                        "path": str(p)})
    for s_ in asset_memory.style_anchors:
        p = Path(getattr(s_, "source", "") or "")
        if p and p.exists():
            out.append({"kind": "style", "name": getattr(s_, "style_id", ""),
                        "label": _asset_label("style",
                                              getattr(s_, "style_id", ""),
                                              getattr(s_, "description", ""),
                                              str(p)),
                        "desc": (getattr(s_, "description", "")
                                 or getattr(s_, "style_id", "") or p.name),
                        "path": str(p)})
    return out


def _retrieve_asset_image(query: str, asset_memory: Optional[AssetMemory]
                          ) -> Optional[tuple[Path, str]]:
    """按关键词重叠给【全部】图片素材打分,取最高分(替代旧的"拿第一张")。
    确定性、可复现;0 重叠时退回第一张存在的图(单素材场景保持旧行为)。

    返回 (路径, 素材语义标签)。标签 = 用户描述 > 入库 VLM caption > 文件名
    (Q-D 链,素材目录里已备好)—— 2026-07-15 裁决 1.2:语义必须跟着图走,
    写 prompt 的人要知道"实际拿到了什么",不是"当时搜了什么"。
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
    return (Path(best["path"]),
            str(best.get("desc") or best.get("label", ""))) if best else None


def re_words(text: str) -> list[str]:
    import re

    return [w.lower() for w in re.findall(r"[a-zA-Z一-鿿0-9]+", text or "")]


def ensure_asset_descriptions(asset_memory: Optional[AssetMemory],
                              mllm=None, cache_dir=None) -> int:
    """Q-D 打标链的 VLM 中环:给【没有用户描述】的素材补语义标签,返回
    补标数量。用户很可能只给一个路径(2026-07-16 裁决:必须兼容)。

    - 图片(identity/style):VLM 看图补 caption 写回 description;
      mock/无 VLM → 不写(目录层有文件名末端兜底)。
    - 视频(video_shots):抽【中间帧】→ VLM caption → 写回 Shot.caption
      (它是 video_extract 检索的匹配键,也是剧本看到的视频语义);
      VLM 不可用/抽帧失败 → 文件名兜底写回(caption 不能留空,否则
      检索永远搜不到这段视频)。抽帧文件放 cache_dir(不给则跳过 VLM
      环,直接文件名兜底 —— 绝不往用户素材目录写临时文件)。"""
    if asset_memory is None:
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
        if mllm is not None:
            try:
                cap = (mllm.caption_image(src) or "").strip()
            except Exception as exc:
                log.warning("asset caption failed for %s: %r",
                            Path(src).name, exc)
        if cap:
            try:
                a.description = cap
                n += 1
            except AttributeError:
                pass                      # StyleRef 若无该字段则跳过(不硬塞)
    # 视频素材:中间帧 caption → Shot.caption(检索键 + 剧本可见语义)
    for sid, shot in (getattr(asset_memory, "video_shots", None) or {}).items():
        cap = (getattr(shot, "caption", "") or "").strip()
        src = getattr(shot, "source_video", "") or ""
        if cap or not src or not Path(src).exists():
            continue
        text = ""
        if mllm is not None and cache_dir is not None:
            out_dir = Path(cache_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            frame = extract_frame(Path(src), 10 ** 6,
                                  out_dir / f"asset_{sid}_mid.png")
            if frame is not None:
                try:
                    text = (mllm.caption_image(frame) or "").strip()
                except Exception as exc:
                    log.warning("asset video caption failed for %s: %r",
                                Path(src).name, exc)
        if text:
            shot.caption = f"{text} (from the user's video clip)"
            n += 1
        else:
            # 末端兜底:文件名。caption 是检索键,留空 = 这段视频永远
            # 检索不到;并大声提示打标质量受限(Q-D 链的诚实降级)。
            shot.caption = Path(src).stem.replace("_", " ")
            log.warning("asset video %s: no VLM caption available — "
                        "falling back to the FILENAME as its label "
                        "(retrieval quality will suffer)", Path(src).name)
    if n:
        log.info("asset labeling: %d asset(s) captioned by the VLM "
                 "(user description > VLM caption > filename)", n)
    return n


def _media_catalog(asset_memory: Optional[AssetMemory]) -> list[dict]:
    """剧本/图计划看的【全媒体】素材目录 = 图片目录 + 视频条目。

    2026-07-16 修复:旧目录只有图片,scene_write 根本不知道用户给了视频、
    里面是什么 —— ASSET MENTION LAW 对视频角色没有输入,素材白给检测也
    不覆盖视频。图片检索(_retrieve_asset_image)仍用纯图目录
    (_asset_catalog),视频文件绝不会被当成图返回。"""
    out = list(_asset_catalog(asset_memory))
    if asset_memory is None:
        return out
    for sid, shot in (asset_memory.video_shots or {}).items():
        src = getattr(shot, "source_video", "") or ""
        if src and Path(src).exists():
            cap = (getattr(shot, "caption", "") or "").strip()                 or Path(src).stem.replace("_", " ")
            out.append({"kind": "video", "name": sid,
                        "label": f"video: {cap}", "desc": cap,
                        "path": str(src)})
    return out


@dataclass
class MovieResult:
    """generate_movie_windowed 的完整回执(全部可审计)。"""

    final_video: Optional[Path]
    storyboard: StoryboardMemory
    shot_results: list = field(default_factory=list)   # SelfImproveResult per shot
    episode_id: str = ""
    decisions: list = field(default_factory=list)      # brain 的 §B/§C 决策流水
    # 需求 1(2026-07-15):基线锚点 {path, route, prompt, via};开关没开 =
    # None。用户裁决:只生成不比较 —— 用户自己看片对比。
    baseline_anchor: Optional[dict] = None


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


def _trim_head(video: Path, seconds: float, out_path: Path) -> Optional[Path]:
    """裁掉视频开头 `seconds` 秒(video-extend 的输出 = 输入片段+续段拼接,
    官方页原文 "the original and new segment are concatenated" —— 裁头后才
    是纯续段)。ffmpeg 缺失/失败 → None(调用方带痕降级用未裁版本)。"""
    if not shutil.which("ffmpeg"):
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{max(0.0, seconds):.2f}",
             "-i", str(video), "-c", "copy", str(out_path)],
            capture_output=True, timeout=120,
        )
    except Exception:
        return None
    if r.returncode != 0 or not out_path.exists()             or out_path.stat().st_size == 0:
        return None
    return out_path


def _probe_seconds(video: Path) -> float:
    """ffprobe 时长(秒);探不到 → 0.0(调用方按未知处理)。"""
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


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


# kind → 该决策要载入 prompt 的技能文件(模型输入,全英文;和修复 brain
# 载 orchestrator/SKILL.md 同一机制)。缺文件时用内联短指令兜底。
_KIND_TO_SKILL = {"image-plan": "image_plan",
                  "generation-condition": "window_generation"}


def _skill_body(kind: str) -> str:
    """载入该决策类型的技能全文(缓存);没有就返回内联短指令。
    装载结果必须响亮可见(2026-07-15 用户令:百分之百确定技能进了 prompt
    —— 首载打 INFO/WARNING,每次 brain 调用另在 brain_calls.jsonl 记
    skill_chars 作逐次证据)。"""
    name = _KIND_TO_SKILL.get(kind, "")
    if name and name not in _SKILL_CACHE:
        try:
            from ..skills.loader import load_skill

            sk = load_skill(name)
            _SKILL_CACHE[name] = sk["body"] if sk and sk["body"].strip() else ""
        except Exception:
            _SKILL_CACHE[name] = ""
        if _SKILL_CACHE[name]:
            log.info("brain skill LOADED: %s (%d chars) → goes into every "
                     "'%s' prompt", name, len(_SKILL_CACHE[name]), kind)
        else:
            log.warning("brain skill MISSING/EMPTY: %s — the '%s' brain gets "
                        "only a terse inline instruction (decision quality "
                        "will suffer)", name, kind)
    body = _SKILL_CACHE.get(name, "")
    if body:
        return body
    return (f"You are the window-generation brain. Pick EXACTLY ONE {kind} "
            "strategy from `menu` for the CURRENT shot. Consider the "
            "storyboard, the asset_catalog, and the episode guidance "
            "(replay_hints = strategies that WORKED on similar past tasks — "
            "prefer them; avoid = strategies that FAILED — never pick them "
            "for a similar shot).")


_SKILL_CACHE: dict = {}


def _brain_pick(llm, kind: str, menu: list[dict], context: dict) -> dict:
    """让 brain 用严格 JSON 从菜单选一项;失败返回 {}(调用方走兜底)。

    prompt = 该决策的【技能全文】(skills/brain_skills/*/SKILL.md,纯英文
    ——模型输入输出一律英文,用户裁决)+ 本回合 JSON 上下文。和
    OrchestratorAgent.decide 同款纪律:只能选菜单里的 name,越界即无效;
    语义附加字段(_EXTRA_FIELDS)轻校验透传,机械字段即使 brain 多嘴也被
    丢弃。MockLLM 回 "ack:..." 必然解析失败 → mock 模式全程走确定性兜底,
    不伪造"brain 决策"。"""
    if llm is None:
        return {}
    skill_name = _KIND_TO_SKILL.get(kind, "")
    skill_text = _skill_body(kind)
    # skill_chars = 进入本次 prompt 的技能全文长度;skill_loaded=False 表示
    # 用的是内联短指令(技能文件缺失)—— 逐次可审计的装载证据。
    skill_proof = {"skill": skill_name, "skill_chars": len(skill_text),
                   "skill_loaded": bool(_SKILL_CACHE.get(skill_name)),
                   # 裁决 1.3:输入也要可审计 —— THIS TURN 的完整上下文
                   # (技能全文不重复存,skill_chars 已证明其在场)
                   "context": context}
    prompt = (
        skill_text
        + "\n\nTHIS TURN (JSON):\n"
        + json.dumps({"menu": menu, **context}, ensure_ascii=False)
        + '\n\nSTRICT JSON only: {"strategy": "<name from menu>", '
          '"reason": "<one short sentence>", ... optional semantic fields '
          "per the skill above (images / video_prompt / use_prev_tail_video)}"
    )
    raw = ""
    try:
        raw = llm.complete(prompt)
        data = _extract_json(raw)
    except Exception as exc:
        brain_log(f"window/{kind}", {
            "label": context.get("shot", {}).get("label")
            if isinstance(context.get("shot"), dict) else None,
            "menu": sorted(m["name"] for m in menu),
            "raw": raw or f"<complete() raised: {exc}>", "parsed": None,
            "usable": False, **skill_proof})
        return {}
    valid = {m["name"] for m in menu}
    usable = isinstance(data, dict) and str((data or {}).get("strategy", "")) in valid
    if not usable:
        brain_log(f"window/{kind}", {
            "label": context.get("shot", {}).get("label")
            if isinstance(context.get("shot"), dict) else None,
            "menu": sorted(valid), "raw": raw,
            "parsed": data if isinstance(data, dict) else None,
            "usable": False, **skill_proof})
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
    # debug 日志(2026-07-14 用户令):brain 的原始输出 + 校验后决策全量落盘,
    # 拿它对照 docs/CONDITION_MODEL_MAP.md §1 就能核对"该策略调了哪个模型"。
    brain_log(f"window/{kind}", {
        "label": context.get("shot", {}).get("label")
        if isinstance(context.get("shot"), dict) else None,
        "menu": sorted(valid), "raw": raw, "parsed": dict(out),
        "usable": True, **skill_proof})
    return out


def _write_outline(llm, user_prompt: str, asset_catalog: list,
                   episode_guidance: dict, max_shots: int,
                   fallback_fn) -> tuple[list[str], list, list[str], str]:
    """§A 真·LLM playwriting → (outline, via)。三层纪律同 _decide:

    1) LLM + scene_write 技能全文 → 严格 JSON {"shots": [...]},逐条校验
       (字符串、非空、去完全重复、1..max_shots 截断)。【分镜数由 brain
       自己定】(用户裁决:绝不预设)——依据是剧情本身 + episode 记忆里
       相似任务的形状经验(past_task_shapes:当年几镜、成没成);
       max_shots 只是成本硬顶,不是创作指令。【绝不靠重复子句凑数】;
    2) 校验不过/LLM 不可用 → fallback_fn(确定性拆条,mock 模式的老路)。

    背景:v0.1 的确定性拆条按 `子句[i % n]` 循环填充——子句少于 n_shots 时
    必然产出重复分镜(实测翻车:2 子句 3 镜,第 3 镜重复第 1 镜)。真剧本
    必须由 LLM 写,拆条只配当兜底。"""
    if llm is not None:
        skill_text = _skill_body_named("scene_write")
        prompt = (
            skill_text
            + "\n\nTHIS TASK (JSON):\n"
            + json.dumps({"user_prompt": user_prompt,
                          "asset_catalog": asset_catalog,
                          "episode_guidance": {
                              "past_task_shapes":
                                  episode_guidance.get("past_task_shapes", []),
                          },
                          "max_shots_hard_cost_cap": max_shots},
                         ensure_ascii=False)
            + '\n\nSTRICT JSON only: {"shots": [{"description": "Shot 1: '
              '<detailed filmable description>", "duration_s": <int 4-10>, '
              '"end_state": "<one sentence: at the CUT, who/what is where, '
              'moving or still, in which direction>"}, '
              "...]} — each description 15-40 words (subject + action + "
              "setting + camera), scene N stated when the location changes. "
              "YOU decide the shot count AND each shot's duration_s (4-10 "
              "seconds, from how long the action NEEDS) from the story "
              "itself (use past_task_shapes as experience from similar "
              f"past tasks); max_shots ({max_shots}) is only a COST ceiling, "
              "never a target — never pad by repeating a shot. HANDOFF LAW: "
              "each shot's opening must continue the PREVIOUS shot's "
              "end_state exactly (position AND motion); to hand motion to "
              "the next shot, do NOT let the mover stop before the cut; a "
              "resting object may only move again if a NEW force/event acts "
              "on it (write that event into the description)."
        )
        raw = ""
        try:
            raw = llm.complete(prompt)
            data = _extract_json(raw)
        except Exception:
            data = None
        brain_log("window/scene_write", {
            "raw": raw, "parsed": data if isinstance(data, dict) else None,
            "usable": bool(isinstance(data, dict)
                           and isinstance(data.get("shots"), list)),
            "skill": "scene_write", "skill_chars": len(skill_text),
            "skill_loaded": bool(skill_text),
            "context": {"user_prompt": user_prompt,
                        "asset_catalog": asset_catalog,
                        "episode_guidance": episode_guidance,
                        "max_shots": max_shots}})
        if isinstance(data, dict) and isinstance(data.get("shots"), list):
            shots, durs, ends, seen = [], [], [], set()
            for s_ in data["shots"][:max_shots]:
                # 兼容两种形态:纯字符串,或 {description, duration_s, end_state}
                if isinstance(s_, dict):
                    text = str(s_.get("description", "")).strip()
                    dur = s_.get("duration_s")
                    end = str(s_.get("end_state", "") or "").strip()
                else:
                    text = str(s_).strip()
                    dur, end = None, ""
                key = text.lower()
                # 完全重复 = 凑数,丢弃(重复分镜正是本函数存在的原因)
                if len(text) >= 12 and key not in seen:
                    seen.add(key)
                    shots.append(text)
                    # 交接棒(需求 ②-①):end_state 是 brain 的决定;没输出
                    # = 空串,不编造(下游按"无交接信息"诚实处理)。
                    ends.append(end)
                    # 时长是 brain 的决定,范围写死 [4,10](2026-07-14 裁决)。
                    # brain 没输出/输出非法 → None = 不向 API 传 duration 字段,
                    # 用模型自然默认(用户裁决:绝不 feed 任何预设值)。
                    try:
                        durs.append(max(4, min(10, int(dur))))
                    except (TypeError, ValueError):
                        durs.append(None)
            if shots:
                # 修正 A(2026-07-16):素材白给检测(警告不阻断)——
                # 用户给了素材,但没有任何分镜描述提及任一素材关键词。
                # skill 只能教,这道确定性检查负责让浪费当场可见。
                if asset_catalog:
                    all_text = " ".join(shots).lower()
                    words = set()
                    for a in asset_catalog:
                        words |= {w for w in re_words(
                            str(a.get("desc") or a.get("label", "")))
                            if len(w) > 3}
                    if words and not any(w in all_text for w in words):
                        log.warning(
                            "scene_write: %d user asset(s) provided but NO "
                            "shot description mentions any of them — the "
                            "script may be wasting the assets (catalog: %s)",
                            len(asset_catalog),
                            [a.get("desc", a.get("label", ""))[:40]
                             for a in asset_catalog])
                return shots, durs, ends, "llm"
    fb = list(fallback_fn())
    # 兜底层没有 brain → None = 不传 duration 字段,API 用自己的自然默认
    # (不是 config 的 shot_duration,也不是我们编的数);end_state 同理为空。
    return fb, [None] * len(fb), [""] * len(fb), "fallback"


def _skill_body_named(name: str) -> str:
    """按名载入技能全文(缓存;缺文件返回 "")。首载响亮打日志。"""
    if name not in _SKILL_CACHE:
        try:
            from ..skills.loader import load_skill

            sk = load_skill(name)
            _SKILL_CACHE[name] = sk["body"] if sk and sk["body"].strip() else ""
        except Exception:
            _SKILL_CACHE[name] = ""
        if _SKILL_CACHE[name]:
            log.info("brain skill LOADED: %s (%d chars)", name,
                     len(_SKILL_CACHE[name]))
        else:
            log.warning("brain skill MISSING/EMPTY: %s — prompt will carry "
                        "NO skill text for this stage", name)
    return _SKILL_CACHE.get(name, "")


def _decide(llm, kind: str, menu: list[dict], context: dict,
            replay_hint: Optional[str], priority: list[str]) -> dict:
    """三层决策(§M 的可执行记忆就落在这):
    1) episode replay 命中且策略仍在菜单 → 直接采纳,via="episode"
       (长期记忆的检索即执行 —— 不再消耗一次 LLM 推理);
    2) brain LLM 严格 JSON → via="llm";
    3) 确定性优先级兜底 → via="fallback"(菜单非空必有解,循环永不卡死)。"""
    names = {m["name"] for m in menu}
    label = (context.get("shot", {}).get("label")
             if isinstance(context.get("shot"), dict) else None)
    if replay_hint and replay_hint in names:
        d = {"strategy": replay_hint, "via": "episode",
             "reason": "replaying a verified strategy from a similar past episode"}
        brain_log(f"window/{kind}", {"label": label, "parsed": dict(d),
                                     "via": "episode", "usable": True})
        return d
    picked = _brain_pick(llm, kind, menu, context)
    if picked:
        return {**picked, "via": "llm"}
    for name in priority:
        if name in names:
            d = {"strategy": name, "via": "fallback",
                 "reason": "deterministic priority (brain reply unusable)"}
            brain_log(f"window/{kind}", {"label": label, "parsed": dict(d),
                                         "via": "fallback", "usable": True})
            return d
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
        img, actual = None, ""
        try:
            img, actual = _make_keyframe(
                src, entry, video_gen, asset_memory, retrieval, out_dir,
                seed=entry.shot_idx * 2 + i, query=query, slot=i)
        except Exception as exc:
            log.info("image plan: slot %d (%s via %s) failed: %s",
                     i, role, src, exc)
        if img is not None:
            # 裁决 1.2:description = 这张图【实际是什么】(素材的真实标签/
            # t2i prompt/源片段 caption),写 prompt 的人按它引用;检索词
            # 另存 retrieval_query 供审计("搜的"和"拿到的"分开记)。
            row = {"path": str(img), "role": role, "source": src,
                   "description": actual or query}
            if actual and actual != query:
                row["retrieval_query"] = query
            produced.append(row)
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
                   slot: int = 0) -> tuple[Optional[Path], str]:
    """按来源产出一张图 → (路径, 实况语义);产不出真图返回 (None, "")
    (绝不放占位图冒充)。实况语义 = 这张图【实际是什么】:
      t2i → 生成 prompt 本身;asset_image → 素材的真实标签(用户描述 >
      入库 VLM caption > 文件名);video_extract → 源片段 caption。
    裁决 1.2:语义跟着图走,后面写 prompt 的人引用的是"实际拿到的",
    不是"当时搜的"。`slot` 区分同一 shot 的多张图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    query = query or entry.description
    if strategy == "t2i":
        out = out_dir / f"shot{entry.shot_idx:03d}_kf{slot}_t2i.png"
        return Path(video_gen.text_to_image(query, out, seed=seed)), query
    if strategy == "asset_image":
        # 按 shot 描述/检索词给全部图片素材打分取最优(Q-D 标签链)。
        got = _retrieve_asset_image(query, asset_memory)
        if got is None:
            return None, ""
        path, label = got
        return path, label
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
                cap = str(getattr(shot, "caption", "") or "")
                return got, (f"a frame extracted from the user's source "
                             f"video ({cap})" if cap else
                             "a frame extracted from the user's source video")
        return None, ""
    return None, ""   # "none"


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


def _desc_of(entry, path) -> str:
    """台账里这张图的实况语义(裁决 1.2:兜底模板/条件清单按它引用内容,
    不写空话)。查不到返回 ""。"""
    p = str(path)
    for im in (getattr(entry, "images", None) or []):
        if str(im.get("path", "")) == p:
            return str(im.get("description", "") or "")
    return ""


def _mention(entry, path, n: int, kind: str = "@Image") -> str:
    """一条内容感知的引用句:'@Image2 shows: <实况语义> — keep it
    consistent.'(语义缺失时退化为角色级措辞,绝不编内容)。"""
    d = _desc_of(entry, path)
    if d:
        return f"{kind}{n} shows: {d} — keep it consistent."
    return f"{kind}{n} is a planned image for this shot — keep it consistent."


def _slot_manifest(strategy: str, entry, prev,
                   use_prev_tail: bool = False) -> list[dict]:
    """方案 A(2026-07-16 裁决):【槽位清单】—— 执行器将要装配的引用槽位,
    在写 prompt 之前算出来,发给写 prompt 的人(brain / enhancer)。编号
    从"brain 要遵守的规则"变成"brain 拿到的数据",错无可猜。

    行:{"slot", "content"(实况语义), "referenceable"}。
    referenceable=False(FIRST_FRAME/LAST_FRAME/kling 的参考视频)= 该路线
    没有 @ 引用通道,prompt 只描述运动,不许写编号。

    ⚠ 单一事实源契约:每个分支的槽位顺序与 _generate_with_condition 对应
    策略块的 payload 装配顺序【一一对应】——改装配必须同步改这里
    (tests/unit/test_slot_manifest.py 锁行为)。"""
    ff, refs, pf, pl = _entry_images(entry)
    kf = ff
    prev_ok = prev is not None and getattr(prev, "video_path", None)

    def _c(path, default: str) -> str:
        """槽位实况语义;用户素材加 "user asset: " 前缀(2026-07-16 修正:
        enhancer 做"剧本提及 → 编号引用"翻译时,一眼锁定哪个槽位是用户
        点名的东西)。"""
        pstr = str(path)
        for im in (getattr(entry, "images", None) or []):
            if str(im.get("path", "")) == pstr:
                d = str(im.get("description", "") or "") or default
                if im.get("source") == "asset_image":
                    return f"user asset: {d}"
                return d
        return default

    rows: list[dict] = []
    if strategy == "flf2v_own_pair" and pf is not None and pl is not None:
        rows = [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": _c(pf, "this shot's planned opening frame")},
                {"slot": "LAST_FRAME", "referenceable": False,
                 "content": _c(pl, "this shot's planned closing frame")}]
    elif strategy == "t2v_own_refs":
        rows = [{"slot": f"@Image{i + 1}", "referenceable": True,
                 "content": _c(p, "a planned reference image")}
                for i, p in enumerate(refs)]
    elif strategy == "flf2v_bridge" and prev_ok:
        anchor = kf or (refs[0] if refs else None)
        rows = [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": "the previous shot's final frame (the moment "
                            "this shot continues from)"}]
        if anchor is not None:
            rows.append({"slot": "LAST_FRAME", "referenceable": False,
                         "content": _c(anchor, "this shot's planned image "
                                               "(the shot must arrive at it)")})
    elif strategy == "ti2v_prev_plus_keyframe" and prev_ok:
        own = refs if refs else ([kf] if kf is not None else [])
        rows = [{"slot": "@Image1", "referenceable": True,
                 "content": "the previous shot's final frame (the exact "
                            "moment to continue from)"}]
        rows += [{"slot": f"@Image{i + 2}", "referenceable": True,
                  "content": _c(p, "a planned image (target look)")}
                 for i, p in enumerate(own)]
    elif strategy == "extend_prev" and prev_ok:
        rows = [{"slot": "CONTINUATION_SOURCE", "referenceable": False,
                 "content": "the previous shot's tail — generation continues "
                            "from its exact final frame; identity/scene/light "
                            "carry over natively"}]
        if pl is not None:
            rows.append({"slot": "LAST_FRAME", "referenceable": False,
                         "content": _c(pl, "this shot's planned closing "
                                           "frame (the extension must arrive "
                                           "at it)")})
    elif strategy == "tiv2v_window" and prev_ok:
        own = [kf] if kf is not None else list(refs or [])
        rows = [{"slot": "@Video1", "referenceable": True,
                 "content": "the previous shot's tail segment — the ongoing "
                            "motion this shot continues"}]
        rows += [{"slot": f"@Image{i + 1}", "referenceable": True,
                  "content": _c(p, "a planned image (soft look reference)")}
                 for i, p in enumerate(own)]
    elif strategy == "multi_image_fusion":
        own = refs if refs else ([kf] if kf is not None else [])
        n = 1
        if prev_ok:
            rows.append({"slot": "reference image 1", "referenceable": True,
                         "content": "the previous shot's final frame — the "
                                    "continuing scene state"})
            n = 2
        rows += [{"slot": f"reference image {n + i}", "referenceable": True,
                  "content": _c(p, "a planned image")}
                 for i, p in enumerate(own)]
        if use_prev_tail and prev_ok:
            rows.append({"slot": "the reference video",
                         "referenceable": False,
                         "content": "the previous shot's tail segment "
                                    "(motion reference; describe the motion "
                                    "to continue in plain words)"})
    elif strategy == "ti2v_prev_last" and prev_ok:
        rows = [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": "the previous shot's final frame (this shot "
                            "opens exactly on it)"}]
    elif strategy == "i2v_keyframe" and kf is not None:
        rows = [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": _c(kf, "this shot's planned opening frame")}]
    return rows


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
        if "extend" in caps and hasattr(video_gen, "extend"):
            menu.append({"name": "extend_prev",
                         "description": "TRUE continuation: the video-extend "
                                        "model generates onward FROM the "
                                        "previous shot's final frame — "
                                        "identity, scene and light carry over "
                                        "natively (the strongest continuity "
                                        "route). `video_prompt` must describe "
                                        "ONLY what happens NEXT plus what to "
                                        "maintain — never re-describe what "
                                        "already happened. A planned "
                                        "'last'-role image (if any) becomes "
                                        "the target final frame."})
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
            # 裁决 1.2:引用必须带内容 —— 每个 @ImageN 说清它实际是什么
            fallback_prompt = spec.prompt + ". " + " ".join(
                _mention(entry, p_, i + 1) for i, p_ in enumerate(refs))
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
            # 裁决 1.2:@Image1 = 上镜尾帧(续接点);本镜图逐张带实况语义
            fallback_prompt = (
                spec.prompt + ". Open on the exact scene state shown in "
                "@Image1 (the final moment of the previous shot). "
                + " ".join(_mention(entry, p_, i + 2)
                           for i, p_ in enumerate(own)))
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
            # 裁决 1.2:kling 措辞 + 实况语义(首张若是上镜尾帧则写续接点)
            parts = []
            for i, p_ in enumerate(imgs):
                if i == 0 and prev is not None and not _desc_of(entry, p_):
                    parts.append("Reference image 1 is the final moment of "
                                 "the previous shot — continue from it.")
                else:
                    parts.append(_mention(entry, p_, i + 1,
                                          kind="reference image "))
            fallback_prompt = spec.prompt + ". " + " ".join(parts)
            return Path(video_gen.multi_image_to_video(
                prompt=brain_prompt or fallback_prompt, images=imgs,
                out_path=out, duration=spec.duration, seed=seed,
                video=tail_video)), cond
        # 不足 → 逐级降级(落到下方对应策略块,degraded_from 保留)
        strategy = ("i2v_keyframe" if kf is not None
                    else "ti2v_prev_last" if prev is not None
                    and prev.video_path else "t2v")
        cond = {"strategy": strategy, "degraded_from": "multi_image_fusion"}

    if strategy == "extend_prev":
        # 真续接(2026-07-16 裁决):video-extend 从上镜【末帧】继续生成,
        # 身份/场景/光线原生延续 —— attempt2 实证 reference_videos 参考通道
        # 接不上画面,prompt 无解,必须换原语。
        # 传上镜【尾段】(不传整镜:上传小、末帧才是接点);输出 = 尾段+
        # 续段拼接(官方语义)→ 裁掉头部尾段时长 = 本镜素材;裁不了(无
        # ffmpeg)→ 未裁版本 + 台账留痕(不装死)。
        tail = _cut_tail(Path(prev.video_path), window_tail_s,
                         cache_dir / f"shot{spec.shot_idx:03d}_prev_tail.mp4")
        src = tail if tail is not None else Path(prev.video_path)
        head_s = _probe_seconds(Path(src))
        raw = cache_dir / f"shot{spec.shot_idx:03d}_extend_raw_s{seed}.mp4"
        cond.update(extended_from=str(src),
                    tail_seconds=(window_tail_s if tail else None),
                    last_image=(str(pl) if pl is not None else None))
        video_gen.extend(
            prompt=brain_prompt or (
                spec.prompt + " — continue seamlessly from where the "
                "previous moment ends; keep the same subject identity, "
                "setting and lighting."),
            video_path=src, out_path=raw, duration=spec.duration,
            seed=seed, last_image=pl)
        if head_s > 0:
            trimmed = _trim_head(raw, head_s, out)
            if trimmed is not None:
                return Path(trimmed), cond
        cond["untrimmed"] = True          # 头部还带着上镜尾段(诚实留痕)
        log.warning("extend_prev: could not trim the %.1fs source head off "
                    "the extend output (ffmpeg/ffprobe unavailable) — using "
                    "the concatenated clip as-is", head_s)
        return Path(raw), cond

    if strategy == "tiv2v_window":
        # 映射表铁律(docs/CONDITION_MODEL_MAP.md §1 #8):tiv2v_window 永远走
        # text-to-video 端点 —— 尾段视频走 reference_videos(@Video1),本镜图
        # (如有)走 reference_images(@Image1,软锚)。旧实现把图当 first_frame
        # 会切到 image-to-video 端点,而 i2v schema 没有 reference_videos(未
        # 验证组合,后端现已直接拒绝)。要硬锁开场帧选 ti2v_prev_last /
        # flf2v_bridge,不选本策略。
        tail = _cut_tail(Path(prev.video_path), window_tail_s,
                         cache_dir / f"shot{spec.shot_idx:03d}_prev_tail.mp4")
        ref = tail if tail is not None else Path(prev.video_path)
        own_imgs = [kf] if kf is not None else list(refs or [])
        cond.update(reference_video=str(ref),
                    tail_seconds=(window_tail_s if tail else None),
                    reference_images=([str(p) for p in own_imgs] or None),
                    anchoring="soft_t2v_video_refs")
        # 裁决 1.2:@Video1 = 续接点;本镜图逐张带实况语义
        fallback_prompt = (
            spec.prompt + ". @Video1 is the immediately preceding moment of "
            "this scene — continue its motion and camera seamlessly. "
            + " ".join(_mention(entry, p_, i + 1)
                       for i, p_ in enumerate(own_imgs)))
        return Path(video_gen.generate(
            prompt=brain_prompt or fallback_prompt, duration=spec.duration,
            out_path=out, fps=fps, seed=seed, reference_video=ref,
            reference_images=(own_imgs or None))), cond

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


# 接点实况缓存:(尾帧路径, mtime) → 一句实况。一镜一次 VLM 调用。
_JUNCTION_CACHE: dict = {}


def _junction_state(mllm, prev, cache_dir: Path) -> str:
    """需求 ②(2026-07-15):看上一镜的【真实尾帧】,出一句续接实况
    ("the apple is at rest at the center of the floor")。写 prompt 的人
    从实况起笔,不再照剧本想象。

    诚实链:无上一镜 / 无 VLM / 尾帧抽不出 / VLM 失败 → ""(跳过,不编)。
    VLM 双模式(用户裁决):describe_junction 由 GeminiVLM(API)和
    LocalQwenVLM(本地)同名实现,models.mllm.name 切换。"""
    if prev is None or not getattr(prev, "video_path", None) or mllm is None:
        return ""
    fn = getattr(mllm, "describe_junction", None)         or getattr(mllm, "caption_image", None)
    if fn is None:
        return ""
    frame = _last_frame(Path(prev.video_path),
                        Path(cache_dir) / "junction_prev_last.png")
    if frame is None:
        return ""
    fp = Path(frame)
    try:
        key = (str(fp.resolve()), fp.stat().st_mtime_ns)
    except OSError:
        return ""
    if key not in _JUNCTION_CACHE:
        try:
            _JUNCTION_CACHE[key] = str(fn(fp) or "").strip()
        except Exception as exc:
            log.warning("junction caption failed: %s — proceeding without "
                        "the actual-state hint", exc)
            _JUNCTION_CACHE[key] = ""
        if _JUNCTION_CACHE[key]:
            log.info("junction state: %s", _JUNCTION_CACHE[key][:160])
    return _JUNCTION_CACHE[key]


def _conditions_for_prompt(strategy: str, entry, prev,
                           use_prev_tail: bool,
                           junction: str = "") -> list[dict]:
    """给 prompt enhancer 的【条件事实清单】(2026-07-15 需求 2):执行器
    按策略把"生成时真的会喂进去什么"翻译成文字 —— 增强器只能利用这些
    事实,不能发明条件。

    方案 A(2026-07-16):媒体条件直接来自槽位清单(_slot_manifest,与
    payload 装配同源),逐条 {kind: image|video, slot, referenceable,
    description} —— 增强器引用编号只许照抄 slot,校验闸在出口把关。
    状态条件(kind=state)照旧。"""
    conds: list[dict] = []
    for r in _slot_manifest(strategy, entry, prev, use_prev_tail):
        conds.append({"kind": ("video" if "video" in r["slot"].lower()
                               else "image"),
                      "slot": r["slot"],
                      "referenceable": bool(r.get("referenceable")),
                      "description": r.get("content", "")})
    # 需求 ②:状态类条件 —— prompt 必须从真实接点起笔、以剧本 end_state 收笔
    if junction:
        conds.append({"kind": "state", "role": "opening_state_actual",
                      "description": junction})
    prev_end = str(getattr(prev, "end_state", "") or "") if prev else ""
    if prev_end:
        conds.append({"kind": "state", "role": "previous_end_state_script",
                      "description": prev_end})
    own_end = str(getattr(entry, "end_state", "") or "")
    if own_end:
        conds.append({"kind": "state", "role": "required_end_state",
                      "description": own_end})
    return conds


# ─────────────────────────────────────────────────────────────────────────
# 基线锚点(2026-07-15 需求 1,开关控制):任务开始时按用户指令【一次调用】
# 直出一条视频,收尾与我们的成片盲测对比 —— 框架到底比"裸调一次模型"好
# 多少,让 verifier 说话。路线映射是确定性的(用户设定):
#   无素材            → seedance-2.0 text-to-video
#   仅图片            → seedance-2.0 image-to-video(ti2v,首图当首帧)
#   有视频(可带图)   → seedance-2.0 text-to-video + reference_images/videos
# 全程 try/except:锚点是附加物,任何失败只记日志,绝不影响正流程。
# ─────────────────────────────────────────────────────────────────────────
def _head_clip(video: Path, seconds: float, out: Path) -> Optional[Path]:
    """取视频开头 ≤seconds 秒(seedance reference_videos 单条 ≤15s 的硬限)。
    时长本来就达标 → 原样返回;ffmpeg/ffprobe 缺失或失败 → None(调用方
    诚实放弃该视频条件)。"""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=30)
        dur = float(probe.stdout.strip())
        if dur <= seconds:
            return video
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-t", f"{seconds:.2f}",
             "-c", "copy", str(out)], capture_output=True, timeout=300)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
    except Exception:
        pass
    return None


def _asset_media(asset_memory) -> tuple[list[Path], list[Path], list[str]]:
    """素材库 → (存在的图片, 存在的视频, 文字描述清单)。"""
    imgs: list[Path] = []
    vids: list[Path] = []
    notes: list[str] = []
    if asset_memory is None:
        return imgs, vids, notes
    img_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    for ident in (asset_memory.identity_anchors or {}).values():
        p = Path(str(ident.source))
        if ident.source and p.exists() and p.suffix.lower() in img_ext:
            imgs.append(p)
            notes.append(f"image: {ident.description or ident.name or p.name}")
    for shot in (asset_memory.video_shots or {}).values():
        p = Path(str(shot.source_video))
        if shot.source_video and p.exists() and p not in vids:
            vids.append(p)
            notes.append(f"video: {shot.caption or p.name}")
    return imgs, vids, notes


def _generate_baseline_anchor(user_prompt: str, asset_memory, video_gen, llm,
                              cache_dir: Path, duration=None) -> Optional[dict]:
    """一次调用直出锚点视频。返回 {path, route, prompt, via} 或 None(失败)。

    用户裁决(2026-07-15):锚点【只生成】——不做 verify_pair 对比、不接
    prompt enhancer,用户自己看片对比。"""
    try:
        imgs, vids, notes = _asset_media(asset_memory)
        out = Path(cache_dir) / "baseline_anchor.mp4"
        # 锚点 prompt:brain 把用户指令浓缩成【单条】视频 prompt(整个故事
        # 一镜到底);LLM 不可用 → 用户指令原文(诚实 fallback)。
        prompt, via = user_prompt, "fallback"
        if llm is not None:
            raw = ""
            try:
                raw = llm.complete(
                    "Condense the following video task into ONE video-"
                    "generation prompt (English, 30-100 words, subject + "
                    "action + setting + camera), covering the WHOLE story "
                    "as a single continuous shot."
                    + (" Mention provided references as @Image1…/@Video1 "
                       "with their purpose. Available materials: "
                       + "; ".join(notes) if notes else "")
                    + f'\n\nTASK: {user_prompt}\n\nSTRICT JSON only: '
                      '{"video_prompt": "..."}')
                data = _extract_json(raw)
                got = (data or {}).get("video_prompt") if isinstance(data, dict) else None
                if isinstance(got, str) and got.strip():
                    prompt, via = got.strip(), "llm"
            except Exception:
                pass
            brain_log("window/baseline_anchor", {
                "raw": raw, "parsed": {"video_prompt": prompt}, "via": via,
                "usable": via != "fallback"})

        # 确定性路线(用户设定的映射,docs/CONDITION_MODEL_MAP.md §5)
        if vids:
            route = "t2v_refs"
            capped = []
            for i, v in enumerate(vids[:3]):          # ≤3 条、每条 ≤15s
                c = _head_clip(v, 15.0,
                               Path(cache_dir) / f"anchor_ref{i}.mp4")
                if c is not None:
                    capped.append(c)
            if not capped:
                log.warning("baseline_anchor: no usable reference video "
                            "(ffmpeg missing / cut failed) — images/t2v only")
            video_gen.generate(
                prompt=prompt, duration=duration, out_path=out, seed=0,
                reference_images=(imgs[:9] or None),
                reference_video=(capped[0] if capped else None))
        elif imgs:
            route = "ti2v"
            if len(imgs) > 1:
                log.info("baseline_anchor: %d images — first one is the "
                         "first frame (user-ruled ti2v route)", len(imgs))
            video_gen.generate(prompt=prompt, duration=duration,
                               out_path=out, seed=0, first_frame=imgs[0])
        else:
            route = "t2v"
            video_gen.generate(prompt=prompt, duration=duration,
                               out_path=out, seed=0)
        log.info("baseline_anchor: generated via %s route → %s", route, out)
        return {"path": str(out), "route": route, "prompt": prompt,
                "via": via}
    except Exception as exc:
        log.warning("baseline_anchor generation failed (%s) — the main "
                    "pipeline continues without it", exc)
        return None


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
    repair_severity: float = 0.0,       # 最坏缺陷低于此值不修(0 关闭,荐 0.6)
    baseline_anchor: bool = False,      # 需求 1(2026-07-15):开工直出锚点视频
    baseline_anchor_duration=None,      # 锚点时长(None = API 默认)
    prompt_enhancer=None,               # 需求 2:可选 PromptEnhancerAgent
    mllm=None,                          # 需求 ②:接点实况 VLM(缺省用 verifier.judge)
) -> MovieResult:
    """窗口式全片生成:§A playwriting → §B keyframe → §C+§D 逐镜窗口循环
    → §E 合成 → §M episode 蒸馏。全程读写 StoryboardMemory(R1)。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    asset_memory = asset_memory or AssetMemory()
    llm = llm or getattr(orchestrator, "llm", None)
    mllm = mllm or getattr(verifier, "judge", None)
    video_gen = generator.video_gen
    decisions: list[dict] = []

    # ── 需求 1:基线锚点(开关控制;失败绝不影响正流程)。用户裁决:只
    # 生成,不做机器对比 —— 用户自己看片。────────────────────────────────
    anchor: Optional[dict] = None
    if baseline_anchor:
        anchor = _generate_baseline_anchor(
            user_prompt, asset_memory, video_gen, llm, cache_dir,
            duration=baseline_anchor_duration)
        decisions.append({"stage": "baseline_anchor",
                          **({k: anchor[k] for k in ("route", "via", "path")}
                             if anchor else {"failed": True})})

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
    # 真·LLM 剧本(scene_write 技能驱动,分镜数按剧情定、描述带细节、绝不
    # 重复凑数);LLM 不可用/输出不合格 → 确定性拆条兜底(mock 模式老路)。
    screenwriter = screenwriter or ScreenwriterAgent()
    director = director or DirectorAgent()
    plan_cfg = getattr(screenwriter, "config", {}) or {}
    asset_catalog0 = _media_catalog(asset_memory)
    outline, shot_durations, shot_end_states, outline_via = _write_outline(
        llm, user_prompt, asset_catalog0,
        episode_guidance=guidance,
        max_shots=int(plan_cfg.get("max_shots", 6)),
        fallback_fn=lambda: screenwriter.run(user_prompt, asset_memory),
    )
    decisions.append({"stage": "playwriting", "label": "outline",
                      "strategy": f"{len(outline)} shots", "via": outline_via,
                      "reason": "LLM playwriting (scene_write skill)"
                                if outline_via == "llm"
                                else "deterministic clause split (fallback)"})
    specs = director.run(outline, asset_memory, lesson_library)
    # 时长是 brain 的决定([4,10]),没决定就是 None(= 生成调用不传
    # duration 字段,API 用默认)—— 一律覆盖 director 从 config 带来的预设
    # (用户裁决:千万不能自己随意指定时长)。
    for spec_, dur_ in zip(specs, shot_durations):
        spec_.duration = (float(dur_) if dur_ is not None else None)
    storyboard = StoryboardMemory.from_outline(
        outline, path=cache_dir / "storyboard.json")
    # 交接棒(需求 ②-①)进台账:brain 声明的镜尾状态,下一镜续接的依据、
    # 评审的镜尾验收标准;空串 = brain 没说,不编造。
    for entry_, end_ in zip(storyboard.entries, shot_end_states):
        entry_.end_state = end_
    log.info("window: playwriting done via=%s — %s",
             outline_via, storyboard.summary())

    # ── §B' Image Plan 阶段(逐 shot:brain 定【数量+角色+来源】→ 产图 →
    #     台账)。用户设定:单图 = 首帧或参考;双图 = 首尾帧或双参考;角色
    #     锁死后续的生成模型族。素材目录(asset_catalog)进决策上下文,brain
    #     看得见用户给了什么(Q-C:靠完整技能让 brain 对任意素材场景做对
    #     决策,不写死"背景图=首帧"这类规则)。──────────────────────────────
    kf_dir = cache_dir / "keyframes"
    asset_catalog = _media_catalog(asset_memory)
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

        # 需求 ②:接点实况 —— VLM 看上一镜真实尾帧出一句状态;和剧本
        # 交接棒(上一镜 end_state / 本镜 end_state)一起进 brain 上下文,
        # prompt 从实况起笔,不照剧本想象。
        shot_dir = cache_dir / f"shot{entry.shot_idx:03d}"
        junction_actual = _junction_state(mllm, prev, shot_dir)
        junction_ctx = {
            "prev_last_frame_actual": junction_actual or None,
            "prev_end_state_script": (getattr(prev, "end_state", "") or None)
            if prev else None,
            "required_end_state": entry.end_state or None,
        }

        # §C brain 选条件策略(episode → llm → 兜底 三层)。
        # 方案 A(2026-07-16):每个候选策略的【槽位清单】随菜单发给 brain
        # —— 它写 video_prompt 时引用编号只许照抄所选策略的清单,不许猜。
        menu = _condition_menu(entry, prev, video_gen)
        slots_by_strategy = {
            m["name"]: _slot_manifest(m["name"], entry, prev,
                                      use_prev_tail=True)
            for m in menu}
        d = _decide(
            llm, "generation-condition", menu,
            {"shot": entry.to_brain_line(),
             "prev_shot": prev.to_brain_line() if prev else None,
             "junction": junction_ctx,
             "slots_by_strategy": slots_by_strategy,
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
        slots = _slot_manifest(d["strategy"], entry, prev, use_tail)
        # ── 需求 2:可选 prompt 润色(条件事实 + 官方 prompt 技巧技能)。
        # 失败返回 None → 保留原 prompt,增强层永远不破坏正流程。
        if prompt_enhancer is not None:
            enhanced = prompt_enhancer.run(
                entry.description, strategy=d["strategy"],
                conditions=_conditions_for_prompt(d["strategy"], entry, prev,
                                                  use_tail,
                                                  junction=junction_actual),
                base_prompt=brain_prompt or spec.prompt, label=entry.label)
            if enhanced:
                brain_prompt = enhanced
                decisions.append({"stage": "prompt_enhance",
                                  "label": entry.label,
                                  "strategy": d["strategy"], "via": "llm"})
        # ── 方案 A 出口闸:prompt 里的引用必须 ⊆ 所选策略的槽位清单。
        # 引用不存在的编号 → 弃用这条 prompt(落内容感知兜底模板),错
        # 编号永远到不了 API;可引用槽位漏提 → 自动补一句(素材不白传)。
        if brain_prompt:
            fixed, audit = validate_references(brain_prompt, slots)
            if not audit["ok"]:
                log.warning("window: %s prompt references unknown slots %s "
                            "(allowed: %s) — dropping it for the "
                            "content-aware fallback template", entry.label,
                            audit["unknown"], audit["allowed"])
                decisions.append({"stage": "ref_validate",
                                  "label": entry.label,
                                  "strategy": d["strategy"], "via": "gate",
                                  "reason": f"unknown refs {audit['unknown']}"
                                            " — fell back to template"})
                brain_prompt = ""
            else:
                if audit["appended"]:
                    log.info("window: %s prompt was missing %s — mention(s) "
                             "appended", entry.label, audit["appended"])
                    decisions.append({"stage": "ref_validate",
                                      "label": entry.label,
                                      "strategy": d["strategy"],
                                      "via": "gate",
                                      "reason": "appended mentions: "
                                                f"{audit['appended']}"})
                brain_prompt = fixed
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
            # 评审上下文 = 生成条件(原生视频评审的核心):reviewer 会拿这些
            # 条件图/参考视频和成片一起看,评"是否贴合条件"。
            clip.conditioning = {
                "video_prompt": brain_prompt or spec.prompt,
                "end_state": entry.end_state or None,
                "junction_prev_actual": junction_actual or None,
                "images": [{"path": im.get("path"), "role": im.get("role")}
                           for im in entry.images
                           if im.get("path") and Path(im["path"]).exists()],
                "reference_video": (cond.get("reference_video")
                                    or cond.get("video")),
            }
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
            repair_severity=repair_severity,
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

        # 评审证据量:0 条 checklist + 0 条物理判定 = 评审员们什么都没说
        # (真实 VLM 静默失败的典型症状:分数全默认、revision 0 即"收敛")。
        # 这种"没证据 = 全过"必须在台账里现形,不许伪装成真收敛。
        n_items = len(best.checklist.items)
        n_verd = len(best.physics_verdicts)
        if n_items == 0 and n_verd == 0:
            log.warning(
                "window: %s review produced ZERO evidence (no checklist "
                "items, no verdicts) — convergence is VACUOUS; check the "
                "VLM warnings above (HTTP errors / unparseable replies)",
                entry.label)
        # 评审轨迹 + 修复动作嵌入台账(§D "意见嵌入轨迹")
        storyboard.add_review(entry.shot_idx, {
            "revision": best.revision,
            "weighted_total": best.metric_scores.get("weighted_total", 0.0),
            "review_evidence": {"checklist_items": n_items,
                                "physics_verdicts": n_verd},
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
                       decisions=decisions, baseline_anchor=anchor)
