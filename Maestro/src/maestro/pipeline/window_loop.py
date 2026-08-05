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
import re
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
from ..pipeline.generate_loop import (SelfImproveResult,
                                       generate_shot_orchestrated)
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
# 审计对齐(2026-07-17):extend_prev(真续接,最强)先于 flf2v_bridge
# (bridge 只在"必须到达本镜图"时才对 —— 确定性层无从判断意图,默认续接)。
_CONDITION_PRIORITY = ["flf2v_own_pair", "extend_prev", "flf2v_bridge",
                       "ti2v_prev_last", "ti2v_prev_plus_keyframe",
                       "t2v_own_refs", "i2v_keyframe", "t2v"]
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
        # 官方肖像(portrait: 前缀)只走 §C 自动附挂专用通道,绝不进检索/
        # 计划目录 —— 2026-07-31 实锤:image_plan 把肖像检索回来当本镜图,
        # 同一张图从两条通道各进一次引用列表,正面全身像支配了开场构图。
        if str(getattr(a, "identity_id", "") or "").startswith("portrait:"):
            continue
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


# ViMax 借鉴(2026-07-17 P1-1):剧本描述里的 <角括号> 实体标记 ——
# 机器可解析的"本镜谁出场",cast 注入与评审 check 由此精准化。
_MARKER_RE = re.compile(r"<([^<>\n]{1,60})>")


def _strip_markers(text: str) -> str:
    """<the cat> → the cat。标记是规划层元数据,进最终 prompt 前必须剥掉
    (生成模型不认识角括号)。"""
    return _MARKER_RE.sub(r"\1", text or "")


# 锚定路线(开场由像素锚 / @Image1 PIN 决定)。attempt3 实锤(2026-07-18):
# 这些路线上 prompt 里的建景句(setting 整句重述)和开场布局重述会和锚
# 竞争 —— t2v 会照文字重新搭场景,首帧软锁被稀释;瘦身法则由此而来。
_ANCHORED_STRATEGIES = {"i2v_keyframe", "flf2v_own_pair", "flf2v_bridge",
                        "ti2v_prev_last", "ti2v_prev_plus_keyframe",
                        "extend_prev", "i2v_first",
                        # 2026-08-04 用户令(引用铁律):锚 = 身份像素,
                        # 不 = 钉帧。ref2v 满手参考图,身份在像素里 ——
                        # 正典闸对它强插外观文字正是 run8 prompt 污染的
                        # 主源,豁免。文字仍是唯一身份载体的只剩纯 t2v。
                        "ref2v"}

# 首帧强锁句(用户实测有效的原话;兜底模板与全修合成共用一处定义)
_PIN_SENTENCE = ("The shot opens EXACTLY on @Image1 — the final moment of "
                 "the previous shot; do not alter its scene or layout.")

# cast 契约值的格式:static: X; dynamic: Y。全角分号/冒号同样合法 ——
# 2026-07-31 实锤:中文描述符用";"绕过了拆分器,static:/dynamic: 标签
# 原文进了肖像 t2i prompt。
_CAST_SPLIT_RE = re.compile(r"static[:\uff1a]\s*(.+?)\s*[;;,]\s*dynamic[:\uff1a].*",
                            re.IGNORECASE | re.DOTALL)


def _static_half(desc: str) -> str:
    """cast 契约值 → static 半句(无标签)。正则不中但 dynamic 标签在
    (brain 写了变体格式)→ 关键词兜底切分 —— 契约标签绝不许漏进任何
    模型 prompt;两个标签都没有 → 原文即静态描述。"""
    s = str(desc or "").strip()
    m = _CAST_SPLIT_RE.match(s)
    if m:
        return m.group(1).strip()
    cut = re.split(r"dynamic[:\uff1a]", s, maxsplit=1, flags=re.IGNORECASE)[0]
    cut = re.sub(r"^\s*static[:\uff1a]\s*", "", cut, flags=re.IGNORECASE)
    return cut.strip().rstrip(";;,. ").strip()


def _scrub_cast_labels(text: str, cast: Optional[dict] = None) -> str:
    """"static: X; dynamic: Y" 是规划层契约格式,不是给生成模型看的
    (attempt3:标签连同 dynamic 清单逐字进了 prompt)。精确清洗:cast
    值整串出现 → 替换为 static 半句;残留的裸 "static:" 标签词剥掉;
    "dynamic:" 还残留(brain 改写过,无法安全定界)→ 响亮告警,不盲动刀。"""
    out = str(text or "")
    for v in (cast or {}).values():
        s = str(v or "").strip()
        m = _CAST_SPLIT_RE.match(s)
        if m and s in out:
            out = out.replace(s, m.group(1).strip())
    out = re.sub(r"\bstatic[:\uff1a]\s*", "", out)
    if re.search(r"\bdynamic[:\uff1a]", out, re.IGNORECASE):
        log.warning("cast label 'dynamic:' survived in an outgoing prompt "
                    "(the writer paraphrased the contract) — passing "
                    "through unmodified; fix the writer via skill")
    return out


# 剧本句的 "Shot N: scene N —" 前缀是台账元数据,进 prompt 前剥掉
_SHOT_PREFIX_RE = re.compile(r"^\s*shot\s+\d+\s*:\s*"
                             r"(scene\s+\d+\s*[—-]\s*)?", re.IGNORECASE)


def _regen_prompt(strategy: str, base: str, hint: str, slots: list,
                  action: str = "", end_state: str = "") -> str:
    """全修(regenerate)的 prompt 合成 —— P0-B(2026-07-18,attempt3):
    hint 非空时【替换】原动作部分,绝不追加("base + Fix: hint" 让动作说
    两遍、身份说三遍,越修越长,首帧软锁被稀释)。

    二轮修订(用户质询:hint 若只写外观不写动作?):PIN + 纯外观修正 =
    整条 prompt 无运动指令 → 静止/循环。不做"hint 有没有动作"的启发式
    检测(动词识别不可靠,静默分支不诚实)——【无条件】在 hint 后接一句
    剧本动作锚:起点(PIN)+ 过程(剧本句)+ 终点(end_state)三件套
    永远齐;hint 本含动作时动作被说两遍 —— 动作是唯一重复有益的内容
    类别(建景句重复诱导重建场景,动作重复是加权)。

    漏提的可引用槽位由引用闸门自动补句;hint 引用了清单外编号 → 回退
    原 base(错编号绝不出门)。"""
    if not hint:
        return base
    pin = _PIN_SENTENCE if strategy == "ti2v_prev_plus_keyframe" else ""
    act = _SHOT_PREFIX_RE.sub("", str(action or "").strip()).rstrip(". ")
    anchor = ""
    if act:
        es = str(end_state or "").strip().rstrip(". ")
        anchor = (f"This shot's scripted action: {act}"
                  + (f", ending as: {es}." if es else "."))
    prompt = " ".join(x for x in (pin, hint.strip(), anchor) if x)
    fixed, audit = validate_references(prompt, slots)
    if not audit["ok"]:
        log.warning("regen hint references unknown slots %s — falling back "
                    "to the original prompt", audit["unknown"])
        return base
    return fixed


_PRESERVE_CLAUSE = "Preserve the established scene, lighting and camera."

# 首帧锁定在【上一镜尾帧】上的策略:它们的第 0 帧 ≈ 上一镜最后一帧,
# 直接拼接会让同一画面连续出现两次(约 42ms 的微顿/轻闪)。拼装时裁掉
# 重复帧 —— 但必须先量证实(用户 2026-07-30:"剪掉首帧不就行了";
# 检查由我加:量不出/不相似就不裁,绝不凭策略名瞎剪真帧)。
_PREV_FRAME_LOCKED = {"ti2v_prev_last", "ti2v_prev_plus_keyframe",
                      "flf2v_bridge"}
# 平均像素差阈值(/255):实测"同帧不同质"≈5-6,镜内正常相邻帧≈1-1.5,
# 真不同帧 >12(movie_20260729_150307 三组实测)。
_DUP_FRAME_MAD = 8.0


def _first_last_mad(prev_video: Path, video: Path,
                    work_dir: Path) -> Optional[float]:
    """上一镜末帧 vs 本镜首帧的平均像素差(/255);任何环节算不出 →
    None(不猜,调用方不裁)。"""
    if not shutil.which("ffmpeg"):
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    a, b = work_dir / "_prev_last.rgb", work_dir / "_cur_first.rgb"
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-sseof", "-0.05",
             "-i", str(prev_video), "-frames:v", "1", "-f", "rawvideo",
             "-pix_fmt", "rgb24", str(a)],
            check=True, capture_output=True, timeout=60)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-i", str(video),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
             str(b)],
            check=True, capture_output=True, timeout=60)
        x = np.fromfile(a, dtype=np.uint8).astype(np.float32)
        y = np.fromfile(b, dtype=np.uint8).astype(np.float32)
        if x.size == 0 or x.size != y.size:
            return None
        return float(np.abs(x - y).mean())
    except Exception:
        return None
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)


# §G 钉帧完整性闸门(2026-08-02 用户批准,默认关)适用路线:开场被
# 一张图钉住(硬锁或 @Image1 软锁)的所有路线 —— 只有"钉住的开场"
# 才存在"第 2 帧抛开钉帧重画"这种失效。
_PIN_GATE_ROUTES = {"i2v_keyframe", "ti2v_prev_last", "flf2v_own_pair",
                    "flf2v_bridge", "ti2v_prev_plus_keyframe", "i2v_first"}
# 其中"上镜末帧锚定"的三条:交付片已被 _drop_first_frame 裁过头,撕裂
# 可能只在【接点】(上镜末帧 vs 本片帧 0)可见 → 闸门要连接点一起量。
_PIN_GATE_PREV = {"ti2v_prev_last", "flf2v_bridge",
                  "ti2v_prev_plus_keyframe", "i2v_first"}


def _pin_frame_mad(video: Path, work_dir: Path,
                   prev_video: Optional[Path] = None) -> Optional[float]:
    """§G:开场撕裂度 = max(接点差, 帧 0→1 差, 帧 1→2 差)(/255)。

    2026-08-02 对抗核查修正(三处盲区,均已实锤):
    - 帧 1→2 撕裂:未裁头路线的"钉两帧后重画"(144652 实录 13.58);
    - 帧 0→1 撕裂:裁头路线(_drop_first_frame 已切掉重复钉帧)上同一
      失效前移一帧;或未裁路线上钉帧只撑了一帧;
    - 接点撕裂(prev_video 给出时,量上镜末帧 vs 本片帧 0):钉帧被
      完全无视 → 片内无撕裂,只有对着上镜末帧才可见。
    体温表:健康相邻帧 ≈1~1.5;撕裂 ≈13+。一个分量算不出就跳过该
    分量;全算不出 → None(不猜,调用方响亮留痕)。纯算术零成本。"""
    if not shutil.which("ffmpeg"):
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    f: dict = {}
    for n in (0, 1, 2):
        out = work_dir / f"_pin_f{n}.rgb"
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-y", "-i", str(video),
                 "-vf", f"select=eq(n\\,{n})", "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", str(out)],
                check=True, capture_output=True, timeout=60)
            arr = np.fromfile(out, dtype=np.uint8).astype(np.float32)
            if arr.size:
                f[n] = arr
        except Exception:
            pass
        finally:
            out.unlink(missing_ok=True)
    mads: list[float] = []
    for a_, b_ in ((0, 1), (1, 2)):
        if a_ in f and b_ in f and f[a_].size == f[b_].size:
            mads.append(float(np.abs(f[a_] - f[b_]).mean()))
    if prev_video is not None and 0 in f:
        out = work_dir / "_pin_prev_last.rgb"
        try:
            # 末帧稳健取法:截末尾 0.5s 倒放取第一帧 = 真末帧(低帧率片
            # 上 -sseof -0.05 会落到最后一帧之后,解码不出任何帧)。
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-y", "-sseof", "-0.5",
                 "-i", str(prev_video), "-vf", "reverse",
                 "-frames:v", "1", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", str(out)],
                check=True, capture_output=True, timeout=60)
            arr = np.fromfile(out, dtype=np.uint8).astype(np.float32)
            if arr.size and arr.size == f[0].size:
                mads.append(float(np.abs(arr - f[0]).mean()))
        except Exception:
            pass
        finally:
            out.unlink(missing_ok=True)
    return max(mads) if mads else None


def _drop_first_frame(video: Path, cond: dict, *,
                      measured_prev: Optional[Path] = None) -> Path:
    """接缝去重帧(用户规矩 2026-07-30:"首帧的时候直接切掉首帧",
    改在【生成时】完成,下游评审/修复/拼装看到的都是切好的版本):
    - 硬锁路线(measured_prev=None):首帧由 API 通道锁死在上一镜尾帧
      上(ti2v_prev_last 的 first_frame 参数 / flf2v_bridge 的首锚),
      重复是 API 保证的 → 无条件切一帧;
    - 软锁路线(传 measured_prev):prompt 话术锁不保证服从 —— 先量
      (上一镜末帧 vs 本镜首帧 MAD < 阈值)证实确为重复才切;没服从时
      首帧是真内容,切了丢帧还毁掉评审要抓的"未按 @Image1 开场"证据。
    切割走 _trim_head(解码级 + 裁后自检);任何环节失败 → 原样返回。
    一切结果如实记入 cond(dedup_first_frame / junction_mad)。"""
    from .timeline import _probe_fps

    if measured_prev is not None:
        mad = _first_last_mad(measured_prev, video, video.parent)
        cond["junction_mad"] = (round(mad, 2) if mad is not None else None)
        if mad is None or mad >= _DUP_FRAME_MAD:
            cond["dedup_first_frame"] = False
            return video
    fps_ = _probe_fps(video) or 24.0
    t = _trim_head(video, 1.0 / max(1.0, fps_),
                   video.with_name(video.stem + "_dedup.mp4"))
    cond["dedup_first_frame"] = bool(t)
    return Path(t) if t else video


def _ensure_cast_portraits(storyboard, asset_memory, video_gen,
                           cache_dir: Path, library=None,
                           llm=None) -> list[dict]:
    """§A' 角色官方肖像(2026-07-31 用户批准的视觉锚,ViMax 肖像库
    嫁接我们的记忆体系;单视图,不固定 seed):

    对每个 cast 角色,按优先级取像:
      ① 用户素材命中(描述符与素材语义词重叠 ≥0.5)→ 用户照片即肖像;
      ② 跨片角色库命中(同名 + 描述符重叠 ≥0.6)→ 复用历史肖像;
      ③ t2i 生成:static 半句逐字 + 全片 setting/光线(吸取 ViMax 白底
        重打光之坑 —— 肖像直接生成在影片风格里),入库供下部片复用。
    产物写进 storyboard.portraits(持久化)并注册进 asset_memory
    (identity 前缀 portrait:),媒体目录/检索链即刻可见。
    诚实链:t2i 失败 → 该角色无肖像,文字契约照旧兜底,响亮记录。"""
    notes: list[dict] = []
    if not getattr(storyboard, "cast", None):
        return notes
    pdir = cache_dir / "portraits"
    for name, desc in storyboard.cast.items():
        if name in (storyboard.portraits or {}):
            continue
        static = _static_half(desc)
        got, src = None, ""
        # ① 用户素材
        for ident in (getattr(asset_memory, "identity_anchors", None)
                      or {}).values():
            a_desc = str(getattr(ident, "description", "") or "")
            a_path = Path(str(getattr(ident, "source", "") or ""))
            if not a_path.is_file() or a_path.suffix.lower() not in (
                    ".png", ".jpg", ".jpeg", ".webp"):
                continue
            wa, wb = set(re_words(static)), set(re_words(a_desc))
            if wa and wb and len(wa & wb) / min(len(wa), len(wb)) >= 0.5:
                got, src = a_path, "user_asset"
                break
        # ② 跨片库
        if got is None and library is not None:
            hit = library.lookup(name, static)
            if hit is not None:
                got, src = Path(hit), "library"
        # ③ t2i 生成(不固定 seed,用户裁决)
        if got is None and video_gen is not None                 and hasattr(video_gen, "text_to_image"):
            pdir.mkdir(parents=True, exist_ok=True)
            slug = "".join(c if c.isalnum() else "_" for c in name)[:40]
            # 背景必须是影片场景本身(2026-07-31 用户裁决:肖像背景按
            # 场景描述来)—— setting 为空说明 §A 时序被破坏,响亮告警,
            # 诚实退到中性底,绝不写 "the film scene" 这类空话给模型。
            setting = str(getattr(storyboard, "setting", "") or "").strip()
            if not setting:
                log.warning("portrait for %s: storyboard.setting is EMPTY — "
                            "portrait background cannot follow the film "
                            "scene (check §A ordering); using a neutral "
                            "backdrop honestly", name)
            # t2i 翻译护栏(2026-08-05 动态语言):中文正典/设定直通
            # flux 会画坏(2026-07-31 实测)→ LLM 先译英文视觉词。
            if llm is not None and re.search(r"[一-鿿]", static + setting):
                for _field in ("static", "setting"):
                    _v = locals()[_field]
                    if not re.search(r"[一-鿿]", _v):
                        continue
                    try:
                        _t = str(llm.complete(
                            "Translate to concise ENGLISH visual words for "
                            "an image model (appearance/scene only, keep "
                            "all colors). Reply with the translation only: "
                            + _v) or "").strip()
                    except Exception:
                        _t = ""
                    if _t and not re.search(r"[一-鿿]", _t):
                        if _field == "static":
                            static = _t[:400]
                        else:
                            setting = _t[:300]
            bg = (f"Background: {setting} — the character stands inside "
                  f"this exact scene, lit by its natural light."
                  if setting else
                  "Background: plain neutral backdrop, soft even light.")
            prompt = (f"full-body portrait of {name}: {static}. Standing, "
                      f"natural pose, facing the camera, whole figure "
                      f"visible. {bg} cinematic still, high detail")
            if re.search(r"[一-鿿]", static + setting):
                log.warning("portrait for %s: cast descriptor / setting "
                            "contain CJK text — model I/O must be ENGLISH "
                            "(scene_write must emit English descriptors); "
                            "portrait quality will suffer", name)
            try:
                got = Path(video_gen.text_to_image(
                    prompt, pdir / f"{slug}.png"))
                src = "t2i"
                if library is not None:
                    library.add(name, static, got)
            except Exception as exc:
                log.warning("portrait for %s failed (%s) — text contract "
                            "remains the only identity carrier", name, exc)
        if got is not None:
            storyboard.portraits[name] = str(got)
            try:
                from ..types import Identity

                asset_memory.identity_anchors[f"portrait:{name}"] = Identity(
                    identity_id=f"portrait:{name}", name=name,
                    source=str(got),
                    description=f"character: official portrait of {name} — "
                                f"{static}")
            except Exception:
                pass
            log.info("portrait: %s ← %s (%s)", name, Path(got).name, src)
            notes.append({"stage": "cast_portrait", "name": name,
                          "via": src, "path": str(got)})
        else:
            notes.append({"stage": "cast_portrait", "name": name,
                          "via": "none"})
    storyboard._save()
    return notes


def _final_cut(storyboard, cache_dir: Path) -> tuple[list, list]:
    """§E 第一步(用户规矩 2026-07-30):终版路径【确定并核验】——
    台账 video_path 是唯一权威;逐镜打印 "label → 文件 [策略]" 清单;
    文件缺失响亮告警 + 记 decisions 后跳过,绝不静默拼错片。

    接缝切割不在这里做(2026-07-30 二次简化,按用户提议前移):
    extend 裁头与首帧去重都发生在【生成时】(_generate_with_condition
    的 extend/_drop_first_frame),下游全链看到的已是切好的版本。"""
    out: list = []
    notes: list[dict] = []
    for e in storyboard.entries:
        p = Path(e.video_path) if e.video_path else None
        strat = str(((e.condition or {}).get("strategy")) or "")
        if p is None or not p.exists():
            log.warning("assemble: %s final video MISSING (%s) — skipped",
                        e.label, e.video_path)
            notes.append({"stage": "assemble", "label": e.label,
                          "action": "skip_missing",
                          "path": str(e.video_path)})
            continue
        # M2:转场片(add_transition 产物)插在本镜之前;文件丢失响亮跳过
        tp = Path(e.transition_path) if getattr(e, "transition_path", None) \
            else None
        if tp is not None:
            if tp.exists():
                log.info("assemble: %s ← transition %s", e.label, tp.name)
                out.append(tp)
                notes.append({"stage": "assemble", "label": e.label,
                              "action": "transition_inserted",
                              "path": str(tp)})
            else:
                log.warning("assemble: %s transition MISSING (%s) — "
                            "skipped", e.label, e.transition_path)
                notes.append({"stage": "assemble", "label": e.label,
                              "action": "transition_missing",
                              "path": str(e.transition_path)})
        log.info("assemble: %s → %s [%s]", e.label, p.name, strat or "?")
        out.append(p)
    return out, notes


def _name_slot_map(slots) -> dict:
    """槽位清单 → {角色名: 记号}。引用铁律(2026-08-04 用户令):prompt
    里只许用记号指称角色 —— 一切确定性写手(对白句等)出字前查这张表。"""
    return {r["name"]: r["slot"] for r in (slots or [])
            if r.get("name") and r.get("referenceable")}


def _with_dialogue(prompt: str, entry, cast: dict,
                   name_to_slot: Optional[dict] = None) -> str:
    """对白镜的口型子句(2026-07-29 音频线):确定性追加在最终 prompt 尾,
    brain/enhancer 都不写它(skill 已注明)。台词已在 prompt 里(引号串
    去重)→ 原样返回。压制背景音是本子句的核心:BGM 由 §F 统一配,
    生成端只许出人声,两层互不打架。
    引用铁律(2026-08-04 用户令):说话人用槽位记号指称 —— 名字对视频
    模型没有意义,多人同框时记号决定口型给谁;清单里没这个角色(真没
    参考图,如 t2v 降级)才落名字。"""
    line = str(getattr(entry, "dialogue", "") or "").strip()
    if not line or not prompt:
        return prompt
    # 台词逐字硬闸(2026-08-05 run10 实跑事故):brain 按英文法把台词
    # 也翻译了("whispers: \"How pitiful…\""),中文查重匹配不上 → 兜底
    # 又补中文原句,英中双份,模型可能开口说英文。确定性剥除:一切
    # "言说动词 + 引号"里【不是原句】的台词,连动词短语整段删。
    def _drop_foreign(m: "re.Match") -> str:
        return "" if line not in m.group(0) else m.group(0)
    prompt = re.sub(
        r'(?:\b(?:says?|said|saying|whispers?|whispering|replies|replied|'
        r'asks?|asking|murmurs?|shouts?|speaks?|speaking)\b[^"“”「」]{0,60}?'
        r'|(?:低声)?说道?[:\uff1a]?\s*|低语[:\uff1a]?\s*|回答[:\uff1a]?\s*|问道?[:\uff1a]?\s*'
        r'|喊道?[:\uff1a]?\s*)'
        r'["“「][^"“”「」]+["”」]',
        _drop_foreign, prompt)
    prompt = re.sub(r"\s{2,}", " ", prompt).strip()
    zh_mode = bool(re.search(r"[一-鿿]", prompt) or re.search(r"[一-鿿]", line))
    # 压制句与台词解耦(2026-08-05 run11b 事故:brain 把台词写进节拍后,
    # 查重提前返回把"无背景音乐"压制句一起跳过 → 可灵自由配乐)。对白镜
    # 无论台词谁写的,压制句永远确保在场。
    _audio_zh = "音频:只有角色说这句台词的人声——无背景音乐、无音效。"
    _audio_en = ("Audio: only the character's voice speaking the line — "
                 "no background music, no sound effects.")
    def _ensure_audio(p_: str) -> str:
        if "无背景音乐" in p_ or "no background music" in p_:
            return p_
        return f"{p_} {_audio_zh}" if zh_mode else f"{p_} {_audio_en}"
    # 查重按台词原文(不带引号):中英/全半角引号形态都算已在场。
    # 说话人失锚硬闸(2026-08-05 run12 事故:"他严厉地公开退婚并说:"
    # —— 台词在场但说话人是代词,模型只能猜"他"是谁):台词已在场时,
    # 确定性把言说动词前的主语替换成说话人记号。
    # 说话人先解析(闸门与兜底共用)
    who = (getattr(entry, "dialogue_speaker", "") or "").strip()
    if not who or (cast and who not in cast):
        who = next(iter(_cast_in_shot(entry.description, cast)),
                   "the character")
    if line in prompt:
        _subj = (name_to_slot or {}).get(who) or who
        def _fix_speaker(m: "re.Match") -> str:
            seg, verb, quote = m.group(1), m.group(2), m.group(3)
            if _subj and _subj in seg:
                return m.group(0)                  # 段内已有记号 → 不动
            head = prompt[max(0, m.start() - len(_subj)):m.start()]
            if _subj and head.endswith(_subj):
                return m.group(0)                  # 记号紧邻 → 不动
            # 内容保全(2026-08-05 run13c 事故:整段替换吃掉台词前的
            # 表演文字):有代词 → 只把第一个 他/她 换成记号;无代词
            # (隐主语)→ 言说动词前【插入】记号,段落原文保留。
            _pn = list(re.finditer(r"[他她](?!们)", seg))
            if _pn:
                # 换【离言说动词最近】的代词(前面的代词可能指别人)
                last = _pn[-1]
                seg = seg[:last.start()] + _subj + seg[last.end():]
                return f"{seg}{verb}{quote}"
            return f"{seg}{_subj}{verb}{quote}"
        prompt = re.sub(
            "([^。;;!!??<>]{0,40}?)"
            "((?:并|随后|然后|接着)?(?:低声)?说道?|says?)"
            "([:\uff1a]?\\s*[\"\u201c]" + re.escape(line) + "[\"\u201d])",
            _fix_speaker, prompt, count=1)
        return _ensure_audio(prompt)
    # 2026-08-04 用户裁决:兜底只补台词本身 —— "mouth moving"/收势句是
    # 特定镜头的收势指导,被机械化成万能后缀是错的;镜头怎么收由剧本
    # end_state 决定,brain 逐镜写。BGM 压制句保留(no-BGM 裁决在岗)。
    subj = (name_to_slot or {}).get(who) or who
    # prompt 语言随剧本(2026-08-05 用户令):中文 prompt 用中文脚手架
    if zh_mode:
        return _ensure_audio(f'{prompt} {subj}说:"{line}"。')
    return _ensure_audio(f'{prompt} {subj} says: "{line}".')


def _scrub_setting_sentence(text: str, setting: str, strategy: str) -> str:
    """P0-A 加固(2026-07-18 二轮):锚定路线出口的建景句确定性拦截。
    canonical setting 原句(大小写不敏感)整句出现在 prompt 里 → 替换为
    preserve 句(整句替换安全,不剪碎句子;最毒的一类噪声不再赌 skill
    自觉);只出现改写片段(无法安全定界)→ 响亮告警不动刀。无锚路线
    不拦 —— 那里 setting 是唯一的场景载体。"""
    if strategy not in _ANCHORED_STRATEGIES:
        return text
    out = str(text or "")
    canon = str(setting or "").strip().rstrip(".")
    if not canon or not out:
        return out
    pat = re.compile(re.escape(canon), re.IGNORECASE)
    if pat.search(out):
        out = pat.sub(_PRESERVE_CLAUSE.rstrip("."), out, count=1)
        log.info("anchored route: canonical setting sentence in the prompt "
                 "replaced with the preserve clause (scene lives in the "
                 "anchor, not the text)")
        return out
    content = {w for w in re_words(canon) if len(w) >= 4}
    if content:
        hit = content & {w for w in re_words(out)}
        if len(hit) / len(content) >= 0.7:
            log.warning("anchored-route prompt appears to paraphrase the "
                        "canonical setting sentence (%d/%d content words) — "
                        "cannot excise safely; the writer must be fixed via "
                        "skill, passing through unmodified",
                        len(hit), len(content))
    return out


def _cast_in_shot(description: str, cast: dict) -> dict:
    """按 <标记> 求本镜出场角色的 cast 子集(大小写不敏感全等匹配)。
    诚实降级:描述无任何标记(旧剧本/兜底层)或标记全都对不上 cast 键
    → 返回全量(宁多注入,绝不静默丢契约)。"""
    if not cast:
        return {}
    marks = {m.strip().lower() for m in _MARKER_RE.findall(description or "")}
    if not marks:
        return dict(cast)
    hit = {k: v for k, v in cast.items() if k.strip().lower() in marks}
    return hit or dict(cast)


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
    # 视频素材打标(2026-07-17 裁决):【原生视频理解】优先 —— shot 可能
    # 直接续用用户片段,标签必须描述整段(身份词+场景+运动),不是单帧;
    # 假设素材段不长,Gemini-flash 直接看整段。降级链:原生视频 caption →
    # 中间帧 caption(无视频通道的 VLM)→ 文件名(末端,响亮)。
    # 抽帧能力本身保留:图计划的 video_extract 仍用它取 key image。
    for sid, shot in (getattr(asset_memory, "video_shots", None) or {}).items():
        cap = (getattr(shot, "caption", "") or "").strip()
        src = getattr(shot, "source_video", "") or ""
        if cap or not src or not Path(src).exists():
            continue
        text = ""
        if mllm is not None:
            native = getattr(mllm, "caption_video", None)
            if native is not None:
                try:
                    text = (native(src) or "").strip()
                except Exception as exc:
                    log.warning("native video caption failed for %s: %r — "
                                "falling back to a middle-frame caption",
                                Path(src).name, exc)
        if not text and mllm is not None and cache_dir is not None:
            out_dir = Path(cache_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dur_s = _probe_seconds(Path(src))
            mid_idx = max(0, int(dur_s * 12)) if dur_s > 0 else 10 ** 6
            frame = extract_frame(Path(src), mid_idx,
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
            cap = (getattr(shot, "caption", "") or "").strip() or Path(src).stem.replace("_", " ")
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
    是纯续段)。

    2026-07-29 现场事故修正(movie_20260729_150307 接缝闪烁的根因):
    旧实现 `-ss` 前置 + `-c copy` 是流拷贝,只能在关键帧处下刀 —— AI
    生成短片常常整段只有一个关键帧,结果【一帧没裁】,只把容器时长
    元数据改小;帧数与时长不符的文件进 concat 后按错误时长排偏移,
    接缝处两镜的帧交错播放 = 画面闪烁。改为解码级精确裁(`-ss` 后置 +
    重编码,音轨一并裁齐),并加【裁后自检】:输出时长 ≉ 原时长-裁量
    → None(诚实降级,坏文件永远到不了拼接台)。
    ffmpeg 缺失/失败 → None(调用方带痕降级用未裁版本)。"""
    if not shutil.which("ffmpeg"):
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_dur = _probe_seconds(video)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video),
             "-ss", f"{max(0.0, seconds):.3f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-avoid_negative_ts", "make_zero", str(out_path)],
            capture_output=True, timeout=300,
        )
    except Exception:
        return None
    if r.returncode != 0 or not out_path.exists()             or out_path.stat().st_size == 0:
        return None
    if src_dur > 0:
        got = _probe_seconds(out_path)
        if abs(got - max(0.0, src_dur - seconds)) > 0.6:
            log.warning("trim_head self-check failed (src=%.2fs cut=%.2fs "
                        "got=%.2fs) — refusing the trimmed file",
                        src_dur, seconds, got)
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


def decision_prompt(skill_text: str, menu: list, context: dict) -> str:
    """window 决策(image-plan / generation-condition)喂给 brain 的完整
    prompt。S0(2026-07-18):单源导出 —— 训练数据集构建器 import 同一个
    函数重建 prompt,训练分布与生产分布逐字符一致(Crayotter 的
    "一处 schema 三处消费" 原则)。"""
    return (
        skill_text
        + "\n\nTHIS TURN (JSON):\n"
        + json.dumps({"menu": menu, **context}, ensure_ascii=False)
        + '\n\nSTRICT JSON only: {"strategy": "<name from menu>", '
          '"reason": "<one short sentence>", ... optional semantic fields '
          "per the skill above (images / video_prompt / use_prev_tail_video)}"
        + (" reason AND video_prompt in CHINESE (excerpt the screenplay's"
           " wording; tokens inline in Chinese sentences); ONLY image"
           " descriptions stay ENGLISH (English-biased image models)."
           if context.get("prompt_language") == "zh" else
           " ALL output text (reason / video_prompt / image descriptions)"
           " must be in ENGLISH, regardless of the user's language.")
    )


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
    prompt = decision_prompt(skill_text, menu, context)
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
    did = brain_log(f"window/{kind}", {
        "label": context.get("shot", {}).get("label")
        if isinstance(context.get("shot"), dict) else None,
        "menu": sorted(valid), "raw": raw, "parsed": dict(out),
        "usable": True, **skill_proof})
    out["decision_id"] = did
    return out


def _extract_frame0(video: Path, out_path: Path) -> Optional[Path]:
    """本镜第 0 帧(M2 转场闭包用)。ffmpeg 缺/失败 → None(不猜)。"""
    if not shutil.which("ffmpeg"):
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-vf", "select=eq(n\\,0)",
             "-frames:v", "1", str(out_path)],
            capture_output=True, timeout=60)
    except Exception:
        return None
    if r.returncode != 0 or not out_path.exists() \
            or out_path.stat().st_size == 0:
        return None
    return out_path


def _write_screenplay(llm, user_prompt: str, screenplay,
                      asset_catalog: list) -> tuple[str, str]:
    """§A0 剧本(M2 新链):用户给剧本 → 原样采用;否则 LLM 按 screenplay
    技能从 idea 写;LLM 不可用/输出不合格 → idea 原文直通(响亮记录,
    下游按旧链行为运转)。"""
    if screenplay and str(screenplay).strip():
        brain_log("window/screenplay", {"via": "user", "usable": True,
                                        "chars": len(str(screenplay))})
        return str(screenplay).strip(), "user"
    if llm is not None:
        skill_text = _skill_body_named("screenplay")
        prompt = (
            skill_text + "\n\nTHIS TASK (JSON):\n"
            + json.dumps({"idea": user_prompt,
                          "asset_catalog": asset_catalog},
                         ensure_ascii=False)
            + '\n\nSTRICT JSON only: {"screenplay": "<the full screenplay '
              "in the SAME LANGUAGE as the idea (a Chinese idea gets a "
              "Chinese screenplay): scene headings, characters in visible "
              'action, at most one short spoken line per beat>"}')
        raw = ""
        try:
            raw = llm.complete(prompt)
            data = _extract_json(raw)
        except Exception:
            data = None
        text = (str(data.get("screenplay", "")).strip()
                if isinstance(data, dict) else "")
        brain_log("window/screenplay", {
            "raw": raw, "usable": bool(text), "skill": "screenplay",
            "skill_chars": len(skill_text),
            "context": {"idea": user_prompt}})
        if text:
            return text, "llm"
    log.warning("screenplay: no LLM / unusable output — passing the raw "
                "idea straight to storyboarding")
    return str(user_prompt), "idea_passthrough"


_BG_CROWD_WORDS = ("people", "person", "guest", "guests", "crowd",
                   "crowds", "officer", "officers", "noble", "nobles",
                   "dancer", "dancers", "couple", "couples", "servant",
                   "servants", "attendant", "attendants", "figure",
                   "figures", "man", "men", "woman", "women", "soldier",
                   "soldiers", "aristocrat", "aristocrats")

_BG_EMPTY_SUFFIX = (", no principal characters, background figures "
                    "anonymous and unobtrusive at the periphery, open "
                    "central area, no modern objects.")


def _scrub_bg_prompt(prompt: str, cast_names) -> str:
    """场景板 prompt 出口闸(2026-08-04 run8 背景污染事故):角色名剥除、
    含人群词的逗号段整段丢弃、恒定追加空景后缀。确定性,不信 LLM 自觉。
    (事故:'masked aristocratic guests, uniformed officers' 写在建景句里
    + 'Location context: 安娜走向殿下…' → flux 画出一对现代婚纱新人,
    帧升级又把它扩散进全片。)"""
    out = str(prompt or "")
    for n in (cast_names or []):
        out = out.replace(str(n), "")
    # 2026-08-05 用户令:板上无主角,但剧本要的人群放行 —— 人群词不再
    # 剥除,只剥角色名 + 恒定无主角后缀。
    out = re.sub(r"\s{2,}", " ", out).strip().rstrip(".")
    return out + _BG_EMPTY_SUFFIX


def _write_bg_prompts(llm, storyboard, bg_keys: list) -> tuple[dict, str]:
    """§A2 场景板 prompt(scene_image skill):每 bg_id 一条空景 t2i
    prompt;LLM 缺/失败 → 确定性模板(setting 过闸,无 Location
    context —— 那句就是幽灵新人的来源)。返回 ({bg: prompt}, via)。"""
    cast_names = sorted((storyboard.cast or {}).keys())
    fallback = {bk: _scrub_bg_prompt(
        f"Establishing frame of the empty location: {storyboard.setting}. "
        f"Wide shot, eye level, deep focus", cast_names) for bk in bg_keys}
    if llm is None:
        return fallback, "fallback"
    skill_text = _skill_body_named("scene_image")
    task = {"setting": storyboard.setting,
            "backgrounds": {bk: [
                _SHOT_PREFIX_RE.sub("", _strip_markers(e.description))[:160]
                for e in storyboard.entries
                if (getattr(e, "bg_id", "") or f"scene_{e.scene_idx}") == bk
            ][:3] for bk in bg_keys}}
    raw = ""
    try:
        raw = llm.complete(skill_text + "\n\nTHIS TASK (JSON):\n"
                           + json.dumps(task, ensure_ascii=False))
        data = _extract_json(raw)
    except Exception:
        data = None
    got = {}
    if isinstance(data, dict) and isinstance(data.get("backgrounds"), dict):
        for bk in bg_keys:
            p = ((data["backgrounds"].get(bk) or {}).get("prompt") or "") \
                if isinstance(data["backgrounds"].get(bk), dict) \
                else str(data["backgrounds"].get(bk) or "")
            if str(p).strip():
                got[bk] = _scrub_bg_prompt(str(p), cast_names)
    brain_log("window/scene_image", {
        "raw": raw[:2000], "usable": bool(got), "skill": "scene_image",
        "parsed": got or None})
    if got:
        for bk in bg_keys:
            got.setdefault(bk, fallback[bk])
        return got, "llm"
    return fallback, "fallback"


def _apply_caption_canon(cast_canon: dict, given_caps: Optional[dict]) -> None:
    """用户裁决(2026-08-04):有参考图的角色,外观唯一法源是 VLM 对图的
    理解 —— static 半边【确定性覆盖】为图注原文,LLM 版本整段丢弃(skill
    约束只是软防,这里是硬闸:LLM 再怎么脑补也出不了门)。dynamic 半边
    保留 LLM 从剧本读出的内容(道具/姿态归剧本管)。原地修改。"""
    for _gn, _cap in (given_caps or {}).items():
        if not _cap or _gn not in cast_canon:
            continue
        _old = str(cast_canon[_gn])
        _dyn = _old.split("dynamic:", 1)[1].strip().rstrip(";") \
            if "dynamic:" in _old else ""
        cast_canon[_gn] = f"static: {_cap}; dynamic: {_dyn or 'as scripted'}"
        log.info("cast canon: %r static half OVERRIDDEN by the VLM image "
                 "caption (LLM appearance text discarded)", _gn)


def _extract_characters(llm, screenplay: str,
                        given: Optional[dict] = None) -> tuple[dict, str]:
    """§A1 角色提取(ViMax character_extractor 规则移植):剧本 →
    {名字: "static: ...; dynamic: ..."} 正典。`given` = 用户剧本 JSON 的
    钦定角色 {名字: 图像打标描述} —— 名字逐字采用、以图为法源;缺失的
    given 名字由确定性兜底补齐(引用链不许因提取遗漏而断)。失败 → {}。"""
    if llm is None or not str(screenplay).strip():
        # 无 LLM 也要保 given 链:名字+图像描述直接成正典
        if given:
            return ({n: f"static: {c or 'as pictured in the official image'}"
                        f"; dynamic: as scripted"
                     for n, c in given.items()}, "given_only")
        return {}, "skipped"
    skill_text = _skill_body_named("character_extract")
    task: dict = {"screenplay": screenplay}
    if given:
        task["given_characters"] = {
            n: {"image_look": (c or "official image provided; caption "
                                    "unavailable")}
            for n, c in given.items()}
    prompt = (
        skill_text + "\n\nTHIS TASK (JSON):\n"
        + json.dumps(task, ensure_ascii=False)
        + '\n\nSTRICT JSON only: {"characters": {"<name>": "static: '
          '<physique/face/hair — near-invariant traits, ENGLISH>; '
          'dynamic: <attire/accessories/props that may vary>"}}')
    raw = ""
    try:
        raw = llm.complete(prompt)
        data = _extract_json(raw)
    except Exception:
        data = None
    chars: dict = {}
    if isinstance(data, dict) and isinstance(data.get("characters"), dict):
        chars = {str(k): str(v) for k, v in data["characters"].items()
                 if str(v).strip()}
    # 钦定角色确定性兜底:提取漏了谁就补谁(名字是引用链的钥匙,
    # 绝不许因 LLM 遗漏而断);以图像打标为 static 法源。
    for n, c in (given or {}).items():
        if n not in chars:
            log.warning("character_extract: given character %r missing "
                        "from the LLM output — backstopped from its "
                        "image caption", n)
            chars[n] = (f"static: {c or 'as pictured in the official image'}"
                        f"; dynamic: as scripted")
    brain_log("window/character_extract", {
        "raw": raw, "usable": bool(chars), "skill": "character_extract",
        "skill_chars": len(skill_text),
        "parsed": ({"characters": chars} if chars else None)})
    return chars, ("llm" if chars else "unusable")


def _is_mostly_chinese(text: str) -> bool:
    """语言拒收闸判据(2026-08-05 run13 事故:英文润色夹着中文台词引号,
    "含任何中文"的旧判据被一个字骗过):剥掉引号内台词后,正文的中文
    字符数必须不少于拉丁字母数(记号 image_N 的拉丁字母已计入,纯中文
    prompt 依然轻松达标)。"""
    body = re.sub(r'["“][^"“”]*["”]', "", str(text or ""))
    cjk = len(re.findall(r"[一-鿿]", body))
    latin = len(re.findall(r"[A-Za-z]", body))
    return cjk > 0 and cjk >= latin


def _prompt_lang(text: str) -> str:
    """prompt 语言随剧本(2026-08-05 用户令):剧本含中文 → 全链视频
    prompt 用中文(记号内嵌中文句、动作摘抄原文);否则英文。"""
    return "zh" if re.search(r"[一-鿿]", str(text or "")) else "en"


def _write_outline(llm, user_prompt: str, asset_catalog: list,
                   episode_guidance: dict, max_shots: int,
                   fallback_fn,
                   cast_canon: Optional[dict] = None,
                   prompt_language: str = "en"
                   ) -> tuple[list[str], list, list[str], dict, str]:
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
                          "prompt_language": prompt_language,
                          "cast_canon": (cast_canon or {}),
                          "asset_catalog": asset_catalog,
                          "episode_guidance": {
                              "past_task_shapes":
                                  episode_guidance.get("past_task_shapes", []),
                          },
                          "max_shots_hard_cost_cap": max_shots},
                         ensure_ascii=False)
            + '\n\nSTRICT JSON only: {"cast": {"<entity name>": "<10-20 '
              'word CANONICAL appearance descriptor (species/build, coat/'
              'wardrobe with colors, distinctive marks) — every shot prompt '
              'will restate it VERBATIM>"}, "setting": "<one canonical '
              'set-dressing + lighting sentence for the (main) scene>", '
              '"shots": [{"description": "Shot 1: '
              '<detailed filmable description — mark every cast character '
              'as <name> in angle brackets, names copied from cast keys>", '
              '"duration_s": <int 4-10>, '
              '"end_state": "<one sentence: at the CUT, who/what is where, '
              'moving or still, in which direction — PLUS the camera\'s '
              'own state (static / pushing in / tracking right at walking '
              'pace ...)>", '
              '"variation": "large|medium|small (expected first-to-last '
              'frame change inside this shot)", '
              '"opening_frame": "<ONLY for the first shot and scene cuts: '
              'a purely STATIC opening snapshot (no ongoing actions); omit '
              'for continuing shots>", '
              '"dialogue": {"speaker": "<the cast key of WHO SPEAKS — copied '
              'verbatim from cast>", "line": "<ONE spoken line of at most '
              '6 words>"} — include ONLY when a cast character visibly '
              'speaks on screen (medium close-up or closer); omit '
              'otherwise, '
              '"bg": "<background id like bg_1 — keep the SAME id while '
              'the shot happens in the same physical space (the master '
              'background is unchanged); switch to a NEW id ONLY when the '
              'action moves to a different space. Predicting this drives '
              'which background reference image the generator receives>"}, '
              '...], "music_plan": {"scene 1": "<ONE music description '
              'for the whole scene: mood, genre, tempo/BPM — all shots in '
              'a scene share one track; omit a scene (or the whole field) '
              'for silence>"}} '
              "— each description 15-40 words (subject + action + "
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
              "on it (write that event into the description). SCRIPT LANGUAGE LAW: "
              + ("EVERYTHING (descriptions, end_state, opening_frame, "
                 "cast descriptors, setting) MUST be in CHINESE, "
                 "EXCERPTING the screenplay's own action and performance "
                 "wording verbatim wherever it exists — translation is "
                 "loss (image-model strings are translated downstream). "
                 if prompt_language == "zh" else
                 "cast descriptors, setting, descriptions, end_state, "
                 "variation and opening_frame MUST be ENGLISH (they feed "
                 "image/video models directly). ")
              + "Entity NAMES in cast keys and dialogue lines always "
              "stay in the user's language. CAST CANON: when the task "
              "JSON carries a non-empty cast_canon, adopt those names and "
              "descriptors VERBATIM in your cast output — you may only ADD "
              "characters it missed, never rename or rewrite them."
        )
        raw = ""
        data = None
        for _attempt in range(2):        # 空响应/坏 JSON 先重试一次
            try:
                raw = llm.complete(prompt)
                data = _extract_json(raw)
            except Exception:
                data = None
            _shots_ok = (isinstance(data, dict)
                         and isinstance(data.get("shots"), list))
            if _shots_ok and prompt_language == "zh":
                # 语言闸(2026-08-05 run11 事故):zh 项目分镜写成英文 →
                # 纠正重试一次;仍英文 → 判不可用,落摘抄兜底(拆剧本
                # 原文,天然中文)。
                _texts = [str(x.get("description", x) if isinstance(x, dict)
                              else x) for x in data["shots"]]
                _zh_n = sum(1 for t in _texts if re.search(r"[一-鿿]",
                            re.sub(r"<[^>]*>", "", t)))
                if _texts and _zh_n < (len(_texts) + 1) // 2:
                    log.warning("scene_write: %d/%d descriptions are NOT "
                                "Chinese on a zh project — SCRIPT LANGUAGE "
                                "LAW violated%s", len(_texts) - _zh_n,
                                len(_texts),
                                " — retrying with a corrective" if
                                _attempt == 0 else
                                "; falling back to verbatim excerpts")
                    data = None
                    _shots_ok = False
                    if _attempt == 0:
                        prompt += ("\n\nYOUR PREVIOUS REPLY VIOLATED THE "
                                   "SCRIPT LANGUAGE LAW: every description/"
                                   "end_state/opening_frame MUST be in "
                                   "CHINESE, excerpting the screenplay "
                                   "verbatim. Rewrite the SAME storyboard "
                                   "in Chinese.")
                        continue
            if _shots_ok:
                break
            if _attempt == 0:
                log.warning("scene_write: LLM reply unusable (raw %d "
                            "chars) — retrying once before the "
                            "deterministic fallback", len(raw or ""))
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
            shots, durs, ends, variations, openings, dialogues, seen = \
                [], [], [], [], [], [], set()
            speakers, bgs = [], []
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
                    var = str(s_.get("variation", "") or "").strip().lower() \
                        if isinstance(s_, dict) else ""
                    variations.append(
                        var if var in ("large", "medium", "small") else "")
                    openings.append(
                        str(s_.get("opening_frame", "") or "").strip()
                        if isinstance(s_, dict) else "")
                    _dlg = s_.get("dialogue") if isinstance(s_, dict) \
                        else None
                    if isinstance(_dlg, dict):
                        dialogues.append(
                            str(_dlg.get("line", "") or "").strip()[:120])
                        speakers.append(
                            str(_dlg.get("speaker", "") or "").strip())
                    else:
                        dialogues.append(
                            str(_dlg or "").strip()[:120])
                        speakers.append("")
                    bgs.append(str(s_.get("bg", "") or "").strip()
                               if isinstance(s_, dict) else "")
                    # 时长是 brain 的决定,范围写死 [4,10](2026-07-14 裁决)。
                    # brain 没输出/输出非法 → None = 不向 API 传 duration 字段,
                    # 用模型自然默认(用户裁决:绝不 feed 任何预设值)。
                    try:
                        durs.append(max(4, min(10, int(dur))))
                    except (TypeError, ValueError):
                        durs.append(None)
            # 跨镜一致性载体(2026-07-17 审计):cast/setting 官方描述符。
            # brain 没输出 → 空(不编造);有输出 → 全链注入。
            cast = ({str(k): str(v) for k, v in data.get("cast").items()
                     if str(v).strip()}
                    if isinstance(data.get("cast"), dict) else {})
            # M2:提取阶段的角色正典优先(同名覆盖,分镜只许补缺)
            cast = {**cast, **(cast_canon or {})}
            setting = str(data.get("setting", "") or "").strip()
            # LANGUAGE LAW 确定性检查(2026-07-31):描述符/setting 直喂
            # 图像模型,必须英文;名字(cast 键)不查 —— 名字可保留原语言。
            cjk_bad = [k for k, v in cast.items()
                       if re.search(r"[一-鿿]", v)]
            if re.search(r"[一-鿿]", setting):
                cjk_bad.append("<setting>")
            if cjk_bad and prompt_language != "zh":
                log.warning("scene_write: cast descriptor/setting in CJK "
                            "for %s — LANGUAGE LAW violated (must be "
                            "English); portraits and prompts will degrade",
                            cjk_bad)
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
                # music_plan:scene 号 → 音乐描述("scene 1"/"1"/1 皆可)
                mp_raw = data.get("music_plan")
                music_plan: dict = {}
                if isinstance(mp_raw, dict):
                    for k, v in mp_raw.items():
                        desc_ = str(v or "").strip()
                        m_ = re.search(r"(\d+)", str(k))
                        if desc_ and m_:
                            music_plan[int(m_.group(1))] = desc_[:300]
                return shots, durs, ends, \
                    {"cast": cast, "setting": setting,
                     "variations": variations,
                     "opening_frames": openings,
                     "dialogues": dialogues,
                     "dialogue_speakers": speakers,
                     "bgs": bgs,
                     "music_plan": music_plan}, "llm"
    fb = list(fallback_fn())
    # 兜底层没有 brain → None = 不传 duration 字段,API 用自己的自然默认
    # (不是 config 的 shot_duration,也不是我们编的数);end_state 同理为空。
    return fb, [None] * len(fb), [""] * len(fb), \
        {"cast": dict(cast_canon or {}), "setting": "",
         "variations": [""] * len(fb),
         "opening_frames": [""] * len(fb), "dialogues": [""] * len(fb),
         "dialogue_speakers": [""] * len(fb), "bgs": [""] * len(fb),
         "music_plan": {}}, "fallback"


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
        # 2026-07-31 用户裁决:episode 记忆【不再直接继承】—— 只作建议
        # 注入上下文,决策仍由 brain 做(旧 via="episode" 短路废除:
        # 历史策略/图计划可能与本次上下文错配,直接照抄会拿到不对的
        # keyframe)。
        context = {**context, "episode_recommendation": {
            "strategy": replay_hint,
            "note": "verified on a similar PAST task — weigh it as "
                    "advice; current-run conditions win"}}
    picked = _brain_pick(llm, kind, menu, context)
    if picked:
        return {**picked, "via": "llm"}
    for name in priority:
        if name in names:
            d = {"strategy": name, "via": "fallback",
                 "reason": "deterministic priority (brain reply unusable)"}
            d["decision_id"] = brain_log(
                f"window/{kind}",
                {"label": label, "parsed": dict(d), "via": "fallback",
                 "usable": True, "menu": [m["name"] for m in menu],
                 "context": context})
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
                        out_dir: Path,
                        cast: Optional[dict] = None,
                        portrait_paths: Optional[set] = None,
                        has_portrait_cast: bool = False
                        ) -> tuple[str, list, str]:
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
        role_ = roles[idx]
        # P2-⑦(ViMax 借鉴):首帧槽位优先用剧本的【开场静态快照】做
        # t2i 底稿(纯静态构图,无进行中动作 —— 比整条 shot 描述更适合
        # 单帧);没有快照(续接镜/旧剧本)照旧用 shot 描述。
        base_ = (entry.opening_frame
                 if role_ in ("first", "first_frame")
                 and getattr(entry, "opening_frame", "")
                 else entry.description)
        specs.append({"source": "", "description":
                      base_ + (" — the closing frame of this shot"
                               if role_ == "last" else "")})
    produced: list = []
    for i, role in enumerate(roles):
        spec_i = specs[i] if i < len(specs) else {}
        src = spec_i.get("source") or _default_source(video_gen, asset_memory)
        # A1 律(2026-08-04 实跑事故):出场者有官方肖像时,禁止用 t2i
        # 按文字重画首帧再硬钉 —— flux 画的脸/衣服 ≠ 参考图,钉死后
        # 肖像 refer 无力回天。参考路线(ref2v)由肖像直接控画面。
        if has_portrait_cast and role in ("first", "first_frame") \
                and src == "t2i":
            log.warning("image plan: %s t2i FIRST FRAME blocked — cast "
                        "with official portraits must not be re-drawn "
                        "from text and hard-pinned; dropping the slot "
                        "(reference route carries the portraits)",
                        entry.label)
            continue
        # A1 律扩展(run10 实跑):参考槽同样不许 t2i 重画【人】——
        # 出场者全有官方肖像时,文字版人像 = 第二个相互打架的身份锚
        # (skill 已立"不得重复自动附挂之物",这里是硬闸)。道具/
        # 场景类 t2i 参考(描述不含人物词)照常放行。
        if has_portrait_cast and src == "t2i" \
                and role in ("reference", "ref") \
                and re.search(r"\b(character|person|woman|man|girl|boy|"
                              r"lady|gentleman|face|portrait)\b",
                              str(spec_i.get("description") or ""),
                              re.IGNORECASE):
            log.warning("image plan: %s t2i PERSON REFERENCE blocked — "
                        "the cast's official portraits already ride; a "
                        "text-drawn person is a second, competing "
                        "identity anchor; dropping the slot",
                        entry.label)
            continue
        # 出口清洗:query 会成为 t2i prompt / 检索词 —— 剥 <标记>、洗
        # cast 契约标签(brain 抄写描述/契约时可能原样带出)。
        query = _scrub_cast_labels(
            _strip_markers(spec_i.get("description") or entry.description),
            cast)
        if re.search(r"[\u4e00-\u9fff]", query):
            log.warning("image plan: t2i/retrieval query contains CJK "
                        "text (%.60s...) — model I/O must be English; "
                        "generation quality will suffer", query)
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
    # 肖像专用通道守卫(2026-07-31 bug ②a):官方肖像只许经 §C 自动附挂
    # 进引用列表;计划图撞上肖像路径(检索误中/LLM 点名)→ 响亮丢弃,
    # 后续按剩余图正常降级。没有这道闸,同一张肖像会双通道进引用、
    # 甚至被钉成首帧 —— shot2 起整片开场全是肖像照。
    if portrait_paths and produced:
        kept = []
        for row in produced:
            try:
                rp = str(Path(row["path"]).resolve())
            except Exception:
                rp = str(row["path"])
            if rp in portrait_paths:
                log.warning("image plan: %s planned the OFFICIAL PORTRAIT "
                            "as its own image — dropped (portraits ride the "
                            "dedicated auto-attach channel only)",
                            entry.label)
                continue
            kept.append(row)
        produced = kept
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
    query = _strip_markers(query or entry.description)
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
            # 取源片段的【中间】帧(比首帧/末帧更代表内容;审计 2026-07-17:
            # 旧代码 idx=10**6 实取的是末帧,与注释不符 —— 已修正)。
            dur_s = _probe_seconds(src)
            mid_idx = max(0, int(dur_s * 12)) if dur_s > 0 else 10 ** 6
            got = extract_frame(src, mid_idx, out)
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


def _prepared_source_videos(asset_memory, cache_dir: Path) -> list[tuple]:
    """用户源视频 → t2v reference_videos 通道的可用形态(2026-07-17 裁决
    G-1:上镜尾帧+有素材 → 走 t2v,素材视频挂 @VideoN 引用)。

    ≤3 条(官方上限),逐条裁到 ≤15s(官方单条上限,取头段 —— 素材参考
    看的是内容不是接点);带素材语义(caption)供清单/兜底模板引用。
    ffmpeg 不可用且超长 → 原样上传交给 API 把关(留痕)。"""
    out: list[tuple] = []
    if asset_memory is None:
        return out
    for sid, shot in (asset_memory.video_shots or {}).items():
        if len(out) >= 3:
            log.info("source videos capped to 3 (seedance-2.0 limit)")
            break
        src = getattr(shot, "source_video", "") or ""
        sp = Path(src)
        if not src or not sp.exists():
            continue
        use = sp
        if _probe_seconds(sp) > 15.0:
            cut = _head_clip(sp, 15.0,
                             Path(cache_dir) / f"srcvid_{sid}_15s.mp4")
            if cut is not None:
                use = Path(cut)
            else:
                log.warning("source video %s >15s and ffmpeg unavailable — "
                            "uploading as-is (API may reject)", sp.name)
        out.append((use, str(getattr(shot, "caption", "") or sp.stem)))
    return out


_CANON_STOPWORDS = {"and", "the", "with", "her", "his", "a", "an", "of",
                    "in", "on", "or", "has", "wearing"}


def _enforce_cast_canon(text: str, shot_cast,
                        cast: Optional[dict]) -> tuple[str, list[dict]]:
    """E 案(2026-08-02 用户批准):正典描述符【逐字契约】的确定性执行。

    实锤病根:enhancer 把 "dark green raincoat" 改写成 "teal raincoat" ——
    清洗器只认原句精确匹配,改写自由漂移 → 同一角色两副长相。
    检查:本镜出场角色 static 半句的内容词在出门 prompt 里的覆盖率
    < 0.75 → 判定被改写/丢失 → 确定性追加一句正典身份子句(这是恢复
    瘦身法则要求的"唯一身份子句",不是加噪)+ 响亮告警;notes 供
    decisions 记账。空 prompt(走兜底模板,模板自带身份)不处理。"""
    out = str(text or "")
    notes: list[dict] = []
    if not out.strip() or not cast:
        return out, notes
    low = out.lower()
    for name in sorted(shot_cast or []):
        static = _static_half(cast.get(name, ""))
        words = [w for w in re_words(static)
                 if len(w) > 2 and w.lower() not in _CANON_STOPWORDS]
        if len(words) < 3:
            continue
        hit = sum(1 for w in words if w.lower() in low)
        cov = hit / len(words)
        if cov < 0.75:
            missing = [w for w in words if w.lower() not in low]
            log.warning("cast canon: %s's canonical look paraphrased or "
                        "missing in the outgoing prompt (coverage %.0f%%, "
                        "missing %s) — appending the canonical identity "
                        "clause", name, cov * 100, missing[:6])
            out = f"{out.rstrip()} {name}: {static}."
            notes.append({"stage": "cast_canon", "name": name,
                          "coverage": round(cov, 2),
                          "action": "canon_appended"})
    return out, notes


def _ref_tok(video_gen, n: int) -> str:
    """引用记号方言(2026-08-03 百炼迁移):后端声明自己的记号
    (kling: <<<image_N>>>,seedance: @ImageN),槽位清单/兜底模板/
    闸门统一按方言渲染 —— 换模型不再全网替换字符串。"""
    fn = getattr(video_gen, "ref_token", None)
    return fn(n) if callable(fn) else f"@Image{n}"


def _portrait_slot_content(name: str) -> str:
    """肖像槽位的单源语义(B 案 2026-08-02):防拷贝子句内置 —— 写 prompt
    的四个角色全部从这行抄,肖像只许当身份参考,绝不许被复刻构图。"""
    return (f"official portrait of {name} — identity ONLY: match the "
            f"character's face/build/wardrobe to it; NEVER copy its pose, "
            f"framing or background")


def _slot_manifest(strategy: str, entry, prev,
                   use_prev_tail: bool = True,
                   source_videos=None, portraits=None,
                   video_gen=None) -> list[dict]:
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
    # ── 百炼可灵新策略(2026-08-03):编号按方言渲染;i2v_first 的
    # refer 编号不含 first_frame(M0 实测:<<<image_1>>> 指首张 refer)──
    if strategy == "ref2v":
        own = list(refs)
        rows = [{"slot": _ref_tok(video_gen, i + 1), "referenceable": True,
                 "content": _c(p, "a planned reference image")}
                for i, p in enumerate(own)]
        rows += [{"slot": _ref_tok(video_gen, len(own) + j + 1),
                  "referenceable": True, "name": n,
                  "content": _portrait_slot_content(n)}
                 for j, n in enumerate(sorted(portraits or {}))]
        return rows
    if strategy == "i2v_first":
        rows = [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": ("the previous shot's final frame (this shot "
                             "opens exactly on it)" if prev_ok else
                             _c(kf, "this shot's planned opening frame"))}]
        own = list(refs)
        rows += [{"slot": _ref_tok(video_gen, i + 1), "referenceable": True,
                  "content": _c(p, "a planned reference image")}
                 for i, p in enumerate(own)]
        rows += [{"slot": _ref_tok(video_gen, len(own) + j + 1),
                  "referenceable": True, "name": n,
                  "content": _portrait_slot_content(n)}
                 for j, n in enumerate(sorted(portraits or {}))]
        return rows
    if strategy == "flf2v_own_pair" and pf is not None and pl is not None:
        rows = [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": _c(pf, "this shot's planned opening frame")},
                {"slot": "LAST_FRAME", "referenceable": False,
                 "content": _c(pl, "this shot's planned closing frame")}]
    elif strategy == "t2v_own_refs":
        rows = [{"slot": f"@Image{i + 1}", "referenceable": True,
                 "content": _c(p, "a planned reference image")}
                for i, p in enumerate(refs)]
        # 角色官方肖像(2026-07-31 视觉锚):排在自有图之后,编号与
        # _generate_with_condition 的装配顺序严格一致。B 案(2026-08-02):
        # 防拷贝子句写进语义行本身 —— 四个写手(brain/enhancer/兜底/全修)
        # 都从这行抄引用语义,缺了它模型会把画面坍缩成肖像构图(实录:
        # 末镜女主正面呆立 = 肖像标准姿势)。
        rows += [{"slot": f"@Image{len(refs) + j + 1}",
                  "referenceable": True, "name": n,
                  "content": _portrait_slot_content(n)}
                 for j, n in enumerate(sorted(portraits or {}))]
        rows += [{"slot": f"@Video{i + 1}", "referenceable": True,
                  "content": f"user asset: {cap}"}
                 for i, (_v, cap) in enumerate(source_videos or [])]
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
        rows += [{"slot": f"@Image{len(own) + j + 2}",
                  "referenceable": True,
                  "content": _portrait_slot_content(n)}
                 for j, n in enumerate(sorted(portraits or {}))]
        rows += [{"slot": f"@Video{i + 1}", "referenceable": True,
                  "content": f"user asset: {cap}"}
                 for i, (_v, cap) in enumerate(source_videos or [])]
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


def _condition_menu(entry, prev, video_gen, portraits=None) -> list[dict]:
    """当前 shot 可用的条件策略(Image Plan 角色 + 存在性 + 能力三重门控)。

    2026-07-31 bug ②a 补:官方肖像走自动附挂通道,本镜没有自己计划的图
    时参考路线依然可用(肖像即引用)—— 否则封掉"肖像当计划图"后,
    身份锚在无图镜上会整体失联。"""
    caps = video_gen.capabilities() if video_gen is not None else set()
    ff, refs, pf, pl = _entry_images(entry)
    has_kf = ff is not None
    has_portraits = bool(portraits)
    has_prev = prev is not None and prev.video_path is not None
    # ── 百炼可灵菜单(2026-08-03 重构):后端声明 first_frame_plus_refs
    # 能力 → 收缩为 4 条;硬钉首帧可与参考图同请求混用(M0 实测),
    # 软钉时代的 8 条菜单只属于旧后端,互不干扰。──────────────────────
    if "first_frame_plus_refs" in caps:
        menu = [{"name": "t2v", "description": "Text only — nothing else "
                 "is available (last resort)."}]
        if refs or has_portraits:
            menu.append({"name": "ref2v",
                         "description": "Reference-to-video: every planned "
                                        "reference image and official "
                                        "portrait rides the reference "
                                        "channel (<<<image_N>>>). THE "
                                        "route for scene cuts with "
                                        "characters. Mention every slot "
                                        "with its content."})
        if has_prev or has_kf:
            menu.append({"name": "i2v_first",
                         "description": "HARD first-frame pin (API level, "
                                        "not prompt wording): the previous "
                                        "shot's final frame (or this "
                                        "shot's own keyframe on a cut) "
                                        "opens the shot exactly, AND "
                                        "portraits/references ride along "
                                        "in the same call. THE route for "
                                        "in-scene continuation. Prompt "
                                        "describes MOTION only."})
        if pf is not None and pl is not None and "flf2v" in caps:
            menu.append({"name": "flf2v_own_pair",
                         "description": "This shot's OWN first+last frame "
                                        "pair — opens on image 1, closes "
                                        "on image 2 exactly; describe the "
                                        "motion between."})
        if has_prev and has_kf and "flf2v" in caps:
            menu.append({"name": "flf2v_bridge",
                         "description": "Bridge: previous shot's last "
                                        "frame → this shot's keyframe as "
                                        "the closing anchor (continuity "
                                        "AND arrival)."})
        return menu
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
    if (refs or has_portraits) and "ref_images" in caps:
        menu.append({"name": "t2v_own_refs",
                     "description": "This shot's planned REFERENCE image(s) "
                                    "ride the seedance t2v reference channel "
                                    "(@Image1, @Image2…), and the user's "
                                    "source video(s) ride as @VideoN when "
                                    "provided. Soft conditioning; no frame "
                                    "is pixel-locked. Mention every slot "
                                    "with its content."})
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
                                        "natively. CANNOT carry any reference "
                                        "image: when this shot HAS references "
                                        "(portraits / planned images), the "
                                        "REFERENCE-FIRST law (2026-08-02) "
                                        "says pick ti2v_prev_plus_keyframe "
                                        "or t2v_own_refs instead. "
                                        "`video_prompt` must describe "
                                        "ONLY what happens NEXT plus what to "
                                        "maintain — never re-describe what "
                                        "already happened. A planned "
                                        "'last'-role image (if any) becomes "
                                        "the target final frame."})
        if (has_kf or refs or has_portraits) and "ref_images" in caps:
            menu.append({"name": "ti2v_prev_plus_keyframe",
                         "description": "THE route when continuing from the "
                                        "previous shot WHILE carrying "
                                        "materials (2026-07-17 ruling): t2v "
                                        "reference channels — @Image1 = the "
                                        "previous shot's last frame (the "
                                        "prompt must open EXACTLY on it; "
                                        "field-verified to pin the first "
                                        "frame), @Image2(…) = this shot's "
                                        "planned/generated images, @VideoN = "
                                        "the user's source video(s) when "
                                        "provided. Reference ACCURACY is "
                                        "everything — every slot mentioned "
                                        "with its content."})
    # 2026-07-17:multi_image_fusion(kling 融合,无指定首帧)与"首帧引用
    # 优先"方针冲突 → 从菜单退役(执行分支保留兼容旧 episode)。
    return menu


def _generate_with_condition(strategy: str, entry, prev, spec: ShotSpec,
                             video_gen, cache_dir: Path, seed: int,
                             fps: int, window_tail_s: float,
                             brain_prompt: str = "",
                             use_prev_tail_video: bool = False,
                             source_videos: Optional[list] = None
                             , portraits=None) -> tuple[Path, dict]:
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

    if strategy == "ref2v":
        # 百炼可灵参考生视频(2026-08-03):计划图 + 肖像全走 refer 通道,
        # 编号按方言(<<<image_N>>>,顺序 = media 数组顺序)。
        p_names = sorted(portraits or {})
        p_paths = [Path(portraits[n]) for n in p_names]
        all_refs = list(refs) + p_paths
        if all_refs:
            cond.update(reference_images=[str(p) for p in all_refs],
                        anchoring="ref2v")
            fallback = (spec.prompt + ". " + " ".join(
                f"{_ref_tok(video_gen, i + 1)} shows: "
                f"{_desc_of(entry, p_) or 'a planned reference image'} — "
                f"keep it consistent." for i, p_ in enumerate(refs))
                + "".join(
                    f" {_ref_tok(video_gen, len(refs) + j + 1)} is the "
                    f"official portrait of {n} — match the character's "
                    f"appearance to it; never copy its pose or framing."
                    for j, n in enumerate(p_names)))
            return Path(video_gen.generate(
                prompt=brain_prompt or fallback, duration=spec.duration,
                out_path=out, fps=fps, reference_images=all_refs,
                seed=seed)), cond
        strategy = "t2v"
        cond = {"strategy": "t2v", "degraded_from": "ref2v"}

    if strategy == "i2v_first":
        # 百炼可灵硬钉首帧 + 参考同请求(M0 实测可混用):续接镜首选。
        # 首帧 = 上镜末帧(硬钉 → 接缝重复帧无条件切)或本镜关键帧。
        first: Optional[Path] = None
        hard_prev = False
        if prev is not None and prev.video_path:
            first = _last_frame(
                Path(prev.video_path),
                cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
            hard_prev = first is not None
        if first is None and kf is not None:
            first = kf
        if first is not None:
            p_names = sorted(portraits or {})
            p_paths = [Path(portraits[n]) for n in p_names]
            all_refs = list(refs) + p_paths
            cond.update(first_anchor=str(first),
                        reference_images=([str(p) for p in all_refs]
                                          or None),
                        anchoring="hard_first_frame")
            fallback = (spec.prompt + " — the opening frame is already "
                        "exact; describe MOTION only and keep the scene."
                        + "".join(
                            f" {_ref_tok(video_gen, len(refs) + j + 1)} is "
                            f"the official portrait of {n} — identity "
                            f"only, never copy its pose or framing."
                            for j, n in enumerate(p_names)))
            outp = Path(video_gen.generate(
                prompt=brain_prompt or fallback, duration=spec.duration,
                out_path=out, fps=fps, first_frame=first,
                reference_images=(all_refs or None), seed=seed))
            if hard_prev:      # 首帧=上镜末帧,通道级保证重复 → 直接切
                return _drop_first_frame(outp, cond), cond
            return outp, cond
        strategy = "t2v"
        cond = {"strategy": "t2v", "degraded_from": "i2v_first"}

    if strategy == "t2v_own_refs":
        # Image Plan reference 角色图 → seedance t2v @refs(无需上镜)。
        # 2026-07-17 G-1:用户源视频同乘 reference_videos(@VideoN)。
        # 2026-07-31 bug ②a:肖像走自动附挂 —— own 图为空但有肖像时本
        # 路线照常成立(肖像即引用),不再降级丢身份锚。
        p_names = sorted(portraits or {})
        p_paths = [Path(portraits[n]) for n in p_names]
        if refs or p_paths:
            src_vids = [v for (v, _c_) in (source_videos or [])]
            own_refs = list(refs)
            refs = own_refs + p_paths
            cond.update(reference_images=[str(p) for p in refs],
                        reference_videos=([str(v) for v in src_vids] or None),
                        anchoring="soft_t2v_refs")
            # 裁决 1.2:引用必须带内容 —— 每个 @ImageN 说清它实际是什么;
            # 肖像槽位 = 身份参考,绝不是要复刻的画面。
            fallback_prompt = (spec.prompt + ". " + " ".join(
                _mention(entry, p_, i + 1) for i, p_ in enumerate(own_refs))
                + "".join(
                    f" @Image{len(own_refs) + j + 1} is the official "
                    f"portrait of {n} — match the character's appearance "
                    f"to it; do not copy its pose or framing."
                    for j, n in enumerate(p_names))
                + "".join(f" @Video{i + 1} is the user's source video "
                          f"({cap}) — keep its subject consistent."
                          for i, (_v, cap) in enumerate(source_videos or [])))
            return Path(video_gen.generate(
                prompt=brain_prompt or fallback_prompt,
                duration=spec.duration, out_path=out, fps=fps,
                reference_images=refs, seed=seed,
                reference_video=(src_vids or None))), cond
        strategy = "t2v"
        cond = {"strategy": "t2v", "degraded_from": "t2v_own_refs"}

    if strategy == "flf2v_bridge":
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        # 护栏(2026-07-17):收场锚只许 'last' 角色图或首帧角色图 ——
        # 身份参考照片当尾帧 = 强迫镜头结束在照片构图上,语义错误。
        anchor_img = pl if pl is not None else kf
        if last is not None and anchor_img is not None:
            cond.update(first_anchor=str(last), last_anchor=str(anchor_img))
            outp = Path(video_gen.frame_to_frame(
                prompt=brain_prompt or spec.prompt, first_frame=last,
                last_frame=anchor_img,
                out_path=out, duration=spec.duration, seed=seed))
            # 首锚 = 上一镜尾帧(通道级保证)→ 首帧必重复,直接切
            return _drop_first_frame(outp, cond), cond
        strategy = "ti2v_prev_last"      # 尾帧抽不出来 → 逐级降级(如实改写)
        cond = {"strategy": strategy, "degraded_from": "flf2v_bridge"}

    if strategy == "ti2v_prev_plus_keyframe":
        # 上镜尾帧 + 本镜图(可多张参考),一次调用。走【t2v +
        # reference_images】通道(refs 仅在 t2v 端点验证过;软锚 —— 构图级
        # 连续,不锁任何帧;要像素级用 ti2v_prev_last / flf2v_bridge)。
        last = _last_frame(Path(prev.video_path),
                           cache_dir / f"shot{spec.shot_idx:03d}_prev_last.png")
        own = refs if refs else ([kf] if kf is not None else [])
        # 2026-07-31 bug ②a:own 为空但有肖像 → 路线照常([尾帧]+肖像),
        # 编号与 _slot_manifest 一致(肖像永远排在 own 之后)。
        p_names = sorted(portraits or {})
        p_paths = [Path(portraits[n]) for n in p_names]
        if last is not None and (own or p_paths):
            all_refs = [last] + own + p_paths
            src_vids = [v for (v, _c_) in (source_videos or [])]
            cond.update(reference_images=[str(p) for p in all_refs],
                        reference_videos=([str(v) for v in src_vids] or None),
                        anchoring="soft_t2v_refs")
            # 裁决 1.2/G-2:@Image1 = 上镜尾帧,prompt 强锁开场(用户实测
            # t2v 的 @Image1 基本能固定首帧);本镜图/素材视频逐个带实况语义
            fallback_prompt = (
                spec.prompt + ". " + _PIN_SENTENCE + " "
                + " ".join(_mention(entry, p_, i + 2)
                           for i, p_ in enumerate(own))
                + "".join(
                    f" @Image{len(own) + j + 2} is the official portrait "
                    f"of {n} — match the character's appearance to it; do "
                    f"not copy its pose or framing."
                    for j, n in enumerate(p_names))
                + "".join(f" @Video{i + 1} is the user's source video "
                          f"({cap}) — keep its subject consistent."
                          for i, (_v, cap) in enumerate(source_videos or [])))
            outp = Path(video_gen.generate(
                prompt=brain_prompt or fallback_prompt,
                duration=spec.duration, out_path=out, fps=fps,
                reference_images=all_refs, seed=seed,
                reference_video=(src_vids or None)))
            # 软锁(@Image1 话术):先量证实服从了锁才切(见 helper 注释)
            return _drop_first_frame(
                outp, cond, measured_prev=Path(prev.video_path)), cond
        # 尾帧抽不出来 → 还有自己的图可用。降级取向:有首帧角色图(含兼容
        # 模式的 keyframe)优先硬锚 i2v;纯参考角色图(从未打算当首帧)才
        # 降到 t2v_own_refs —— 角色语义在降级里也不许错配。
        strategy = ("i2v_keyframe" if kf is not None
                    else "t2v_own_refs" if (refs or p_paths) else "t2v")
        cond = {"strategy": strategy,
                "degraded_from": "ti2v_prev_plus_keyframe"}
        if strategy == "t2v_own_refs":
            d_refs = list(refs) + p_paths     # 肖像照常附挂(降级不丢身份锚)
            cond.update(reference_images=[str(p) for p in d_refs],
                        anchoring="soft_t2v_refs")
            return Path(video_gen.generate(
                prompt=brain_prompt or spec.prompt, duration=spec.duration,
                out_path=out, fps=fps, reference_images=d_refs,
                seed=seed)), cond

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
            outp = Path(video_gen.generate(
                prompt=brain_prompt or spec.prompt, duration=spec.duration,
                out_path=out,
                fps=fps, first_frame=last, seed=seed))
            # 首帧参数 = 上一镜尾帧(通道级保证)→ 首帧必重复,直接切
            return _drop_first_frame(outp, cond), cond
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


def _junction_state(mllm, prev, cache_dir: Path, tail_s: float = 2.0, portraits=None) -> str:
    """需求 ②(2026-07-15;2026-07-28 升级为看片段):出一句续接实况
    ("the cat is trotting right, camera tracking")。写 prompt 的人从
    实况起笔,不再照剧本想象。

    升级理由(用户裁决):单帧只能靠运动模糊【猜】"动没动",上一镜
    末尾约 2 秒的片段才能真判断运动状态。诚实链:VLM 支持视频
    (describe_junction_video,GeminiVLM)→ 看尾段;不支持(LocalQwen)
    / 尾段裁不出 / 视频路失败 → 回退单帧 describe_junction;仍不行 →
    ""(跳过,不编)。"""
    if prev is None or not getattr(prev, "video_path", None) or mllm is None:
        return ""

    def _cached(media: Path, call) -> str:
        try:
            key = (str(media.resolve()), media.stat().st_mtime_ns)
        except OSError:
            return ""
        if key not in _JUNCTION_CACHE:
            try:
                _JUNCTION_CACHE[key] = str(call(media) or "").strip()
            except Exception as exc:
                log.warning("junction caption failed: %s — trying the next "
                            "fallback", exc)
                _JUNCTION_CACHE[key] = ""
            if _JUNCTION_CACHE[key]:
                log.info("junction state: %s", _JUNCTION_CACHE[key][:160])
        return _JUNCTION_CACHE[key]

    # 首选:末尾片段(真实运动信息)
    vfn = getattr(mllm, "describe_junction_video", None)
    if vfn is not None:
        tail = _cut_tail(Path(prev.video_path), tail_s,
                         Path(cache_dir) / "junction_prev_tail.mp4")
        if tail is not None:
            # 具名矢量(2026-08-05 用户令):肖像随片发,who=角色名
            def _vfn_named(media, _fn=vfn, _p=portraits):
                try:
                    return _fn(media, portraits=_p)
                except TypeError:      # 旧后端无 portraits 参数 → 匿名矢量
                    return _fn(media)
            got = _cached(Path(tail), _vfn_named)
            if got:
                return got
            log.warning("junction: video-tail reading failed/empty — "
                        "falling back to the single last frame")
    # 回退:单帧(旧行为,LocalQwen 等无视频通道的后端走这里)
    fn = getattr(mllm, "describe_junction", None) \
        or getattr(mllm, "caption_image", None)
    if fn is None:
        return ""
    frame = _last_frame(Path(prev.video_path),
                        Path(cache_dir) / "junction_prev_last.png")
    if frame is None:
        return ""
    return _cached(Path(frame), fn)


def _junction_is_continuation(shot_cast_names, exit_vec,
                              portraits: Optional[dict] = None) -> bool:
    """钉/切路由判据(2026-08-05 用户令):本镜主体 ⊆ 上镜末帧可见主体
    (名字直配或肖像路径判等)→ 续拍(钉帧);否则场内切换(硬切+运镜
    转场桥)。矢量缺失/无主体 → 保守判续拍(旧行为)。"""
    if not isinstance(exit_vec, dict):
        return True
    vec_who = [str(s.get("who", "")).strip()
               for s in (exit_vec.get("subjects") or [])
               if isinstance(s, dict)]
    if not vec_who:
        return True
    this = [n for n in (shot_cast_names or [])
            if n in (portraits or {})]
    if not this:
        return True
    vec_paths = {str((portraits or {}).get(w, "")) for w in vec_who}
    vec_paths.discard("")
    for n in this:
        if n in vec_who:
            continue
        if str((portraits or {}).get(n, "")) in vec_paths:
            continue
        return False
    return True


def _map_markers(text: str, name_to_slot: dict) -> str:
    """end_state/描述的 <名字> 标记 → 记号(用户令 2026-08-05:映射在
    数据层做,写手照抄)。有槽位 → <<<image_N>>>;无槽位 → 去尖括号留名
    (名字泄漏闸兜底告警)。"""
    out = str(text or "")
    for n, tok in (name_to_slot or {}).items():
        out = out.replace(f"<{n}>", tok).replace(n, tok)
    # 去掉残余 <标记> 尖括号 —— 但绝不碰 <<<image_N>>> 记号本体
    return re.sub(r"<([^<>]{1,24})>",
                  lambda m: m.group(0)
                  if re.fullmatch(r"(?:image|video)_\d+", m.group(1))
                  else m.group(1), out)


def _map_junction(vec, name_to_slot: dict, cast,
                  portraits: Optional[dict] = None) -> Optional[dict]:
    """出场矢量 → 写手可直接照抄的记号版(用户令 2026-08-05):
    - 有记号的主体:who 替换成 <<<image_N>>>,保 position/pose/motion;
    - 其余所有人(无槽位的 cast + 幻觉路人)一律聚合进 background_figures
      一句话 —— prompt 中有方位的主体只能是记号,背景人物不许单列定位。"""
    if not isinstance(vec, dict):
        return None
    subs, bg_bits = [], []
    # 肖像路径等价(用户令 2026-08-05):矢量认出的名字与本镜清单可能
    # 同脸不同名(军官甲/乙共用一张肖像)—— 按 portraits 的图像路径判等。
    _slot_by_path = {}
    for _n, _tok in (name_to_slot or {}).items():
        _pp = (portraits or {}).get(_n)
        if _pp:
            _slot_by_path.setdefault(str(_pp), _tok)
    for s_ in (vec.get("subjects") or []):
        if not isinstance(s_, dict):
            continue
        who = str(s_.get("who", "")).strip()
        tok = (name_to_slot or {}).get(who) \
            or _slot_by_path.get(str((portraits or {}).get(who, "")))
        if tok:
            subs.append({**s_, "who": tok})
        else:
            pose = str(s_.get("pose") or who or "a figure")
            pos = str(s_.get("position") or "")
            bg_bits.append(f"{pose} ({pos})" if pos else pose)
    out = {"subjects": subs, "camera": vec.get("camera") or {},
           "unfinished_action": vec.get("unfinished_action")}
    if bg_bits:
        # 用户令(2026-08-05):归属不确定的人物描述【删除】—— 不给
        # 写手任何可抄的个体描述/方位;只留一句无描述的通用背景子句。
        out["background_figures"] = (
            f"{len(bg_bits)} unresolved background figure(s) present — "
            "write AT MOST one generic subordinated clause (e.g. "
            "'background figures remain still'); NEVER describe, name or "
            "position them individually")
    return out


def _parse_exit_vector(text: str):
    """出场矢量解析:接点 VLM 的 JSON 回复 → dict;不是 JSON(旧后端
    散文/坏回复)→ None,调用方原文照发 —— 绝不编造结构。"""
    t = str(text or "").strip()
    if not t:
        return None
    data = _extract_json(t)
    if isinstance(data, dict) and ("subjects" in data or "camera" in data):
        return data
    return None


def _conditions_for_prompt(strategy: str, entry, prev,
                           use_prev_tail: bool,
                           junction: str = "",
                           source_videos: Optional[list] = None,
                           cast: Optional[dict] = None,
                           setting: str = "",
                           portraits=None, video_gen=None,
                           prompt_language: str = "en") -> list[dict]:
    """给 prompt enhancer 的【条件事实清单】(2026-07-15 需求 2):执行器
    按策略把"生成时真的会喂进去什么"翻译成文字 —— 增强器只能利用这些
    事实,不能发明条件。

    方案 A(2026-07-16):媒体条件直接来自槽位清单(_slot_manifest,与
    payload 装配同源),逐条 {kind: image|video, slot, referenceable,
    description} —— 增强器引用编号只许照抄 slot,校验闸在出口把关。
    状态条件(kind=state)照旧。"""
    conds: list[dict] = []
    for r in _slot_manifest(strategy, entry, prev, use_prev_tail,
                            source_videos=source_videos,
                            portraits=portraits, video_gen=video_gen):
        conds.append({"kind": ("video" if "video" in r["slot"].lower()
                               else "image"),
                      "slot": r["slot"],
                      "referenceable": bool(r.get("referenceable")),
                      "description": r.get("content", "")})
    # 需求 ②:状态类条件 —— prompt 必须从真实接点起笔、以剧本 end_state 收笔
    conds.append({"kind": "state", "role": "prompt_language",
                  "description": prompt_language})
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
    # 跨镜一致性描述符(2026-07-17;2026-07-18 attempt3 修订:按锚定/
    # 无锚分流)。锚定路线上场景/身份已经在锚里 —— prompt 重述整句会诱导
    # t2v 重新建景,稀释首帧软锁;注记直接写进条件行,不赌 skill 记忆。
    anchored = strategy in _ANCHORED_STRATEGIES
    for name, desc in (cast or {}).items():
        conds.append({
            "kind": "cast", "role": name, "description": desc,
            "note": ("contract format — restate ONLY the static half as "
                     "natural prose; the words 'static:'/'dynamic:' and "
                     "the dynamic list never enter the prompt."
                     + (" Anchored route: ONE short identity clause is "
                        "enough — the anchor carries the look."
                        if anchored else
                        " Unanchored route: the full static half, woven "
                        "as natural prose, is the only identity carrier."))})
    if setting:
        conds.append({
            "kind": "setting", "role": "scene_setting",
            "description": setting,
            "note": ("background contract — the anchor already carries the "
                     "scene; do NOT restate this as a scene-establishing "
                     "sentence, write only a short preserve clause like "
                     "'preserve the established scene, lighting and camera'."
                     if anchored else
                     "unanchored route — weave these setting words into "
                     "the prompt; they are the only scene carrier.")})
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
    pin_gate_mad: float = 0.0,          # §G 钉帧完整性闸门阈值(≤0 关闭;
                                        # 荐 8.0 —— 帧 1→2 像素差超阈当场重掷)
    enable_audio: bool = False,         # 音频线(2026-07-29):对白原生音频+配乐
    character_library=None,             # 跨片角色肖像库(2026-07-31)
    screenplay: Optional[str] = None,   # M2:用户自带剧本(给了就跳过 §A0)
    enable_review: bool = True,         # M2:评审/修复总开关(关 = 首选即收)
    given_characters: Optional[dict] = None,  # 剧本 JSON 的 角色名→图片路径
                                        # (路径可 None = 图缺失,落回生成链)
    repair_mode: str = "full",          # 修复两段:full / consistency(只
                                        # 转场)/ classic(只旧策略)
    enable_transitions: bool = False,   # 转场桥开关(2026-08-04 用户令:
                                        # 默认关;开着才把 add_transition
                                        # 提供给修复菜单)
    enable_bg_frame_upgrade: bool = False,  # 背景实拍帧升级(2026-08-04
                                        # 裁决默认关:实拍帧带主人公,
                                        # 当背景参考=身份噪声扩散器)
    enable_bgm: bool = False,           # 背景音乐(2026-08-05 用户令:
                                        # 取消,默认永关;--bgm 显式开)
                                        # 才有意义;关 = 只保对白原生音)
    baseline_anchor: bool = False,      # 需求 1(2026-07-15):开工直出锚点视频
    baseline_anchor_duration=None,      # 锚点时长(None = API 默认)
    prompt_enhancer=None,               # 需求 2:可选 PromptEnhancerAgent
    mllm=None,                          # 需求 ②:接点实况 VLM(缺省用 verifier.judge)
) -> MovieResult:
    """窗口式全片生成:§A playwriting → §B keyframe → §C+§D 逐镜窗口循环
    → §E 合成 → §M episode 蒸馏。全程读写 StoryboardMemory(R1)。"""
    # ── 运行环境硬预检(2026-08-02 事故:一台没装 ffmpeg 的机器上,
    # extend 裁尾/裁头全失败 → 每镜滚雪球叠上前镜(8s→16s→25s→31s),
    # 终版又被旧 mock 兜底写成假 mp4。ffmpeg/ffprobe 是接缝裁切、接点
    # 去重、拼接、配乐的共同地基 —— 缺了就当场拒跑,绝不瘸着腿产片。──
    _missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if _missing:
        raise RuntimeError(
            f"windowed pipeline PREFLIGHT failed: {_missing} not found on "
            f"PATH. ffmpeg/ffprobe are REQUIRED (extend head-trim, junction "
            f"dedup, final concat and the audio stage all depend on them) — "
            f"install ffmpeg on this machine first (macOS: brew install "
            f"ffmpeg; Ubuntu: apt install ffmpeg).")
    try:
        import cv2  # noqa: F401  段级修复(regenerate_segment)解码用
    except Exception:
        log.warning("PREFLIGHT: cv2 (opencv-python) unavailable — "
                    "regenerate_segment repairs will degrade to whole-clip "
                    "no-ops on this machine (pip install opencv-python)")
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

    # ── §A0 剧本 + §A1 角色提取(M2 新链):idea → 剧本(用户提供则跳过)
    # → 角色正典 → 分镜。剧本/提取失败逐级诚实降级(idea 直通、空正典),
    # 分镜阶段的 cast 输出仍是兜底。──────────────────────────────────────
    screenwriter = screenwriter or ScreenwriterAgent()
    director = director or DirectorAgent()
    plan_cfg = getattr(screenwriter, "config", {}) or {}
    asset_catalog0 = _media_catalog(asset_memory)
    screenplay_text, sp_via = _write_screenplay(
        llm, user_prompt, screenplay, asset_catalog0)
    decisions.append({"stage": "screenplay", "via": sp_via,
                      "chars": len(screenplay_text)})
    # 钦定角色图像打标(剧本 JSON 输入):VLM 看图出英文 static 描述,
    # 图为法源;VLM 缺/失败 → 空描述照样入链(名字纪律不受影响),留痕。
    given_caps: dict = {}
    for _gn, _gp in (given_characters or {}).items():
        cap = ""
        if _gp and Path(_gp).exists() and mllm is not None:
            # 正典打标优先走 caption_identity(2026-08-04 run7 事故:通用
            # 一句话 caption 不带服装颜色,LLM 把黑军装脑补成白色);基类
            # 默认降级到 caption_image,无需判分支。
            _fn = getattr(mllm, "caption_identity", None) \
                or getattr(mllm, "caption_image", None)
            if _fn is not None:
                try:
                    cap = str(_fn(Path(_gp)) or "").strip()
                except Exception as exc:
                    log.warning("given character %r: image caption failed "
                                "(%s) — descriptor rides on the script only",
                                _gn, exc)
        # 成功也要留痕(事故教训:静默成功让"打标到底跑没跑"无从考证)
        log.info("given character %r canon caption (%d chars): %s",
                 _gn, len(cap), (cap[:160] + "…") if len(cap) > 160 else cap)
        given_caps[_gn] = cap
    if given_caps:
        decisions.append({"stage": "given_characters",
                          "names": sorted(given_caps),
                          "captioned": sorted(n for n, c in
                                              given_caps.items() if c)})
    cast_canon, ce_via = _extract_characters(
        llm, screenplay_text, given=(given_caps or None)) \
        if (sp_via != "idea_passthrough" or given_caps) else ({}, "skipped")
    _apply_caption_canon(cast_canon, given_caps)
    decisions.append({"stage": "character_extract", "via": ce_via,
                      "characters": sorted(cast_canon)})

    # prompt 语言随剧本(2026-08-05 用户令);动态全局语言:中文项目
    # 所有模型输出(VLM 图注/矢量/评审、LLM 理由)一律中文 —— 各后端
    # 指令懒读 output_lang(),唯一保留项是直发 flux 的 t2i 字符串。
    prompt_lang = _prompt_lang(screenplay_text or user_prompt)
    from ..language import set_output_lang
    set_output_lang(prompt_lang)
    log.info("window: prompt language = %s (follows the screenplay; "
             "ALL model outputs follow it too)", prompt_lang)
    # ── §A playwriting:剧本 → outline → specs → 台账 ────────────────────
    outline, shot_durations, shot_end_states, script_meta, outline_via = \
        _write_outline(
        llm, screenplay_text, asset_catalog0,
        episode_guidance=guidance,
        max_shots=int(plan_cfg.get("max_shots", 6)),
        # 兜底拆条必须拆【剧本】而不是原始 idea(2026-08-04 实跑事故:
        # scene_write 空响应走兜底,拆了默认 prompt,整片内容全错)
        fallback_fn=lambda: screenwriter.run(screenplay_text or user_prompt,
                                             asset_memory),
        cast_canon=cast_canon,
        prompt_language=prompt_lang,
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
        # <角括号> 标记是规划层元数据 —— spec.prompt 会进兜底模板/生成
        # prompt,必须剥掉(台账 description 保留标记供出场解析)。
        spec_.prompt = _strip_markers(spec_.prompt)
    storyboard = StoryboardMemory.from_outline(
        outline, path=cache_dir / "storyboard.json")
    # 交接棒(需求 ②-①)进台账:brain 声明的镜尾状态,下一镜续接的依据、
    # 评审的镜尾验收标准;空串 = brain 没说,不编造。
    for i_, (entry_, end_) in enumerate(zip(storyboard.entries,
                                            shot_end_states)):
        entry_.end_state = end_
        vars_ = script_meta.get("variations") or []
        opens_ = script_meta.get("opening_frames") or []
        dlgs_ = script_meta.get("dialogues") or []
        spks_ = script_meta.get("dialogue_speakers") or []
        bgs_ = script_meta.get("bgs") or []
        entry_.variation = vars_[i_] if i_ < len(vars_) else ""
        entry_.opening_frame = opens_[i_] if i_ < len(opens_) else ""
        entry_.dialogue_speaker = spks_[i_] if i_ < len(spks_) else ""
        entry_.bg_id = bgs_[i_] if i_ < len(bgs_) else ""
        entry_.dialogue = dlgs_[i_] if i_ < len(dlgs_) else ""
    # 跨镜一致性描述符进台账(影片级,一次定稿全链照抄)。setting 必须
    # 在 §A' 之前就位 —— 2026-07-31 实锤:原顺序里肖像先生成、setting 后
    # 赋值,肖像 prompt 拿到空场景,背景全错。
    storyboard.cast = dict(script_meta.get("cast", {}))
    storyboard.music_plan = dict(script_meta.get("music_plan", {}))
    storyboard.setting = str(script_meta.get("setting", ""))

    # 钦定角色肖像预填(剧本 JSON,最高优先):名字→用户图直接入台账,
    # §A' 对已有名字跳过(零 t2i);图缺失(路径救援也没救回)的名字不
    # 预填 —— 自然落回 §A' 生成链。
    for _gn, _gp in (given_characters or {}).items():
        if not _gp or not Path(_gp).exists():
            continue
        storyboard.portraits[_gn] = str(_gp)
        try:
            from ..types import Identity
            asset_memory.identity_anchors[f"portrait:{_gn}"] = Identity(
                identity_id=f"portrait:{_gn}", name=_gn, source=str(_gp),
                description=f"character: official portrait of {_gn} — "
                            f"{given_caps.get(_gn, '')}")
        except Exception:
            pass
        decisions.append({"stage": "cast_portrait", "name": _gn,
                          "via": "user_json", "path": str(_gp)})
    if given_characters:
        storyboard._save()
    # §A' 角色官方肖像:用户素材 > 跨片库 > t2i
    decisions.extend(_ensure_cast_portraits(
        storyboard, asset_memory, video_gen, cache_dir,
        library=character_library, llm=llm))
    # §A2 场景锚帧(M2):每场景一张 establishing 图(无角色),入台账
    # 持久化;可灵后端下逐镜注入 reference 行 → ref2v/i2v_first 附挂保
    # 背景。t2i 失败 → 该场景无锚,响亮记录,文字建景照旧兜底。
    _caps0 = (video_gen.capabilities() if video_gen is not None else set())
    if video_gen is not None and "t2i" in _caps0 \
            and hasattr(video_gen, "text_to_image"):
        adir = cache_dir / "anchors"
        # B 案(2026-08-04):背景资产按 brain 预测的 bg_id 分组(同 id =
        # 同一物理空间共用一张);旧剧本无 bg 字段 → 回退 scene_N 键。
        # 起底图 = t2i establishing(无角色);该 bg 首镜验收后升级为
        # 实拍抽帧(src=frame),后续镜以真实宫殿为准。
        _bg_keys: list = []
        for e in storyboard.entries:
            k = (getattr(e, "bg_id", "") or f"scene_{e.scene_idx}")
            if k not in _bg_keys:
                _bg_keys.append(k)
        # scene_image skill(2026-08-04 用户令):空景板 prompt 由专门
        # skill 撰写 + 确定性出口闸(角色名/人群词剥除、空景后缀恒定)。
        # 旧的字符串拼接("Location context: 角色动作…")是幽灵人物的
        # 来源,废除。
        _need_keys = [k for k in _bg_keys
                      if k not in (storyboard.backgrounds or {})]
        _bg_prompts, _bg_via = (_write_bg_prompts(llm, storyboard,
                                                  _need_keys)
                                if _need_keys else ({}, "none"))
        if _need_keys:
            decisions.append({"stage": "scene_image", "via": _bg_via,
                              "backgrounds": sorted(_need_keys)})
        for _bk in _need_keys:
            aprompt = _bg_prompts[_bk]
            adir.mkdir(parents=True, exist_ok=True)
            try:
                got = video_gen.text_to_image(
                    aprompt, adir / f"bg_{_bk}.png")
                storyboard.backgrounds[_bk] = {"path": str(got),
                                               "src": "t2i"}
                decisions.append({"stage": "background_asset", "bg": _bk,
                                  "via": "t2i", "path": str(got)})
            except Exception as exc:
                log.warning("background asset %s failed (%s) — that "
                            "background rides on text only", _bk, exc)
                decisions.append({"stage": "background_asset", "bg": _bk,
                                  "via": "failed"})
        storyboard._save()
    if storyboard.cast:
        log.info("window: cast canon — %s",
                 "; ".join(f"{k}: {v[:60]}" for k, v in
                           storyboard.cast.items()))
    log.info("window: playwriting done via=%s — %s",
             outline_via, storyboard.summary())

    # ── §B' Image Plan 阶段(逐 shot:brain 定【数量+角色+来源】→ 产图 →
    #     台账)。用户设定:单图 = 首帧或参考;双图 = 首尾帧或双参考;角色
    #     锁死后续的生成模型族。素材目录(asset_catalog)进决策上下文,brain
    #     看得见用户给了什么(Q-C:靠完整技能让 brain 对任意素材场景做对
    #     决策,不写死"背景图=首帧"这类规则)。──────────────────────────────
    kf_dir = cache_dir / "keyframes"
    asset_catalog = _media_catalog(asset_memory)
    portrait_paths: set = set()
    for _pp in (storyboard.portraits or {}).values():
        try:
            portrait_paths.add(str(Path(_pp).resolve()))
        except Exception:
            portrait_paths.add(str(_pp))
    for entry, spec in zip(storyboard.entries, specs):
        menu = _image_plan_menu(video_gen, asset_memory)
        d = _decide(
            llm, "image-plan", menu,
            {"shot": entry.to_brain_line(),
             "cast": storyboard.cast, "setting": storyboard.setting,
             "storyboard": storyboard.to_brain_json(),
             "asset_catalog": asset_catalog,
             "episode_guidance": guidance},
            replay_hint=replay_plan.get(entry.label),
            priority=_PLAN_PRIORITY,
        )
        decisions.append({"stage": "image_plan", "label": entry.label, **d})
        _shot_cast_b = _cast_in_shot(entry.description, storyboard.cast)
        plan_final, images, degraded_from = _execute_image_plan(
            d, entry, video_gen, asset_memory, retrieval, kf_dir,
            cast=storyboard.cast, portrait_paths=portrait_paths,
            has_portrait_cast=any(n in (storyboard.portraits or {})
                                  for n in _shot_cast_b))
        storyboard.set_image_plan(entry.shot_idx, plan_final, images,
                                  degraded_from=degraded_from)
        log.info("window: %s image-plan → %s (via=%s, %d image(s)%s)",
                 entry.label, plan_final, d["via"], len(images),
                 f", degraded from {degraded_from}" if degraded_from else "")

    # ── §C+§D 大循环:逐镜窗口生成 + 小循环评审修复 ─────────────────────────
    # G-1(2026-07-17):用户源视频备成 t2v reference_videos 可用形态
    # (≤3 条、逐条 ≤15s、带语义),整个 run 备一次,t2v 引用策略共用。
    source_videos = _prepared_source_videos(asset_memory,
                                            cache_dir / "asset_labels")
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
        junction_actual = _junction_state(mllm, prev, shot_dir,
                                          tail_s=window_tail_s,
                                          portraits=storyboard.portraits)
        # 结构化出场矢量(用户裁决):VLM 尾段报告现在是 JSON —— 解析
        # 成功则矢量随上下文进 brain / enhancer(速度续接的依据);解析
        # 失败原文照发(旧后端散文兜底,诚实降级)。
        _exit_vec = _parse_exit_vector(junction_actual)
        junction_ctx = {
            "prev_last_frame_actual": junction_actual or None,
            "prev_exit_vector": _exit_vec,
            "prev_end_state_script": (getattr(prev, "end_state", "") or None)
            if prev else None,
            "required_end_state": entry.end_state or None,
        }

        # P1-1(ViMax 借鉴):按 <标记> 确定性解析本镜出场角色 ——
        # cast 注入与评审 check 都只对出场者,不再靠 LLM 自判。
        shot_cast = _cast_in_shot(entry.description, storyboard.cast)
        shot_portraits = {n: storyboard.portraits[n] for n in shot_cast
                          if n in (storyboard.portraits or {})}
        # M2 场景锚注入(可灵后端):锚帧作为确定性 reference 行进台账,
        # 现有清单/装配/闸门管线自动附挂;按路径去重,幂等。
        if "first_frame_plus_refs" in (video_gen.capabilities() or set()):
            _bgkey = (getattr(entry, "bg_id", "")
                      or f"scene_{entry.scene_idx}")
            _bg = (storyboard.backgrounds or {}).get(_bgkey) \
                or ({"path": storyboard.scene_anchors.get(entry.scene_idx),
                     "src": "t2i"}
                    if (storyboard.scene_anchors or {}).get(entry.scene_idx)
                    else None)
            if _bg and _bg.get("path") and Path(_bg["path"]).exists() \
                    and not any(im.get("source") == "background"
                                or str(im.get("path")) == str(_bg["path"])
                                for im in (entry.images or [])):
                if _bg.get("src") == "frame":
                    _bgdesc = (f"the OFFICIAL look of background {_bgkey} "
                               f"(a real frame from an earlier shot of "
                               f"this film) — the shot MUST take place in "
                               f"this SAME space: identical architecture, "
                               f"floor, furniture and lighting; never "
                               f"invent a different hall; IGNORE the "
                               f"people in it — do not copy them or "
                               f"their positions")
                else:
                    _bgdesc = (f"the OFFICIAL look of background {_bgkey} "
                               f"— the shot MUST take place in this SAME "
                               f"space: identical architecture, floor, "
                               f"furniture and lighting; never invent a "
                               f"different hall; do not copy its empty "
                               f"framing")
                # 布局恒定律(2026-08-05 shot6 审计事故):背景板【前插】
                # —— 恒为自有图第一位,<<<image_1>>> 永远是背景;计划图
                # 挤占 1 号造成的逐镜编号漂移(审计误判、brain 换习惯)
                # 从此消失。清单与装配读同一列表,前插两侧同步生效。
                entry.images = [{
                    "path": str(_bg["path"]), "role": "reference",
                    "source": "background",
                    "description": _bgdesc}] + list(entry.images or [])

        # §C brain 选条件策略(episode → llm → 兜底 三层)。
        # 方案 A(2026-07-16):每个候选策略的【槽位清单】随菜单发给 brain
        # —— 它写 video_prompt 时引用编号只许照抄所选策略的清单,不许猜。
        menu = _condition_menu(entry, prev, video_gen,
                               portraits=shot_portraits)
        # 钉帧默认化(2026-08-04 用户令):同 scene 续拍必须钉上镜尾帧
        # (i2v_first),不给 brain 裁量;跨 scene(或无上镜)才开放全
        # 菜单。旧后端菜单没有 i2v_first → 不受影响。
        needs_bridge = False
        if prev is not None and prev.video_path and entry.shot_idx > 0:
            _prev_entry = storyboard.entries[entry.shot_idx - 1]
            _pin_only = [m for m in menu if m["name"] == "i2v_first"]
            _cut_only = [m for m in menu if m["name"] == "ref2v"]
            if _pin_only and _prev_entry.scene_idx == entry.scene_idx:
                # 钉/切路由(2026-08-05 用户令):主体相同 → 钉帧续拍;
                # 主体不同 → ref2v 硬切 + 自动运镜转场桥(否则钉住的
                # 像素和要求的人物是两拨人,模型只能原地变形换人)。
                if _junction_is_continuation(shot_cast, _exit_vec,
                                             storyboard.portraits):
                    menu = _pin_only
                    log.info("window: %s same-scene continuation → menu "
                             "pinned to i2v_first (rule)", entry.label)
                elif _cut_only:
                    menu = _cut_only
                    needs_bridge = True
                    log.info("window: %s same-scene SUBJECT CUT (tail "
                             "subjects differ) → ref2v hard cut + camera-"
                             "move junction bridge", entry.label)
                else:
                    menu = _pin_only
                    log.info("window: %s subject cut but no ref2v in menu "
                             "— pinning as fallback", entry.label)
        slots_by_strategy = {
            m["name"]: _slot_manifest(m["name"], entry, prev,
                                      use_prev_tail=True,
                                      source_videos=source_videos,
                                      portraits=shot_portraits,
                                      video_gen=video_gen)
            for m in menu}
        # 记号化接点(用户令 2026-08-05:映射在数据层做,写手照抄)——
        # 用清单里记号最多的策略做映射源(ref2v/i2v_first 编号同序);
        # 矢量主体 who→记号,无槽者聚合为背景一句;end_state 标记同映射。
        _ns_best = max((_name_slot_map(v) for v in slots_by_strategy.values()),
                       key=len, default={})
        _junction_mapped = dict(junction_ctx)
        _vm = _map_junction(_exit_vec, _ns_best, storyboard.cast,
                            portraits=storyboard.portraits)
        if _vm is not None:
            _junction_mapped["prev_last_frame_actual"] = _vm
            _junction_mapped.pop("prev_exit_vector", None)
        for _k in ("prev_end_state_script", "required_end_state"):
            if _junction_mapped.get(_k):
                _junction_mapped[_k] = _map_markers(_junction_mapped[_k],
                                                    _ns_best)
        d = _decide(
            llm, "generation-condition", menu,
            {"shot": entry.to_brain_line(),
             "prompt_language": prompt_lang,
             "prev_shot": prev.to_brain_line() if prev else None,
             "junction": _junction_mapped,
             "cast": storyboard.cast, "setting": storyboard.setting,
             "cast_in_shot": sorted(shot_cast),
             "slots_by_strategy": slots_by_strategy,
             "storyboard": storyboard.to_brain_json(),
             "episode_guidance": guidance},
            replay_hint=replay_cond.get(entry.label),
            priority=_CONDITION_PRIORITY,
        )
        decisions.append({"stage": "condition", "label": entry.label, **d})
        # 草稿留档(消融实验前提):brain 的 video_prompt 原文,在一切
        # 清洗/润色/闸门/对白追加之前,逐字入台账。
        entry.draft_prompt = str(d.get("video_prompt") or "")
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
        # brain 的上下文里 shot 描述带 <标记>、cast 带契约标签,它写
        # prompt 时可能照抄 —— 出口一律剥标记+清洗标签(enhanced 同理)。
        brain_prompt = _scrub_setting_sentence(
            _scrub_cast_labels(_strip_markers(d.get("video_prompt", "")),
                               storyboard.cast),
            storyboard.setting, d["strategy"])
        use_tail = bool(d.get("use_prev_tail_video", False))
        slots = _slot_manifest(d["strategy"], entry, prev, use_tail,
                               source_videos=source_videos,
                               portraits=shot_portraits,
                               video_gen=video_gen)
        # ── 需求 2:可选 prompt 润色(条件事实 + 官方 prompt 技巧技能)。
        # 失败返回 None → 保留原 prompt,增强层永远不破坏正流程。
        if prompt_enhancer is not None:
            enhanced = prompt_enhancer.run(
                _strip_markers(entry.description), strategy=d["strategy"],
                conditions=_conditions_for_prompt(d["strategy"], entry, prev,
                                                  use_tail,
                                                  junction=json.dumps(
                                                      _map_junction(
                                                          _exit_vec,
                                                          _name_slot_map(slots),
                                                          storyboard.cast,
                                                          portraits=storyboard.portraits)
                                                      or junction_actual,
                                                      ensure_ascii=False),
                                                  source_videos=source_videos,
                                                  cast=shot_cast,
                                                  setting=storyboard.setting,
                                                  portraits=shot_portraits,
                                                  video_gen=video_gen,
                                                  prompt_language=prompt_lang),
                base_prompt=brain_prompt or spec.prompt,
                label=entry.label)
            if enhanced and prompt_lang == "zh" \
                    and not _is_mostly_chinese(enhanced):
                # 语言拒收闸(2026-08-05 run12 shot4:enhancer 漂回英文)
                # —— zh 项目润色产物非中文 → 整个弃用,保留中文原稿。
                log.warning("window: %s enhancer output is NOT Chinese on "
                            "a zh project — enhancement DISCARDED, keeping "
                            "the draft", entry.label)
                enhanced = None
            if enhanced:
                brain_prompt = _scrub_setting_sentence(
                    _scrub_cast_labels(_strip_markers(enhanced),
                                       storyboard.cast),
                    storyboard.setting, d["strategy"])
                decisions.append({"stage": "prompt_enhance",
                                  "label": entry.label,
                                  "strategy": d["strategy"], "via": "llm"})
        # ── E 案:正典描述符逐字契约 —— 只管【无锚】路线(文本是唯一
        # 身份载体);硬锚路线(首帧/肖像携带身份,prompt 只写运动)强行
        # 追加正典 = 稀释钉帧,豁免。
        if d["strategy"] not in _ANCHORED_STRATEGIES:
            brain_prompt, canon_notes = _enforce_cast_canon(
                brain_prompt, shot_cast, storyboard.cast)
        else:
            canon_notes = []
        for cn in canon_notes:
            decisions.append({**cn, "label": entry.label})
        # ── 音频线(2026-07-29,enable_audio 门控):对白镜临时开
        # generate_audio;口型子句在引用闸门【之后】追加(审查修正:
        # 闸门丢弃 prompt 时子句不能陪葬,顺序与全修闭包一致)。
        want_audio = bool(enable_audio and entry.dialogue)
        # 哑镜保险(2026-08-05 run12 shot5:对答两句全写在描述里,
        # dialogue 字段空 → 音频参数没开,台词无声):出门 prompt 含
        # 言说句 → 照样开原生音频;压制句缺失则补。
        if enable_audio and not want_audio and brain_prompt \
                and re.search("(?:说道?|says?)\\s*[:\uff1a]?\\s*[\"\u201c]",
                              brain_prompt):
            want_audio = True
            log.warning("window: %s prompt carries spoken lines but the "
                        "dialogue field is EMPTY (multi-line exchange?) — "
                        "enabling native audio anyway; scene_write should "
                        "have split the exchange into one shot per line",
                        entry.label)
            if "无背景音乐" not in brain_prompt \
                    and "no background music" not in brain_prompt:
                brain_prompt += ("音频:只有角色对白的人声——无背景音乐、"
                                 "无音效。" if re.search(r"[一-鿿]",
                                                        brain_prompt)
                                 else " Audio: only the characters' "
                                      "voices — no background music, "
                                      "no sound effects.")

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
        if want_audio:
            brain_prompt = _with_dialogue(brain_prompt or spec.prompt,
                                          entry, storyboard.cast,
                                          name_to_slot=_name_slot_map(slots))
        # 名字终换闸(2026-08-05 用户令:"保证所有名称都用引用"):
        # 引号外的角色名,有槽位的【确定性替换】成记号;换不了的才告警
        # (无槽者本该被 enhancer 删/改视觉把手)。台词引号内永不动。
        if brain_prompt:
            _ns_final = _name_slot_map(slots)
            _parts = re.split(r'(["“][^"“”]*["”])', brain_prompt)
            for _i in range(0, len(_parts), 2):        # 偶数段 = 引号外
                for _n, _tok in _ns_final.items():
                    if _n in _parts[_i]:
                        _parts[_i] = _parts[_i].replace(_n, _tok)
            brain_prompt = "".join(_parts)
            _noq = re.sub(r'["“][^"“”]*["”]', "", brain_prompt)
            _leak = [n for n in (storyboard.cast or {}) if n in _noq]
            if _leak:
                log.warning("window: %s outgoing prompt still carries "
                            "SLOTLESS cast name(s) %s outside quotes — "
                            "enhancer should have deleted or handled "
                            "them", entry.label, _leak)
                decisions.append({"stage": "name_leak", "label": entry.label,
                                  "names": _leak})
        for s in range(max(1, n_candidates)):
            _old_ga = getattr(video_gen, "generate_audio", False)
            if want_audio:
                video_gen.generate_audio = True
            try:
                video_path, cond = _generate_with_condition(
                    d["strategy"], entry, prev, spec, video_gen,
                    shot_dir, seed=s, fps=fps, window_tail_s=window_tail_s,
                    brain_prompt=brain_prompt, use_prev_tail_video=use_tail,
                    source_videos=source_videos, portraits=shot_portraits)
            except Exception as exc:
                # 2026-08-04 run7 shot4 事故:CDN 下载断连这类瞬时故障曾
                # 一步降级到无参考 t2v(身份/衣着全错)。参考图是镜头的
                # 命 —— 先同策略重试一次,再失败才降级。
                log.warning("window: conditioned generation failed (%s): %s "
                            "— retrying the SAME strategy once before any "
                            "degrade", d["strategy"], exc)
                try:
                    video_path, cond = _generate_with_condition(
                        d["strategy"], entry, prev, spec, video_gen,
                        shot_dir, seed=s, fps=fps,
                        window_tail_s=window_tail_s,
                        brain_prompt=brain_prompt,
                        use_prev_tail_video=use_tail,
                        source_videos=source_videos,
                        portraits=shot_portraits)
                    cond["retried_after"] = f"exception: {exc}"[:200]
                except Exception as exc2:
                    log.info("window: conditioned generation failed twice "
                             "(%s): %s — falling back to plain t2v for "
                             "this seed", d["strategy"], exc2)
                    # 审查修正:对白镜的兜底 t2v 同样要带口型子句(否则
                    # 音频开着、台词丢了,模型自由配音)。
                    video_path, cond = _generate_with_condition(
                        "t2v", entry, prev, spec, video_gen, shot_dir,
                        seed=s, fps=fps, window_tail_s=window_tail_s,
                        brain_prompt=(_with_dialogue(spec.prompt, entry,
                                                     storyboard.cast)
                                      if want_audio else ""))
                    # 异常降级必须留痕:没有这两行,台账会谎称 brain 主动
                    # 选了 t2v
                    cond["degraded_from"] = d["strategy"]
                    cond["degraded_reason"] = f"exception: {exc2}"[:200]
            finally:
                video_gen.generate_audio = _old_ga
            if want_audio:
                # 注:记录的是"本镜请求了原生音频"(部分旧路线 payload
                # 不带该参数,见 video_gen_backends._is_range_family)。
                cond["generate_audio"] = True
            cond["seed"] = s
            # §G 钉帧完整性闸门(2026-08-02 用户批准,默认关:阈值 ≤0)。
            # 只看钉了开场的路线(按 cond 里的【实际】路线,降级后不误判):
            # 开场撕裂度(接点/帧0→1/帧1→2 取最大)爆表 = 模型抛开钉帧
            # 重画全景 → 当场重掷一次(测量零成本;重掷是一次生成费,
            # 换掉更贵的"送评审→修复弯路")。

            def _gate_measure(vp, strat):
                prev_v = (Path(prev.video_path)
                          if strat in _PIN_GATE_PREV and prev is not None
                          and prev.video_path else None)
                m = _pin_frame_mad(Path(vp), shot_dir, prev_video=prev_v)
                if m is None:
                    # 用户显式开了闸门却测不了 → 必须响亮(哑火≠健康)
                    log.warning("window: %s PIN GATE could not measure "
                                "(ffmpeg/numpy/decode failure) — gate "
                                "DISARMED for this candidate", entry.label)
                    decisions.append({"stage": "pin_gate",
                                      "label": entry.label,
                                      "action": "measure_failed"})
                return m

            if (pin_gate_mad > 0
                    and cond.get("strategy") in _PIN_GATE_ROUTES):
                mad_ = _gate_measure(video_path, cond.get("strategy"))
                cond["pin_gate_mad"] = (round(mad_, 2)
                                        if mad_ is not None else None)
                if mad_ is not None and mad_ > pin_gate_mad:
                    log.warning("window: %s PIN GATE tripped (frame1→2 MAD "
                                "%.2f > %.2f) — rerolling seed %d once",
                                entry.label, mad_, pin_gate_mad, s)
                    decisions.append({"stage": "pin_gate",
                                      "label": entry.label,
                                      "strategy": cond.get("strategy"),
                                      "seed": s, "mad": round(mad_, 2),
                                      "action": "reroll"})
                    rerolled = False
                    _old_ga_g = getattr(video_gen, "generate_audio", False)
                    if want_audio:
                        video_gen.generate_audio = True
                    try:
                        video_path, cond = _generate_with_condition(
                            d["strategy"], entry, prev, spec, video_gen,
                            shot_dir, seed=s + 1000, fps=fps,
                            window_tail_s=window_tail_s,
                            brain_prompt=brain_prompt,
                            use_prev_tail_video=use_tail,
                            source_videos=source_videos,
                            portraits=shot_portraits)
                        rerolled = True
                    except Exception as exc:
                        # 重掷失败 → 如实保留被拦的原片(留给评审看),
                        # 绝不空手;台账记明原因。
                        log.warning("window: pin-gate reroll failed (%s) — "
                                    "keeping the tripped clip", exc)
                        decisions.append({"stage": "pin_gate",
                                          "label": entry.label,
                                          "action": "reroll_failed",
                                          "reason": str(exc)[:200]})
                    finally:
                        video_gen.generate_audio = _old_ga_g
                    if rerolled:
                        if want_audio:
                            cond["generate_audio"] = True
                        cond["seed"] = s + 1000
                    # 复测同样按重掷后的【实际】路线门控:重掷内部降级到
                    # 无钉开场的路线(如 t2v_own_refs)就不复测 —— 否则
                    # 会给无钉片记假的 pin 失败信号(对抗核查修正)。
                    if rerolled and cond.get("strategy") in _PIN_GATE_ROUTES:
                        mad2 = _gate_measure(video_path,
                                             cond.get("strategy"))
                        cond["pin_gate_mad"] = (round(mad2, 2)
                                                if mad2 is not None else None)
                        if mad2 is not None and mad2 > pin_gate_mad:
                            log.warning("window: %s PIN GATE still tripped "
                                        "after reroll (MAD %.2f) — keeping "
                                        "the clip; the reviewer will see it",
                                        entry.label, mad2)
                            decisions.append({"stage": "pin_gate",
                                              "label": entry.label,
                                              "seed": s + 1000,
                                              "mad": round(mad2, 2),
                                              "action": "still_tripped"})
            cond["final_prompt"] = brain_prompt or cond.get(
                "final_prompt", "")
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
                "cast": (shot_cast or None),
                "dialogue": (entry.dialogue or None),
                "dialogue_speaker": (getattr(entry, "dialogue_speaker", "")
                                     or None),
                "setting": (storyboard.setting or None),
                "images": ([{"path": im.get("path"), "role": im.get("role")}
                            for im in entry.images
                            if im.get("path") and Path(im["path"]).exists()]
                           + [{"path": pp, "role": "identity_portrait",
                               "name": n}
                              for n, pp in shot_portraits.items()
                              if Path(pp).exists()]),
                "reference_video": (cond.get("reference_video")
                                    or cond.get("video")),
            }
            initial.append(clip)

        # R-1(2026-07-17 裁决):全修闭包 —— "regenerate" 严格按该镜的
        # 【原始条件方法】重生成(同策略/同条件/同底 prompt + hint;引用
        # 闸门复用,漏提的槽位自动补句)。非窗口管线无此闭包,保持旧行为。
        def _regen_original(seed: int, hint: str = "", first_frame=None,
                            _strategy=d["strategy"], _bp=brain_prompt,
                            _entry=entry, _prev=prev, _spec=spec,
                            _ut=use_tail, _slots=slots,
                            _dir=shot_dir, _cast=storyboard.cast,
                            _setting=storyboard.setting,
                            _portraits=shot_portraits):
            # P0-B(2026-07-18):hint 替换原动作,不再 " Fix: " 追加 ——
            # 合成逻辑在 _regen_prompt(可测);剧本动作锚保证 motion
            # 永远在场;hint 过同一套出口清洗(标签+建景句)。
            # 2026-07-31 裁决 1(ViMax 肖像替换):first_frame = 修好的
            # 关键帧 —— 正式顶替台账里的首帧图(坏帧就是病根,修好的
            # 才是本镜官方关键帧;replaced_from 留痕),再按原条件重跑。
            if first_frame is not None and Path(first_frame).exists():
                _swapped = False
                for _im in (_entry.images or []):
                    if _im.get("role") in ("first", "first_frame"):
                        _im["replaced_from"] = _im.get("path")
                        _im["path"] = str(first_frame)
                        _swapped = True
                        break
                if not _swapped:
                    _entry.images = list(_entry.images or []) + [{
                        "path": str(first_frame), "role": "first_frame",
                        "source": "repair_edit",
                        "description": "identity-repaired keyframe "
                                       "(portrait replacement)"}]
                if getattr(_entry, "keyframe_path", None):
                    _entry.keyframe_path = str(first_frame)
            hint_ = _scrub_cast_labels(_strip_markers(hint), _cast)
            prompt = _regen_prompt(_strategy, _bp or _spec.prompt,
                                   hint_, _slots,
                                   action=_spec.prompt,
                                   end_state=_entry.end_state)
            prompt = _scrub_setting_sentence(prompt, _setting, _strategy)
            # E 案:全修 hint 的逐字契约同样只管无锚路线(硬锚豁免)。
            if _strategy not in _ANCHORED_STRATEGIES:
                prompt, _cn = _enforce_cast_canon(
                    prompt, _cast_in_shot(_entry.description, _cast), _cast)
            # 音频线:对白镜的全修不能把对白修没 —— hint 替换正文后口型
            # 子句重新追加(_with_dialogue 引号串去重),原生音频同款开关。
            _wa = bool(enable_audio and getattr(_entry, "dialogue", ""))
            if _wa:
                prompt = _with_dialogue(
                    prompt, _entry, _cast,
                    name_to_slot=_name_slot_map(_slot_manifest(
                        _strategy, _entry, _prev, use_prev_tail=_ut,
                        source_videos=source_videos, portraits=_portraits,
                        video_gen=video_gen)))
            _old_ga2 = getattr(video_gen, "generate_audio", False)
            if _wa:
                video_gen.generate_audio = True
            try:
                v_path, r_cond = _generate_with_condition(
                    _strategy, _entry, _prev, _spec, video_gen, _dir,
                    seed=seed, fps=fps, window_tail_s=window_tail_s,
                    brain_prompt=prompt,
                    use_prev_tail_video=_ut, source_videos=source_videos,
                    portraits=_portraits)
            finally:
                video_gen.generate_audio = _old_ga2
            r_cond["regen_of_original"] = True
            r_cond["regen_prompt_mode"] = ("hint_replace" if hint_
                                           else "base")
            if first_frame is not None:
                r_cond["keyframe_replaced"] = str(first_frame)
            if _wa:
                r_cond["generate_audio"] = True
            return v_path, r_cond

        # M2 转场闭包(add_transition 修复工具,规则定死):上一镜尾帧 +
        # 本镜首帧 → flf2v 3 秒;brain 只写运动 prompt,帧提取/时长/落盘
        # 全由闭包确定性完成。无上镜或后端无 flf2v → 不提供(菜单不出现)。
        def _transition_fn(prompt_text: str, current_video,
                           _prev=prev, _dir=shot_dir,
                           _idx=entry.shot_idx, _cast=storyboard.cast):
            last = _last_frame(Path(_prev.video_path),
                               _dir / f"shot{_idx:03d}_trans_prev.png")
            first = _extract_frame0(Path(current_video),
                                    _dir / f"shot{_idx:03d}_trans_first.png")
            if last is None or first is None:
                raise RuntimeError("transition: boundary frame extraction "
                                   "failed — tool unavailable this turn")
            outp = _dir / f"shot{_idx:03d}_transition.mp4"
            return Path(video_gen.frame_to_frame(
                prompt=(_scrub_cast_labels(_strip_markers(prompt_text),
                                           _cast).strip()
                        or "a smooth cinematic transition between the two "
                           "frames; steady camera, no new subjects"),
                first_frame=last, last_frame=first, out_path=outp,
                duration=3, seed=777))

        # 转场开关(2026-08-04 用户令):默认关 —— transition_fn=None 时
        # add_transition 根本不进修复菜单(orchestrator 按可用性门控)。
        transition_fn = (_transition_fn
                         if enable_transitions
                         and prev is not None and prev.video_path
                         and "flf2v" in (video_gen.capabilities() or set())
                         and hasattr(video_gen, "frame_to_frame") else None)

        if not enable_review:
            # M2 评审总开关(关):首选候选直接收货 —— 不评审、不修复、
            # 不分胜负;结局如实记 review_disabled(绝不冒充 verified)。
            best = initial[0]
            res = SelfImproveResult(clip=best, converged=False,
                                    gen_calls=len(initial))
            res.stop_reason = "review_disabled"
            res.initial_winner = str(best.video_path)
        else:
            # §D 小循环:评审(VLM skill 维度)→ 汇总 → 定位(帧/段)→ brain
            # 修复 → Verifier 闸门 —— 全部在 generate_shot_orchestrated 内。
            res = generate_shot_orchestrated(
                spec, board=board, generator=generator, refiner=refiner,
                verifier=verifier, cache_dir=shot_dir,
                orchestrator=orchestrator,
                asset_memory=asset_memory, lesson_library=lesson_library,
                image_edit=image_edit, tournament=tournament,
                retrieval=retrieval,
                skill_library=skill_library, fps=fps,
                n_candidates=n_candidates,
                max_turns=max_turns, summarizer=summarizer,
                initial_candidates=initial,
                patience=patience, quality_bar=quality_bar,
                repair_severity=repair_severity,
                regen_fn=_regen_original,
                transition_fn=transition_fn,
                repair_mode=repair_mode,
            )
        shot_results.append(res)
        best = res.clip
        # 运镜转场桥(2026-08-05 用户令):场内切换自动生成 上镜末帧 →
        # 本镜首帧 的 flf2v 桥(3s,prompt 只写运镜);拼装时插在本镜前。
        # 失败 → 硬切照常,绝不致命(run7 教训)。
        if needs_bridge and prev is not None and prev.video_path \
                and best.video_path \
                and not getattr(best, "transition_path", None) \
                and "flf2v" in (video_gen.capabilities() or set()) \
                and hasattr(video_gen, "frame_to_frame"):
            try:
                _bl = _last_frame(
                    Path(prev.video_path),
                    shot_dir / f"shot{entry.shot_idx:03d}_jbridge_prev.png")
                _bf = _extract_frame0(
                    Path(best.video_path),
                    shot_dir / f"shot{entry.shot_idx:03d}_jbridge_first.png")
                if _bl is not None and _bf is not None:
                    _bp = ("转场运镜:镜头以一次平稳连贯的摇移或推拉,从上一"
                           "画面的主体自然过渡到新画面的主体与构图,沿途保持"
                           "同一空间、光线与人群状态,不引入新人物与新动作,"
                           "一气呵成。" if prompt_lang == "zh" else
                           "Transition camera move: one smooth continuous "
                           "pan/track from the previous framing's subjects "
                           "to the new framing, same space, lighting and "
                           "crowd; no new subjects, no new actions.")
                    _bridge = video_gen.frame_to_frame(
                        prompt=_bp, first_frame=_bl, last_frame=_bf,
                        out_path=(shot_dir /
                                  f"shot{entry.shot_idx:03d}_junction_bridge"
                                  ".mp4"),
                        duration=3, seed=777)
                    entry.transition_path = str(_bridge)
                    storyboard._save()
                    decisions.append({"stage": "junction_bridge",
                                      "label": entry.label,
                                      "path": str(_bridge)})
                    log.info("window: %s junction bridge generated → %s",
                             entry.label, Path(str(_bridge)).name)
            except Exception as exc:
                log.warning("window: %s junction bridge FAILED (%s) — "
                            "plain hard cut stays", entry.label, exc)

        # M2:修复环选择了 add_transition → 转场片路径进台账持久化
        if getattr(best, "transition_path", None):
            entry.transition_path = str(best.transition_path)
            storyboard._save()
            decisions.append({"stage": "transition", "label": entry.label,
                              "path": entry.transition_path})
        # B 案帧升级 —— 2026-08-04 用户裁决后【默认禁用】:实拍首帧里
        # 站着主人公,把它当背景参考注进后续镜 = 身份噪声扩散器(run8
        # 事故:幽灵新人经帧升级污染全片)。空景 t2i 板从头用到尾;
        # 要回头开走 enable_bg_frame_upgrade。
        _bgkey2 = (getattr(entry, "bg_id", "")
                   or f"scene_{entry.scene_idx}")
        _bg2 = (storyboard.backgrounds or {}).get(_bgkey2)
        if enable_bg_frame_upgrade and _bg2 and _bg2.get("src") == "t2i" \
                and best.video_path \
                and Path(best.video_path).exists():
            _real = _extract_frame0(
                Path(best.video_path),
                cache_dir / "backgrounds" / f"{_bgkey2}_real.png")
            if _real is not None:
                storyboard.backgrounds[_bgkey2] = {"path": str(_real),
                                                   "src": "frame"}
                storyboard._save()
                decisions.append({"stage": "background_asset",
                                  "bg": _bgkey2, "via": "frame_upgrade",
                                  "path": str(_real)})

        # S0(RL 数据管道):每镜结束记一条【结局】—— 数据集构建器按
        # label 把本镜所有决策(条件/润色/图计划)连到这条结局上打标签;
        # 修复决策另有逐条 repair/outcome(靠 decision_id 连接)。
        brain_log("window/shot_outcome", {
            "label": entry.label, "shot_idx": entry.shot_idx,
            "converged": bool(res.converged),
            "stop_reason": res.stop_reason,
            "repair_turns": len(res.actions),
            "gen_calls": res.gen_calls,
            "condition_decision_id": d.get("decision_id"),
            "decided_strategy": d["strategy"], "decided_via": d["via"],
        })

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
        # pin_gate_mad 是逐候选的噪声测量值,不算条件分歧(对抗核查修正:
        # 不剔除它,开闸后每个多候选镜都会被误判"分歧"而膨胀 per_seed)。
        distinct = {json.dumps({k: v for k, v in c.items()
                                if k not in ("seed", "pin_gate_mad")},
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
    clips, assemble_notes = _final_cut(storyboard, cache_dir)
    decisions.extend(assemble_notes)
    # §E 对账(2026-08-02 事故):终版逐镜实测时长 vs 计划时长 —— 未裁净
    # 的 extend(30.9s 冒充 6s 镜、还带着前两镜画面)在拼接前响亮暴露。
    for e_ in storyboard.entries:
        if not e_.video_path or not Path(e_.video_path).exists():
            continue
        planned_ = (getattr(specs[e_.shot_idx], "duration", None)
                    if e_.shot_idx < len(specs) else None)
        if not planned_:
            continue
        actual_ = _probe_seconds(Path(e_.video_path))
        if actual_ > 0 and abs(actual_ - float(planned_)) > max(
                1.5, 0.3 * float(planned_)):
            log.warning("assemble AUDIT: %s runs %.1fs but the plan says "
                        "%.1fs — the clip likely still carries previous-"
                        "shot footage (untrimmed extend) or was cut wrong",
                        e_.label, actual_, float(planned_))
            decisions.append({"stage": "assemble", "label": e_.label,
                              "action": "duration_mismatch",
                              "actual_s": round(actual_, 2),
                              "planned_s": float(planned_)})
    if clips:
        try:
            from ..tools.video_concat import VideoConcatTool
            from .audio_stage import add_music, any_audio, normalize_for_concat

            concat_in = list(clips)
            if enable_audio and any_audio(clips):
                # 对白镜带音轨、静音镜没有 → concat 前统一(静音镜补
                # 无声 AAC 轨),否则 -c copy 拼接在音轨参差时产出坏文件。
                try:
                    concat_in = normalize_for_concat(
                        clips, cache_dir / "concat_norm")
                except Exception as exc:
                    log.warning("window: audio normalize failed (%s) — "
                                "falling back to raw concat", exc)
                    concat_in = list(clips)
            final = VideoConcatTool().run(concat_in, cache_dir / "movie.mp4")

            # §F 配乐(2026-07-29 极简版):music_plan 逐 scene 一条曲 →
            # 音乐床 → 人声闪避混音 → -14 LUFS。失败绝不毁正片。
            if enable_audio and not enable_bgm:
                log.info("window: BGM disabled by flag — shipping with "
                         "dialogue audio only")
            if enable_audio and enable_bgm and final is not None:
                scored = add_music(final, storyboard, video_gen,
                                   cache_dir / "movie_scored.mp4")
                if scored is not None:
                    final = scored
                    log.info("window: scored film → %s", final)
        except Exception as exc:          # 拼接失败 → 不合成,单镜可用
            log.warning("window: FINAL MERGE FAILED (%s) — no movie.mp4 "
                        "was produced; per-shot clips remain", exc)
            decisions.append({"stage": "assemble", "action": "merge_failed",
                              "reason": str(exc)[:300]})

    # ── §M 收工:蒸馏 episode(good/bad 由客观收敛状态判定)────────────────
    episode_id = ""
    if episode_memory is not None and not enable_review:
        # M2:评审关着 → 无客观收敛信号,蒸馏会把全部镜头误判为 avoid,
        # 诚实做法是跳过(响亮记录)。
        log.info("window: episode distillation SKIPPED (review disabled — "
                 "no objective outcome signal)")
        episode_memory = None
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
