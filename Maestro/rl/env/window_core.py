"""window_core — 生产 window_loop.py 生成路径函数的逐字移植
(2026-08-19 用户令【训练=生产完全同构 + rl/ 自包含】)。

来源:src/maestro/pipeline/window_loop.py 行 96-3627 原文;仅五处惰性
import 改指 rl/env 内部 shim(language / cine / skills 装载)。修改生产
原件必须同步改这里 —— tests/unit/test_rl_env_parity.py 锁差异。
被排除的只有生产 driver(generate_movie_windowed:由 rl/env/loop.py
按同一流程移植,差异仅限用户明令三点:K 组采样、skill 判官择主干、
无修复/评审板)与 baseline anchor(RL 不用)。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from env.logging_utils import brain_log, get_logger
from env.ref_slots import validate_references
from env.space_bible import space_semantic_line
from env.skills import extract_json as _extract_json
from env.cine import extract_frame
from env.storyboard import StoryboardMemory  # noqa: F401(注解/driver 用)

# 注解占位:原件签名里的类型名(future-annotations 下永不求值;
# rl/ 不引 maestro.types,运行期只按鸭子类型)
AssetMemory = ShotSpec = CandidateClip = None

log = get_logger("maestro.window")


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



def _names_to_tokens(text: str, name_to_slot: dict) -> str:
    """名字终换(共享闸,2026-08-06 xiaoming run2 事故:全修路径绕过了
    主链的名字终换闸,裸名直达 API):引号外的有槽角色名确定性替换成
    记号;台词引号内永不动。"""
    if not text or not name_to_slot:
        return text
    parts = re.split(r'(["“][^"“”]*["”])', text)
    for i in range(0, len(parts), 2):
        for n, tok in name_to_slot.items():
            if n in parts[i]:
                parts[i] = parts[i].replace(n, tok)
    return "".join(parts)


def _scene_text_for_prompt(text: str) -> str:
    """台账文本 → 可入 prompt 的画面散文(共用清洗器,2026-08-08 用户
    连环质询:重掷锚句与派生双镜是机械搬运路径,没有写手把关,台账里
    合法存在的记录格式原文直达 API)。四步:剥 "Shot N:" 前缀;剥旁白
    (带引号与无引号两种形态 —— 旁白是后期制作);剥 "音效:" 标注;
    删【纯声词句】(整句只剩声词+顿号 → 无声样片里的纯噪声);收尾
    标点归一。声词嵌在动作句里("枪声中他转身")不动 —— 只删纯声词句。"""
    t = _SHOT_PREFIX_RE.sub("", str(text or "").strip())
    t = re.sub(r"(?:画外)?旁白[:：]?\s*[\"“][^\"“”]*"
               r"[\"”]。?\s*", "", t)
    t = re.sub(r"(?:画外)?旁白[:：].*?(?=音效[:：]|$)", "", t, flags=re.S)
    t = re.sub(r"音效[:：][^。]*。?", "", t)
    parts = [s for s in re.split(r"(?<=[。;;!?！?])", t) if s.strip()]
    keep = []
    for s in parts:
        core = s
        for w in _scripted_sounds(s):
            core = core.replace(w, "")
        if re.sub(r"[、,,。;;!?！?\s]", "", core):
            keep.append(s)
    return "".join(keep).strip().rstrip("。. ")


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
    act = _scene_text_for_prompt(action)
    anchor = ""
    if act:
        es = _scene_text_for_prompt(end_state)
        # 锚句 = 纯画面散文(2026-08-08 用户质询:"本镜剧本动作:/收束
        # 为:"是给人看的脚手架标签,不是画面语言,模型会当字面文本读
        # —— prompt 就该像主链一样是纯描述)。动作句 + 收束句直接拼,
        # 语言随原料(act/end_state 本就来自剧本语言)。
        from env.language import zh as _zh
        if _zh():
            anchor = f"{act}。" + (f"{es}。" if es else "")
        else:
            anchor = f"{act}." + (f" {es}." if es else "")
    prompt = " ".join(x for x in (pin, hint.strip(), anchor) if x)
    # pin 承接句回注(2026-08-08 根修:融合 run1 shot4 重掷丢机器句后
    # 反而更差 —— 派生帧 refer 随行但 prompt 无所指,锚定被稀释):清单
    # 里有 pin_frame 行且 hint 没提它 → 机器句照初掷同款前置。
    _pin_row = next((r_ for r_ in (slots or [])
                     if r_.get("source") == "pin_frame"), None)
    if _pin_row and _pin_row.get("slot") \
            and _pin_row["slot"] not in prompt:
        from env.language import zh as _zh2
        prompt = ((f"画面从{_pin_row['slot']}所示的首帧继续。" if _zh2()
                   else f"The video continues from the first frame shown "
                        f"in {_pin_row['slot']}. ") + prompt)
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
    from env.cine import _probe_fps

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


# 锚定边界(2026-08-06 rainnight:游走匹配吃出"轮回应声/属摩擦声"
# 前缀垃圾)——声名必须从标点/行首边界起,整词捕获。
_SOUND_WORD_RE = re.compile(
    r"(?:^|[,\uff0c\u3001。;\uff1b:\uff1a!\uff01?\uff1f\s\u3000])"
    r"([一-鿿]{1,7}声|[一-鿿]{0,4}回音|鸟鸣|蝉鸣|[一-鿿]{0,4}轰鸣|海浪拍|"
    r"waves?\s+(?:roar|crash)|wind\s+how|footsteps)")
# "X声"泛匹配的言说姿态/否定词黑名单(低声说≠环境声);按【后缀】判
# ——泛匹配可能带前缀字("他低声"),endswith 才拦得住。
_SOUND_BLACKLIST = ("无声", "有声", "出声", "失声", "低声", "轻声", "大声",
                    "高声", "连声", "齐声", "柔声", "厉声", "沉声", "朗声",
                    "同声")


def _sound_ok(m: str) -> bool:
    return bool(m) and m != "声" and not any(
        m.endswith(b) for b in _SOUND_BLACKLIST)


_RUN_AMBIENCE: list = []
_RUN_SOUND_LEXICON: list = []


def set_run_ambience(*texts) -> None:
    """run 级环境声(2026-08-06:场景头"浪声轰鸣"/setting"海浪拍击"是
    全场级声学事实,逐镜描述可能不再重复)——起跑时从 setting+剧本
    扫一次,_scripted_sounds 自动并入。"""
    global _RUN_AMBIENCE
    _RUN_AMBIENCE = []
    for t in texts:
        for m in _SOUND_WORD_RE.findall(str(t or "")):
            if _sound_ok(m) and m not in _RUN_AMBIENCE:
                _RUN_AMBIENCE.append(m)


def set_run_sound_lexicon(*texts) -> None:
    """剧本级声词词典(2026-08-06 cinegraph run3 事故:分镜把"冰冷金属
    摩擦声"嵌进句中,距标点 >7 字,锚定正则漏提 → 覆盖闸误报全缺、
    音效镜险些哑掉)。剧本里声词天然贴标点,起跑扫一次;之后逐镜
    提取按【词典词直查子串】兜住句中嵌入——锚定正则只负责发现新词。"""
    global _RUN_SOUND_LEXICON
    _RUN_SOUND_LEXICON = []
    for t in texts:
        for m in _SOUND_WORD_RE.findall(str(t or "")):
            if _sound_ok(m) and m not in _RUN_SOUND_LEXICON:
                _RUN_SOUND_LEXICON.append(m)


def _scripted_sounds(*texts) -> list:
    """剧本载明的环境声(2026-08-06 run5 音频死循环:剧本明写"浪声
    轰鸣",压制句+评审却把一切音效当缺陷,修复轮打不可能赢的仗)。
    识别描述/剧本片段里的声音词,返回去重列表(顺序稳定):
    run 级环境声 ∪ 词典词子串直查 ∪ 锚定正则新词;包含去重留超集。"""
    out = list(_RUN_AMBIENCE)
    joined = "\n".join(str(t or "") for t in texts)
    for w in _RUN_SOUND_LEXICON:
        if w in joined and w not in out:
            out.append(w)
    for t in texts:
        for m in _SOUND_WORD_RE.findall(str(t or "")):
            if _sound_ok(m) and m not in out:
                out.append(m)
    # 包含去重(枪声 ⊂ 突发的巨大枪声 → 留超集,压制句不列双份)
    return [w for w in out if not any(w != o and w in o for o in out)]


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
        r'|(?:低声)?说[^"“「。;；!！?？]{0,8}?[:\uff1a]?\s*|低语[:\uff1a]?\s*|回答[:\uff1a]?\s*|问道?[:\uff1a]?\s*'
        r'|喊道?[:\uff1a]?\s*)'
        r'["“「][^"“”「」]+["”」]',
        _drop_foreign, prompt)
    prompt = re.sub(r"\s{2,}", " ", prompt).strip()
    zh_mode = bool(re.search(r"[一-鿿]", prompt) or re.search(r"[一-鿿]", line))
    # 压制句与台词解耦(2026-08-05 run11b 事故:brain 把台词写进节拍后,
    # 查重提前返回把"无背景音乐"压制句一起跳过 → 可灵自由配乐)。对白镜
    # 无论台词谁写的,压制句永远确保在场。
    _snd = _scripted_sounds(getattr(entry, "description", ""),
                            getattr(entry, "end_state", ""))
    if _snd:
        _audio_zh = (f"音频:角色说这句台词的人声与剧本写明的环境声"
                     f"({'、'.join(_snd)})——无背景音乐、无其他音效。")
        _audio_en = ("Audio: the character's voice speaking the line plus "
                     f"the scripted ambient sound ({', '.join(_snd)}) — "
                     "no background music, no other effects.")
    else:
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
            # 只换主语位代词(句首/标点后)——动词附着的宾语代词不换
            # (2026-08-05 run13d 事故:"仰望着他说"的"他"是被仰望的
            # 对象,换成说话人记号后变成仰望自己)。
            _pn = [m for m in re.finditer(r"[他她](?!们)", seg)
                   if m.start() == 0 or seg[m.start() - 1] in ",\uff0c\u3001;\uff1b:\uff1a \u3000"]
            if _pn:
                # 换【离言说动词最近】的代词(前面的代词可能指别人)
                last = _pn[-1]
                seg = seg[:last.start()] + _subj + seg[last.end():]
                return f"{seg}{verb}{quote}"
            sep = "" if (not seg or seg[-1] in ",\uff0c\u3001;\uff1b:\uff1a \u3000") else "\uff0c"
            return f"{seg}{sep}{_subj}{verb}{quote}"
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
            from env.skills import skill_body as _load_body

            _SKILL_CACHE[name] = _load_body(name)
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


def _brain_pick(llm, kind: str, menu: list[dict], context: dict,
                temperature=None) -> dict:
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
        raw = (llm.complete(prompt) if temperature is None
               else llm.complete(prompt, temperature=temperature))
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
                        given: Optional[dict] = None,
                        prompt_language: str = "en") -> tuple[dict, str]:
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
          'dynamic: <attire/accessories/props that may vary>"}}'
        + (" Character NAMES (the keys) MUST be in the screenplay's own "
           "language (e.g. 男人/黑帮老大/女子) — never English "
           "descriptions; only the static/dynamic VALUES stay English."
           if prompt_language == "zh" else ""))
    raw = ""
    chars: dict = {}
    for _attempt in range(2):
        try:
            raw = llm.complete(prompt)
            data = _extract_json(raw)
        except Exception:
            data = None
        chars = {}
        if isinstance(data, dict) \
                and isinstance(data.get("characters"), dict):
            chars = {str(k): str(v) for k, v in data["characters"].items()
                     if str(v).strip()}
        # 名字语言闸(2026-08-06 rainnight run2 事故:派生角色被起成
        # "the gunman",正典键顺流污染全链 —— 治本在出生地):zh 项目
        # 派生名必须含中文;钦定名(given)原样豁免。纠正重试一次,
        # 仍违规 → 响亮告警放行(名字仍可用,只是语言不合规)。
        _bad = [k for k in chars
                if prompt_language == "zh"
                and not re.search(r"[一-鿿]", k)
                and k not in (given or {})]
        if not _bad:
            break
        log.warning("character_extract: derived cast keys NOT in the "
                    "script's language: %s — %s", _bad,
                    "corrective retry" if _attempt == 0
                    else "keeping with a loud warning")
        if _attempt == 0:
            prompt += ("\n\nYOUR PREVIOUS REPLY VIOLATED THE NAME "
                       f"LANGUAGE LAW: keys {_bad} must be renamed in the "
                       "screenplay's own language (short noun phrases, "
                       "e.g. 黑帮老大/女子). Same characters, same "
                       "static/dynamic values, ONLY the keys renamed.")
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



def _sound_coverage(screenplay_text: str, shots: list) -> list:
    """声效覆盖判据(2026-08-06 rainnight run3 事故:剧本"雨滴敲击车窗
    声"被分镜化成"雨滴敲击车身",声词丢失 → 音效镜闸认不出 → 哑片)。
    返回【剧本载明但没有落进任何 shot 描述/end_state 的声音词】列表。"""
    spl = set(_scripted_sounds(screenplay_text))
    if not spl:
        return []
    texts, have = [], set()
    for s_ in shots:
        if isinstance(s_, dict):
            texts.append(str(s_.get("description") or "") + "\n"
                         + str(s_.get("end_state") or ""))
            have.update(_scripted_sounds(texts[-1]))
    full = "\n".join(texts)
    # 覆盖判据【直查全文子串】(2026-08-06 cinegraph run3 误报:声词嵌
    # 在句中,锚定正则重提取为空,三个在场声词被冤成全缺)——词在
    # 全文里就是在;正则提取词仅补"分镜只写了子串"(h ⊂ w)的超集案
    # (2026-08-06 run5:剧本"突发的巨大枪声" vs 分镜"枪声")。
    return sorted(w for w in spl
                  if w not in full and not any(h in w for h in have))


def _dial_text(shot_dict) -> str:
    """dialogue 字段两种形态通吃:{speaker, line} 字典(scene_write 原始
    输出)或纯字符串(台账扁平化)。2026-08-06 run4 事故:判据把字典当
    字符串读,7 条完美台词全被冤枉成 mangled。"""
    d = (shot_dict or {}).get("dialogue")
    if isinstance(d, dict):
        d = d.get("line")
    return str(d or "").strip()


def _dialogue_coverage(screenplay_text: str, shots: list) -> dict:
    """台词完整性判据(2026-08-06 xiaoming run2 事故:scene_write 把
    台词截半 —— "大海真大啊,大到能吞掉我所有的失败。"只剩前半句)。
    法则:剧本每个引号台词块必须完整、逐字地落进 dialogue 字段;长块
    允许按镜序拆进多镜,但各段拼接必须一字不差。
    返回 {ok, missing:[整块], mangled:[不属于任何块的 dialogue]}。"""
    _norm = lambda t: re.sub(r"\s+", "", str(t or ""))
    # 引号内的短名号("阿浪")不是台词块:无标点且 <6 字 → 不参检
    spans = [x for x in re.findall(r'[“"]([^“”"]{2,})[”"]',
                                   screenplay_text or "")
             if len(x) >= 6 or re.search(r"[。!！?？,，;；…]", x)]
    dials = [_dial_text(s_)
             for s_ in shots if isinstance(s_, dict)]
    dials = [d for d in dials if d]
    ndials = [_norm(d) for d in dials]
    missing = []
    for sp in spans:
        nsp = _norm(sp)
        if any(nd == nsp for nd in ndials):
            continue
        remaining = nsp
        for nd in ndials:            # 镜序贪心拼接
            if remaining.startswith(nd):
                remaining = remaining[len(nd):]
        if remaining:
            missing.append(sp)
    # mangled v2(2026-08-06 run3 假阳性:剧本源引号不闭合,完美台词
    # 被冤枉):判据 = 是否为剧本原文(归一化)的子串,与 span 无关。
    _nsp_all = _norm(screenplay_text)
    mangled = [d for d, nd in zip(dials, ndials) if nd not in _nsp_all]
    return {"ok": not missing and not mangled,
            "missing": missing, "mangled": mangled}


def _patch_dialogue_coverage(screenplay_text: str, shots: list) -> list:
    """台词截断确定性补丁(纠正重试仍失败时的兜底)。剧本源的引号可能
    不配对(xiaoming 剧本实测),按引号块补齐不可靠 —— 改为【句尾补全】:
    dialogue 去尾标点后在剧本原文中定位,若其后原文仍在句中(下一个字
    不是句号级标点),则沿原文补到句末,替换 dialogue。跨句的块级缺失
    交给纠正重试(需要语义),这里只修"腰斩半句"这一客观错误。"""
    sp = str(screenplay_text or "")
    patched = []
    for s_ in shots:
        if not isinstance(s_, dict):
            continue
        d = _dial_text(s_)
        core = d.rstrip("。!！?？…,，;；\"”“")
        if not core:
            continue
        i = sp.find(core)
        if i < 0:
            continue
        j = i + len(core)
        if j < len(sp) and sp[j] in "。!！?？…":
            continue                     # 本就断在句尾 → 不是腰斩
        k = j
        while k < len(sp) and sp[k] not in "。!！?？":
            k += 1
        if k >= len(sp) or k - i > 120:
            continue                     # 找不到句尾/异常长 → 不硬补
        full = sp[i:k + 1]
        if full != d:
            if isinstance(s_.get("dialogue"), dict):
                s_["dialogue"]["line"] = full
            else:
                s_["dialogue"] = full
            patched.append({"was": d, "now": full})
            log.warning("dialogue TRUNCATION patched: %r → %r",
                        d[:30], full[:60])
    return patched


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
        # 预算拆帐(2026-08-06 cinegraph run3 事故:空回复烧掉唯一
        # attempt,声效闸失守只能响亮放行):坏回复(空/坏 JSON)与
        # 语义闸门纠错互不挤兑——坏回复独立 2 发,每道闸门各 1 发。
        _bad = 0
        _retried: set = set()
        while True:
            try:
                raw = llm.complete(prompt)
                data = _extract_json(raw)
            except Exception as exc:
                log.warning("scene_write: LLM call failed (%s)",
                            str(exc)[:160])
                data = None
            _shots_ok = (isinstance(data, dict)
                         and isinstance(data.get("shots"), list))
            if not _shots_ok:
                _bad += 1
                if _bad <= 2:
                    log.warning("scene_write: LLM reply unusable (raw %d "
                                "chars) — retry %d/2 before the "
                                "deterministic fallback", len(raw or ""),
                                _bad)
                    continue
                break
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
                                "lang" not in _retried else
                                "; falling back to verbatim excerpts")
                    data = None
                    _shots_ok = False
                    if "lang" not in _retried:
                        _retried.add("lang")
                        prompt += ("\n\nYOUR PREVIOUS REPLY VIOLATED THE "
                                   "SCRIPT LANGUAGE LAW: every description/"
                                   "end_state/opening_frame MUST be in "
                                   "CHINESE, excerpting the screenplay "
                                   "verbatim. Rewrite the SAME storyboard "
                                   "in Chinese.")
                        continue
                    break
            if _shots_ok:
                # 台词完整性闸(2026-08-06 xiaoming run2 事故:台词被
                # 截半):缺块/残句 → 纠正重试一次;仍失败 → 确定性
                # 补丁(截断 dialogue 替换成完整台词块)。
                _cov = _dialogue_coverage(user_prompt, data["shots"])
                if not _cov["ok"]:
                    log.warning("scene_write: DIALOGUE COVERAGE failed "
                                "(missing %d, mangled %d) — %s",
                                len(_cov["missing"]), len(_cov["mangled"]),
                                "corrective retry" if "dial" not in _retried
                                else "deterministic patch")
                    if "dial" not in _retried:
                        _retried.add("dial")
                        prompt += (
                            "\n\nYOUR PREVIOUS REPLY VIOLATED THE DIALOGUE "
                            "VERBATIM & COMPLETE LAW. These screenplay "
                            "speech blocks are missing or truncated in the "
                            "dialogue fields: "
                            + json.dumps({"missing": _cov["missing"],
                                          "mangled": _cov["mangled"]},
                                         ensure_ascii=False)
                            + " Every quoted speech block must land "
                            "COMPLETE and VERBATIM in a shot's dialogue "
                            "field (split a long block across consecutive "
                            "shots ONLY if the pieces concatenate exactly; "
                            "size shot durations to fit). If a speech has "
                            "NO shot at all, the beat itself is missing — "
                            "ADD the shot (the ceiling allows it). Rewrite "
                            "the storyboard with complete dialogue.")
                        data = None
                        _shots_ok = False
                        continue
                    _patch_dialogue_coverage(user_prompt, data["shots"])
                # cast 键语言闸(2026-08-06 rainnight 事故:zh 项目派生
                # 角色被起成 "the mob boss"——实体名恒用用户语言;钦定
                # 角色名(cast_canon 键)原样豁免)。
                if prompt_language == "zh" \
                        and isinstance(data.get("cast"), dict):
                    _bad_keys = [k for k in data["cast"]
                                 if not re.search(r"[一-鿿]", str(k))
                                 and k not in (cast_canon or {})]
                    if _bad_keys:
                        log.warning("scene_write: cast keys NOT in the "
                                    "script's language on a zh project: "
                                    "%s — %s", _bad_keys,
                                    "corrective retry" if "cast" not in
                                    _retried else
                                    "keeping with a loud warning")
                        if "cast" not in _retried:
                            _retried.add("cast")
                            prompt += (
                                "\n\nYOUR PREVIOUS REPLY VIOLATED THE "
                                "SCRIPT LANGUAGE LAW for entity names: "
                                f"cast keys {_bad_keys} must be named in "
                                "the SCRIPT's language (e.g. 男人/黑帮老大"
                                "/女子), never English descriptions. "
                                "Rewrite the SAME storyboard with "
                                "script-language cast keys, updating the "
                                "<name> markers to match.")
                            data = None
                            _shots_ok = False
                            continue
                # 声效覆盖闸(2026-08-06 rainnight run3 事故):剧本载明
                # 的声音词必须落进某镜的描述 —— 声音是剧本内容,丢词
                # 即丢内容(音效镜靠它开原生音频)。缺 → 纠正重试;
                # 仍缺 → 响亮告警放行(放置归属需要语义,不硬猜)。
                _snd_missing = _sound_coverage(user_prompt, data["shots"])
                if _snd_missing:
                    log.warning("scene_write: SOUND COVERAGE failed — "
                                "scripted sounds %s land in NO shot "
                                "description — %s", _snd_missing,
                                "corrective retry" if "sound" not in
                                _retried else
                                "keeping with a loud warning")
                    if "sound" not in _retried:
                        _retried.add("sound")
                        prompt += (
                            "\n\nYOUR PREVIOUS REPLY DROPPED SCRIPTED "
                            "SOUNDS. These sound words from the screenplay "
                            "appear in NO shot description: "
                            + json.dumps(_snd_missing, ensure_ascii=False)
                            + " Sound annotations ARE script content — "
                            "carry each sound word VERBATIM into the "
                            "description (or end_state) of the shot where "
                            "it occurs. Rewrite the SAME storyboard with "
                            "the sounds carried through.")
                        data = None
                        _shots_ok = False
                        continue
                break
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
            cameras = []          # cinegraph 机位标注(缺省回退在返回处)
            facings = []          # camera_facing(2026-08-10:选图专用
                                  # 证据,永不进 prompt)
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
                    # cinegraph 机位标注(可选字段;非法 → None)
                    _camv = (s_.get("camera")
                             if isinstance(s_, dict) else None)
                    try:
                        cameras.append(
                            int(str(_camv).strip().lstrip("Cc"))
                            if _camv is not None
                            and str(_camv).strip().lstrip("Cc").isdigit()
                            else None)
                    except Exception:
                        cameras.append(None)
                    facings.append(str(s_.get("camera_facing", "")
                                       or "").strip()[:160]
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
                     "cameras": cameras,
                     "camera_facings": facings,
                     "music_plan": music_plan}, "llm"
    fb = list(fallback_fn())
    # 兜底层没有 brain → None = 不传 duration 字段,API 用自己的自然默认
    # (不是 config 的 shot_duration,也不是我们编的数);end_state 同理为空。
    return fb, [None] * len(fb), [""] * len(fb), \
        {"cast": dict(cast_canon or {}), "setting": "",
         "variations": [""] * len(fb),
         "opening_frames": [""] * len(fb), "dialogues": [""] * len(fb),
         "cameras": [None] * len(fb),
         "camera_facings": [""] * len(fb),
         "dialogue_speakers": [""] * len(fb), "bgs": [""] * len(fb),
         "music_plan": {}}, "fallback"


def _skill_body_named(name: str) -> str:
    """按名载入技能全文(缓存;缺文件返回 "")。首载响亮打日志。"""
    if name not in _SKILL_CACHE:
        try:
            from env.skills import skill_body as _load_body

            _SKILL_CACHE[name] = _load_body(name)
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
            replay_hint: Optional[str], priority: list[str],
            temperature=None) -> dict:
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
    picked = _brain_pick(llm, kind, menu, context,
                         temperature=temperature)
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
    # 语言随项目(2026-08-06 run5 事故:引用闸兜底补句把英文 t2i 描述
    # 整段灌进中文 prompt):zh 项目补句用短中文,英文描述不进 prompt。
    from env.language import zh as _zh
    if _zh() and not _is_mostly_chinese(str(d or "")):
        return f"画面中包含{kind}{n}所示之物,外观与其保持一致。"
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
        # 派生缝合(2026-08-07 用户令,条件②):派生帧作 pin_frame 行,
        # 末位编号、执行器专属提及("画面从 <<<image_K>>> 继续");挂它
        # 时背景板剔除(板磁铁法:帧自带空间,板会抢方向盘)。
        _pin_paths = {str(im.get("path")) for im in
                      (getattr(entry, "images", None) or [])
                      if im.get("source") == "pin_frame"}
        own = [x for x in refs if str(x) not in _pin_paths]
        if _pin_paths:
            _bgp = {str(im.get("path")) for im in
                    (getattr(entry, "images", None) or [])
                    if im.get("source") == "background"}
            own = [x for x in own if str(x) not in _bgp]
        rows = [{"slot": _ref_tok(video_gen, i + 1), "referenceable": True,
                 "content": _c(p, "a planned reference image")}
                for i, p in enumerate(own)]
        rows += [{"slot": _ref_tok(video_gen, len(own) + j + 1),
                  "referenceable": True, "name": n,
                  "content": _portrait_slot_content(n)}
                 for j, n in enumerate(sorted(portraits or {}))]
        if _pin_paths:
            rows.append({
                "slot": _ref_tok(video_gen,
                                 len(own) + len(portraits or {}) + 1),
                "referenceable": True, "source": "pin_frame",
                "content": ("the first frame itself (executor owns its "
                            "mention — never reference this slot "
                            "yourself)")})
        return rows
    if strategy == "i2v_first":
        rows = [{"slot": "FIRST_FRAME", "referenceable": False,
                 "content": ("the previous shot's final frame (this shot "
                             "opens exactly on it)" if prev_ok else
                             _c(kf, "this shot's planned opening frame"))}]
        own = list(refs)
        # 板磁铁事故(2026-08-06 rainnight run4 shot4:钉帧镜挂空背景板
        # refer + "保留<<<image_1>>>空间"子句,可灵把画面吸成了无人的
        # 板本体,人物被清场):硬钉上镜末帧时,钉住的帧自带空间,
        # 背景板 refer 一律剔除(与"帧升级默认禁用"同一哲学)。
        if prev_ok:
            _bgp = {str(im.get("path")) for im in
                    (getattr(entry, "images", None) or [])
                    if im.get("source") == "background"}
            own = [x for x in own if str(x) not in _bgp]
        rows += [{"slot": _ref_tok(video_gen, i + 1), "referenceable": True,
                  "content": _c(p, "a planned reference image")}
                 for i, p in enumerate(own)]
        rows += [{"slot": _ref_tok(video_gen, len(own) + j + 1),
                  "referenceable": True, "name": n,
                  "content": _portrait_slot_content(n)}
                 for j, n in enumerate(sorted(portraits or {}))]
        # 首帧本体引用(2026-08-06 用户令:承接句必须用引用,不用裸词
        # ——首帧同图再挂一路 refer,末位编号,承接句指着它说;该行
        # 的提及由执行器机器句负责,写手绝不自行提及)。
        if prev_ok:
            rows.append({
                "slot": _ref_tok(video_gen,
                                 len(own) + len(portraits or {}) + 1),
                "referenceable": True, "source": "pin_frame",
                "content": ("the first frame itself (executor owns its "
                            "mention — never reference this slot "
                            "yourself)")})
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
        # 派生缝合(2026-08-07):pin_frame 行末位随行,与 _slot_manifest
        # 1:1(own 剔板剔 pin → 肖像 → pin 末位)。
        p_names = sorted(portraits or {})
        p_paths = [Path(portraits[n]) for n in p_names]
        _pin = [im for im in (getattr(entry, "images", None) or [])
                if im.get("source") == "pin_frame"]
        _own = [x for x in refs
                if str(x) not in {str(i_.get("path")) for i_ in _pin}]
        if _pin:
            _bgp = {str(im.get("path")) for im in
                    (getattr(entry, "images", None) or [])
                    if im.get("source") == "background"}
            _own = [x for x in _own if str(x) not in _bgp]
        all_refs = _own + p_paths + [Path(_pin[0]["path"])] if _pin \
            else _own + p_paths
        if all_refs:
            cond.update(reference_images=[str(p) for p in all_refs],
                        anchoring="ref2v")
            fallback = (spec.prompt + ". " + " ".join(
                f"{_ref_tok(video_gen, i + 1)} shows: "
                f"{_desc_of(entry, p_) or 'a planned reference image'} — "
                f"keep it consistent." for i, p_ in enumerate(_own))
                + "".join(
                    f" {_ref_tok(video_gen, len(_own) + j + 1)} is the "
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
            _own = list(refs)
            if hard_prev:
                # 与 _slot_manifest 的板剔除 1:1(板磁铁事故)
                _bgp = {str(im.get("path")) for im in
                        (getattr(entry, "images", None) or [])
                        if im.get("source") == "background"}
                _own = [x for x in _own if str(x) not in _bgp]
            all_refs = _own + p_paths
            # 首帧本体 refer(与 _slot_manifest 的 pin_frame 行 1:1)
            if hard_prev:
                all_refs = all_refs + [Path(first)]
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


def _script_cast_continuity(prev_entry, entry, cast,
                            portraits: Optional[dict] = None):
    """钉/切【提前】路由(2026-08-05 用户令):分镜剧本自带人物,生成前
    就能定策略 —— 比较【上一镜 end_state 的人物】与【本镜 opening_frame/
    描述 的人物】(同脸不同名按肖像路径判等):
      本镜开场人物 ⊆ 上镜收尾人物 → (True, 理由)  钉帧续拍
      出现上镜结尾没有的人物     → (False, 理由) 转场策略(不钉帧)
    任一侧提取不到人物(旧剧本无标记等)→ 保守判续拍(旧行为)。"""
    prev_txt = (getattr(prev_entry, "end_state", "") or
                getattr(prev_entry, "description", "") or "")
    cur_txt = (getattr(entry, "opening_frame", "") or
               getattr(entry, "description", "") or "")
    prev_cast = set(_cast_in_shot(prev_txt, cast) or {})
    cur_cast = set(_cast_in_shot(cur_txt, cast) or {})
    if not prev_cast or not cur_cast:
        return True, "无法从剧本提取人物 — 保守判续拍"
    # 同脸判等:名字不同但肖像同图(军官甲/乙)按脸算同一主体
    def _face(n):
        return str((portraits or {}).get(n) or f"name:{n}")
    prev_faces = {_face(n) for n in prev_cast}
    new_faces = sorted(n for n in cur_cast if _face(n) not in prev_faces)
    if new_faces:
        return False, (f"本镜开场出现上镜结尾没有的人物 {new_faces}"
                       f"(上镜结尾: {sorted(prev_cast)})")
    return True, (f"本镜开场人物 {sorted(cur_cast)} ⊆ "
                  f"上镜结尾人物 {sorted(prev_cast)}")


def _judge_junction_cast(llm, prev_entry, entry, cast,
                         portraits: Optional[dict] = None,
                         prompt_language: str = "en"):
    """人物同异判官(2026-08-07 用户令,三条件融合派):分镜文字只提到
    谁 ≠ 画面里只有谁("上镜末尾<小明>在哭"时小红可能仍在画面里)——
    交给 LLM 用剧情推理【实际在场】的人物集合;坏输出退化为确定性
    集合相等比对(同脸不同名按肖像路径判等)。
    返回 (same: bool, reason: str, cur_open_cast: list)。"""
    cast_keys = sorted((cast or {}).keys())

    def _face(n):
        return str((portraits or {}).get(n) or f"name:{n}")

    def _same_sets(a, b):
        return {_face(n) for n in a} == {_face(n) for n in b}

    prev_txt = ((getattr(prev_entry, "description", "") or "") + "\n"
                + (getattr(prev_entry, "end_state", "") or ""))
    cur_txt = ((getattr(entry, "description", "") or "") + "\n"
               + (getattr(entry, "opening_frame", "") or ""))
    if llm is not None and cast_keys:
        try:
            raw = llm.complete(
                "Two consecutive shots of one film. Reason about who is "
                "PHYSICALLY IN FRAME at the END of the previous shot and "
                "at the START of the current shot. The storyboard text "
                "may mention only the character who acts — others can "
                "still be in frame (e.g. both sit in the same car "
                "throughout). Use the story logic. Only these character "
                "names exist: " + json.dumps(cast_keys, ensure_ascii=False)
                + "\nPREVIOUS SHOT (description + end state):\n" + prev_txt
                + "\nCURRENT SHOT (description + opening):\n" + cur_txt
                + '\nSTRICT JSON only: {"prev_end_cast": [<names>], '
                  '"cur_open_cast": [<names>], "reason": "<one sentence'
                + (", in Chinese" if prompt_language == "zh" else "")
                + '>"}')
            data = _extract_json(raw) or {}
            pe = [n for n in (data.get("prev_end_cast") or [])
                  if n in cast_keys]
            co = [n for n in (data.get("cur_open_cast") or [])
                  if n in cast_keys]
            if pe and co:
                same = _same_sets(pe, co)
                reason = (f"LLM 判定 上镜末在场{sorted(pe)} vs 本镜开场"
                          f"{sorted(co)}: {str(data.get('reason'))[:120]}")
                brain_log("window/junction_cast_judge", {
                    "raw": raw, "parsed": data, "usable": True,
                    "error": None})
                return same, reason, sorted(set(co))
            brain_log("window/junction_cast_judge", {
                "raw": raw, "parsed": data, "usable": False,
                "error": "empty/unknown cast lists"})
        except Exception as exc:
            log.warning("junction cast judge failed (%s) — deterministic "
                        "fallback", str(exc)[:120])
    # 确定性兜底:标记提取 + 集合相等(新法:增员或离场都算变)
    prev_cast = set(_cast_in_shot(
        (getattr(prev_entry, "end_state", "") or
         getattr(prev_entry, "description", "") or ""), cast) or {})
    cur_cast = set(_cast_in_shot(
        (getattr(entry, "opening_frame", "") or
         getattr(entry, "description", "") or ""), cast) or {})
    if not prev_cast or not cur_cast:
        return True, "兜底:任一侧提取不到人物 — 保守判人物一致", \
            sorted(cur_cast)
    same = _same_sets(prev_cast, cur_cast)
    return same, (f"兜底集合比对 上镜末{sorted(prev_cast)} vs 本镜开场"
                  f"{sorted(cur_cast)}"), sorted(cur_cast)


def _parse_tail_report(text):
    """片尾理解报告解析:{"camera_angle", "character_actions"} → dict;
    坏 JSON → None(调用方原文照发,诚实降级)。"""
    data = _extract_json(str(text or "").strip())
    if not isinstance(data, dict):
        return None
    if not (data.get("camera_angle") or data.get("character_actions")):
        return None
    return {"camera_angle": str(data.get("camera_angle") or ""),
            "character_actions": [
                a for a in (data.get("character_actions") or [])
                if isinstance(a, dict)]}


def _map_tail_report(report, name_to_slot: dict, cast,
                     portraits: Optional[dict] = None):
    """报告的 who → 记号(同 _map_junction 的肖像路径判等);无槽者保名。"""
    if not isinstance(report, dict):
        return report
    _slot_by_path = {}
    for _n, _tok in (name_to_slot or {}).items():
        _pp = (portraits or {}).get(_n)
        if _pp:
            _slot_by_path.setdefault(str(_pp), _tok)
    acts = []
    for a in (report.get("character_actions") or []):
        who = str(a.get("who", "")).strip()
        tok = (name_to_slot or {}).get(who) \
            or _slot_by_path.get(str((portraits or {}).get(who, "")))
        acts.append({**a, "who": tok or who})
    return {"camera_angle": report.get("camera_angle", ""),
            "character_actions": acts}


def _derive_junction_frame(video_gen, mllm, llm, prev, prev_entry, entry,
                           open_cast, cast, portraits, bg_path,
                           shot_dir: Path, prompt_lang: str,
                           stitcher=None, tail_report=None,
                           space_view=None):
    """交界派生(2026-08-07 条件②;2026-08-09 扩展到同人同景):ViMax
    双镜切缝合。参考图 = [上镜真实末帧] + [本镜开场人物肖像…] (+ 换景
    时新背景板;同景【不挂板】—— 板会和运镜打架,切后空间由双镜的
    3D 推导给出)。两镜描述:缝合师 agent(默认)→ 坏输出退化模板
    装配(第一镜=end_state,第二镜=本镜剧本,记号指称)。切后帧过帧
    审查(只认矛盾);拒 → 换 seed 重派生一次;仍拒/失败 → None
    (调用方按交界类型降级:人物变→cut;同人同景→continue)。"""
    from env.cine import (_frame_after_cut,
                           _spaced_retry,
                           frame_review_ok)
    shot_dir.mkdir(parents=True, exist_ok=True)
    tail = _last_frame(Path(prev.video_path),
                       shot_dir / "junction_prev_last.png")
    if tail is None:
        return None
    _desc_src = (getattr(entry, "opening_frame", "")
                 or getattr(entry, "description", ""))
    # 人物全引用(2026-08-08 用户令):记号覆盖【描述里出现的所有
    # cast】∪ 判官的开场名单 —— 第二镜描述里不许有裸名人物。
    _names = sorted(set(open_cast)
                    | set(_cast_in_shot(_desc_src, cast) or set()))
    refs: list = [Path(tail)]
    tokmap: dict = {}
    for n in _names:
        p = (portraits or {}).get(n)
        if p and Path(p).exists():
            refs.append(Path(p))
            tokmap[n] = _ref_tok(video_gen, len(refs))
    bg_tok = None
    if bg_path and Path(bg_path).exists():
        refs.append(Path(bg_path))
        bg_tok = _ref_tok(video_gen, len(refs))
    # ②空间圣经(2026-08-10 用户令):同景派生挂【朝向视图】当布局法
    # (换景派生已有新景板,不叠挂)—— 红墙案根修:切后方向有据可依。
    sv_tok = None
    if space_view is None or bg_tok is not None:
        pass
    elif Path(str(space_view.get("path") or "")).exists():
        refs.append(Path(space_view["path"]))
        sv_tok = _ref_tok(video_gen, len(refs))
    # 两镜描述:缝合师 agent 组稿(2026-08-09 用户令,默认启用)——
    # 第一镜以实拍片尾报告为准,第二镜提炼"切后第一眼";坏输出退化
    # 模板装配(共用清洗器,2026-08-08 连环质询的六种脏全免疫)。
    first_desc = second_desc = None
    if stitcher is not None:
        _slot_table = ([{"slot": _ref_tok(video_gen, 1),
                         "kind": "tail_frame",
                         "content": "the exact frame the first shot "
                                    "starts on (the previous clip's "
                                    "real final frame)"}]
                       + [{"slot": t, "kind": "portrait", "name": n,
                           "content": f"official portrait of {n}"}
                          for n, t in tokmap.items()]
                       + ([{"slot": bg_tok, "kind": "location",
                            "content": "the second shot's location "
                                       "(empty plate)"}]
                          if bg_tok else [])
                       + ([{"slot": sv_tok, "kind": "space_view",
                            "content": ("this location seen from the "
                                        "second shot's direction: "
                                        + str(space_view.get("caption")
                                              or "")[:200])}]
                          if sv_tok else []))
        try:
            got = stitcher.run(
                prev_end_state=_scene_text_for_prompt(_strip_markers(
                    getattr(prev_entry, "end_state", "")
                    or getattr(prev_entry, "description", ""))),
                tail_report=tail_report,
                cur_opening=_scene_text_for_prompt(
                    _map_markers(_desc_src, tokmap)),
                slot_table=_slot_table, prompt_language=prompt_lang)
        except Exception as exc:
            log.warning("junction stitcher errored (%s) — template "
                        "assembly", str(exc)[:120])
            got = None
        if got:
            first_desc = got["first_shot_desc"]
            second_desc = got["second_shot_desc"]
    _stitch_via = "agent" if (first_desc and second_desc) else "template"
    if not first_desc or not second_desc:
        # 确定性模板装配(退化路径,台账留痕由调用方 decisions 记)
        first_desc = _scene_text_for_prompt(_strip_markers(
            getattr(prev_entry, "end_state", "")
            or getattr(prev_entry, "description", "")))
        second_desc = _scene_text_for_prompt(_map_markers(_desc_src,
                                                          tokmap))
        if bg_tok:
            # 场景也是引用(2026-08-08 用户令):换景时第二镜描述正文
            # 里直接指称场景板,不只靠随行图
            second_desc = ((f"在{bg_tok}所示的场景中,{second_desc}"
                            if prompt_lang == "zh" else
                            f"In the location shown in {bg_tok}, "
                            f"{second_desc}"))
    tok1 = _ref_tok(video_gen, 1)
    # ViMax 双镜骨架原文 + 槽位语义(多参考是我们的扩展:原版只挂父帧,
    # 新人长相全靠模型瞎想 —— 挂肖像把长相钉死)
    prompt = ("Two shots. The transition between the shots is a cut to. "
              "The style of the two shots should be consistent."
              f"\nThe first shot description: {first_desc}."
              f"\nThe second shot description: {second_desc}."
              f"\n{tok1} is the exact frame the first shot starts on."
              + "".join(f" {t} is the official portrait of {n} — identity "
                        f"only, never copy its pose or framing."
                        for n, t in tokmap.items())
              + (f" {bg_tok} shows the second shot's location — same "
                 f"space, ignore its empty framing." if bg_tok else "")
              + ((" " + space_semantic_line(space_view, zh=False)
                  .replace("the SAME location",
                           f"{sv_tok} is the SAME location", 1))
                 if sv_tok else ""))
    want = _strip_markers(getattr(entry, "opening_frame", "")
                          or getattr(entry, "description", ""))
    # ④布局比对(2026-08-10):审查意图带上视图图注 —— 图注与切后帧
    # 的【空间排布】矛盾(红墙变白墙)即拒;天色/光线/时辰明确豁免
    # (首跑事故:审查把"红霞 vs 橙蓝晚霞"当矛盾,三连拒全是天色)。
    if space_view is not None and space_view.get("caption"):
        want += ("\nLAYOUT LAW — judge ONLY the spatial arrangement of "
                 "fixed elements (walls, doors, furnishings, terrain, "
                 "their positions); lighting, sky, weather and "
                 "time-of-day may differ freely: "
                 + str(space_view["caption"])[:300])
    for attempt, seed in enumerate((777, 778)):
        vout = shot_dir / f"junction_two_shot_s{seed}.mp4"
        fout = shot_dir / f"junction_derived_s{seed}.png"
        try:
            if not vout.exists():
                old_a = getattr(video_gen, "generate_audio", False)
                video_gen.generate_audio = False
                try:
                    _spaced_retry(
                        lambda: video_gen.generate(
                            prompt, 5, vout, fps=24, seed=seed,
                            reference_images=refs),
                        tag=f"junction derivation shot{entry.shot_idx}")
                finally:
                    video_gen.generate_audio = old_a
            frame = _frame_after_cut(vout, fout)
        except Exception as exc:
            log.warning("window: junction derivation FAILED for shot %d "
                        "(%s) — falling back to the hard-cut route",
                        entry.shot_idx, str(exc)[:160])
            return None
        if frame_review_ok(mllm, llm, frame, want):
            # 交界档案(2026-08-10 用户令:落台账供分析)—— 缝合师
            # via/两镜描述/双镜工件,由调用方并入 junction_meta
            try:
                entry.junction_meta = {
                    **(getattr(entry, "junction_meta", None) or {}),
                    "stitcher": {"via": _stitch_via,
                                 "first": first_desc[:300],
                                 "second": second_desc[:300]},
                    "two_shot_video": str(vout), "seed": seed}
            except Exception:
                pass
            return Path(frame)
        log.warning("window: derived junction frame REJECTED (attempt "
                    "%d) — %s", attempt + 1,
                    "re-deriving with a new seed" if attempt == 0
                    else "falling back to the hard-cut route")
    return None


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


