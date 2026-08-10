"""空间圣经(2026-08-10 用户令)——给空间建"定妆照体系"。

洞见:一致性不需要"真",只需要"唯一"。布局漂移的根源是每镜现场
脑补空间;解法是把脑补挪到资产期只做一次(多视图注册表),生成期
按镜头朝向挑视图当锚,收货后用实拍清场帧顶替脑补图(实拍 > 脑补,
新实拍 > 旧实拍)。与人物肖像体系逐层同构:
  多视图注册表 ↔ 正/侧/背定妆照;按朝向挑图 ↔ 侧脸镜挑侧面照;
  清场回流 ↔ (ViMax)帧历史池;布局比对 ↔ 帧审查矛盾法。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..logging_utils import brain_log, get_logger

log = get_logger("maestro.space_bible")

# 朝向词汇表(固定四向;回流拿不准时追加 new_N):
# 视图越多单张越少被用,一致性反而稀释;四向覆盖"正反打+左右环视"。
_VIEWS = ("left", "right", "reverse")
_VIEW_PROMPTS = {
    "left": ("Rotate the camera about 90 degrees to the LEFT from the "
             "reference view."),
    "right": ("Rotate the camera about 90 degrees to the RIGHT from the "
              "reference view."),
    "reverse": ("Turn the camera around about 180 degrees — show what "
                "lies OPPOSITE the reference view."),
}


def _edit_view_prompt(scene_desc: str, turn: str) -> str:
    """派生视图指令:场景语境 + 转向 + 两句灵魂(参照内一致/参照外
    自然延展)+ 空景。"""
    return (f"Same location as the reference image: {scene_desc}. {turn} "
            "Keep the lighting, palette, materials and every element "
            "visible in the reference exactly consistent; extend the "
            "scene naturally where the reference does not show it. "
            "No people, empty scene.")


_PAN_PROMPT = (
    "Locked-off position, one continuous take, empty scene with no "
    "people or animals. {desc} The camera rotates smoothly IN PLACE a "
    "full 360 degrees around its own axis at constant speed, revealing "
    "the entire surroundings of this exact location, and returns to "
    "the starting view. Every fixed element stays consistent; nothing "
    "appears or disappears.")


def build_space_views(storyboard, image_edit, mllm, out_dir: Path,
                      bg_descs: Optional[dict] = None,
                      video_gen=None) -> list:
    """①资产期(2026-08-10 二版,用户裁决):环视视频抽帧法 —— 首帧
    硬钉主板,可灵拍"原地 360° 环视"(视频通道的 3D 一致性保证同一
    空间),1/4、1/2、3/4 处抽帧为 pan_90/pan_180/pan_270 三视图 +
    VLM 图注(以成品为准 —— 转没转够不重要,图注说了算,挑图按图注
    匹配)。视频端失败 → 退 seedream 逐视图编辑老法(图像编辑对整景
    旋转无 3D 理解,元素会漂 —— 仅作兜底);再败 → master 恒在。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: list = []
    for bg_id, rec in (storyboard.backgrounds or {}).items():
        master = str(rec.get("path") or "")
        if not master or not Path(master).exists():
            continue
        views = storyboard.spaces.setdefault(bg_id, {})
        if "master" not in views:
            views["master"] = {"path": master,
                               "caption": _caption(mllm, master),
                               "src": rec.get("src", "t2i"),
                               "shot_idx": None}
        desc = str((bg_descs or {}).get(bg_id) or storyboard.setting or "")
        if all(v in views for v in ("pan_90", "pan_180", "pan_270")) \
                or any(v in views for v in _VIEWS):
            continue
        got = _views_from_pan_video(bg_id, master, desc, video_gen,
                                    mllm, out_dir, views, decisions)
        if not got:
            _views_from_image_edit(bg_id, master, desc, image_edit,
                                   mllm, out_dir, views, decisions)
    storyboard._save()
    return decisions


def _views_from_pan_video(bg_id, master, desc, video_gen, mllm, out_dir,
                          views, decisions) -> bool:
    """环视视频 → 三帧视图。失败 → False(调用方退图像编辑)。"""
    if video_gen is None or not hasattr(video_gen, "generate"):
        return False
    vout = Path(out_dir) / f"{bg_id}_pan360.mp4"
    try:
        if not vout.exists():
            from ..cinegraph.first_frame_factory import _spaced_retry
            old_a = getattr(video_gen, "generate_audio", False)
            video_gen.generate_audio = False
            try:
                _spaced_retry(
                    lambda: video_gen.generate(
                        _PAN_PROMPT.format(desc=desc), 10, vout,
                        fps=24, seed=777, first_frame=Path(master)),
                    tag=f"space pan video {bg_id}")
            finally:
                video_gen.generate_audio = old_a
        import cv2
        cap = cv2.VideoCapture(str(vout))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n < 8:
            cap.release()
            raise RuntimeError(f"pan video unreadable ({n} frames)")
        kept_paths = [Path(master)]
        for frac, view in ((0.25, "pan_90"), (0.5, "pan_180"),
                           (0.75, "pan_270")):
            if view in views:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
            ok, fr = cap.read()
            if not ok:
                continue
            outp = Path(out_dir) / f"{bg_id}_{view}.png"
            cv2.imwrite(str(outp), fr)
            # 验收(2026-08-10 用户裁决:环视是采样器不是量角器 ——
            # 入池的每张必须验证过【独特】):与 master 及已收帧的
            # MAD < 12 = 没转动/近似重复 → 丢弃留痕,宁缺勿滥。
            mads = [_frame_mad(outp, p) for p in kept_paths]
            if mads and all(m >= 0 for m in mads) and min(mads) < 12.0:
                log.info("space bible: %s/%s dropped — no real rotation "
                         "(min MAD %.1f vs kept views)", bg_id, view,
                         min(mads))
                outp.unlink(missing_ok=True)
                decisions.append({"stage": "space_view", "bg": bg_id,
                                  "view": view, "via": "dropped_dup"})
                continue
            kept_paths.append(outp)
            views[view] = {"path": str(outp),
                           "caption": _caption(mllm, outp),
                           "src": "derived", "shot_idx": None}
            decisions.append({"stage": "space_view", "bg": bg_id,
                              "view": view, "via": "pan_video"})
        cap.release()
        return any(v in views for v in ("pan_90", "pan_180", "pan_270"))
    except Exception as exc:
        log.warning("space bible: pan-video views for %s FAILED (%s) — "
                    "falling back to image-edit views", bg_id,
                    str(exc)[:140])
        return False


def _views_from_image_edit(bg_id, master, desc, image_edit, mllm,
                           out_dir, views, decisions) -> None:
    """兜底:seedream 逐视图编辑(无 3D 理解,元素会漂 —— 仅当视频
    端不可用;缺席留痕不断链)。"""
    can_edit = image_edit is not None and \
        type(image_edit).__name__ != "MockImageEditClient"
    for view in _VIEWS:
        if view in views:
            continue
        outp = Path(out_dir) / f"{bg_id}_{view}.png"
        if not outp.exists():
            if not can_edit:
                log.warning("space bible: no image-edit client — %s "
                            "view of %s skipped", view, bg_id)
                continue
            try:
                from ..cinegraph.first_frame_factory import _spaced_retry
                _spaced_retry(
                    lambda: image_edit.edit(
                        Path(master),
                        _edit_view_prompt(desc, _VIEW_PROMPTS[view]),
                        outp),
                    tag=f"space view {bg_id}/{view}")
            except Exception as exc:
                log.warning("space bible: %s view of %s FAILED (%s) — "
                            "skipped", view, bg_id, str(exc)[:120])
                continue
        views[view] = {"path": str(outp),
                       "caption": _caption(mllm, outp),
                       "src": "derived", "shot_idx": None}
        decisions.append({"stage": "space_view", "bg": bg_id,
                          "view": view, "via": "derived"})


# 清单式图注指令(2026-08-10 用户质询:通用图注是"场景类型概括",
# 同一天台的四个朝向全写成同一句 —— 图注是挑图的唯一依据,必须是
# 【可分辨的固定物清单】。仅存台账供匹配,永不进生成 prompt。)
_INVENTORY_INSTRUCTION = (
    "List the DISTINCTIVE fixed elements visible in this exact frame "
    "and where each sits (left/center/right, near/far): walls, doors, "
    "windows, furniture, large fixtures, landmarks. Concrete inventory "
    "only — no mood words, no scene-type summary, no people. One line.")


def _caption(mllm, path) -> str:
    if mllm is None:
        return ""
    fn = getattr(mllm, "caption_image_with_instruction", None)
    if fn is not None:
        try:
            return str(fn(Path(path), _INVENTORY_INSTRUCTION)
                       or "").strip()[:400]
        except Exception:
            pass
    fn = getattr(mllm, "caption_image", None)
    if fn is None:
        return ""
    try:
        return str(fn(Path(path)) or "").strip()[:400]
    except Exception as exc:
        log.warning("space bible: caption failed (%s)", str(exc)[:100])
        return ""


def _frame_mad(a, b) -> float:
    """两图 MAD(缩至 160x90;任一读不出 → -1 诚实未知)。"""
    try:
        import cv2
        ia = cv2.imread(str(a))
        ib = cv2.imread(str(b))
        if ia is None or ib is None:
            return -1.0
        ia = cv2.resize(ia, (160, 90)).astype("float32")
        ib = cv2.resize(ib, (160, 90)).astype("float32")
        return float(abs(ia - ib).mean())
    except Exception:
        return -1.0


def pick_space_view(llm, storyboard, bg_id: str,
                    shot_opening_desc: str) -> Optional[dict]:
    """②生成期:按"切后第一眼"的镜头朝向,从该 bg 的视图池挑一张
    (按图注匹配,ViMax 选图同构)。坏输出/无 LLM → master 兜底;
    池子空 → None。返回 {view, path, caption}。"""
    views = (getattr(storyboard, "spaces", None) or {}).get(bg_id) or {}
    if not views:
        return None
    def _mk(view):
        it = views[view]
        return {"view": view, "path": it["path"],
                "caption": it.get("caption", "")}
    if len(views) == 1 or llm is None:
        return _mk(next(iter(views)))
    menu = [{"view": v, "caption": it.get("caption", "")[:200]}
            for v, it in views.items()]
    raw = ""
    try:
        raw = llm.complete(
            "A film set has several photographed VIEWS of one location "
            "(captions below). Pick the ONE view whose visible content "
            "best matches what the camera should see in this shot "
            "opening — use the shot's FACING words (shot size, camera "
            "position, what it faces) and the subjects' movement "
            "direction as the evidence.\nSHOT OPENING: "
            + str(shot_opening_desc)[:500]
            + "\nVIEWS: " + json.dumps(menu, ensure_ascii=False)
            + '\nSTRICT JSON only: {"view": "<one of the view names>"}')
        got = (json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
               .get("view"))
        if got in views:
            brain_log("window/space_view_pick", {
                "raw": raw, "parsed": {"view": got}, "usable": True,
                "error": None, "context": {"bg": bg_id}})
            return _mk(got)
    except Exception:
        pass
    brain_log("window/space_view_pick", {
        "raw": raw, "parsed": None, "usable": False,
        "error": "bad pick — master fallback", "context": {"bg": bg_id}})
    return _mk("master" if "master" in views else next(iter(views)))


def space_semantic_line(view_rec: dict, zh: bool) -> str:
    """②语义行:布局法 —— 引用视图自己的图注;构图可自由,固定
    元素的位置与外观是法律(替换掉误伤布局权威的防复刻老句)。"""
    cap = str(view_rec.get("caption") or "").strip()
    if zh:
        return (f"此为同一地点朝此方向的实景"
                + (f"({cap[:120]})" if cap else "")
                + "——画面中固定元素(墙面、门窗、陈设、地貌)的位置"
                  "与外观必须与此图一致;光线、天色随剧情,取景构图"
                  "可自由。")
    return ("the SAME location seen from this direction"
            + (f" ({cap[:160]})" if cap else "")
            + " — every fixed element (walls, doors, furnishings, "
              "terrain) must keep the position and look shown here; "
              "lighting and sky follow the story; framing is free.")


def washed_frame_upgrade(storyboard, bg_id: str, tail_frame: Path,
                         image_edit, mllm, llm, out_dir: Path,
                         shot_idx: int) -> Optional[str]:
    """③收货回流:清场 → 验收(还有人 → 放弃)→ 定朝向(自信匹配
    才顶替,拿不准追加 new_N)→ 写入(实拍>脑补,新实拍>旧实拍)。
    永不抛异常 —— 回流失败 = 保持现状,绝不比没有回流差。"""
    try:
        views = storyboard.spaces.setdefault(bg_id, {})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        can_edit = image_edit is not None and \
            type(image_edit).__name__ != "MockImageEditClient"
        if not can_edit or not Path(tail_frame).exists():
            return None
        washed = out_dir / f"{bg_id}_frame_shot{shot_idx:03d}.png"
        if not washed.exists():
            from ..cinegraph.first_frame_factory import _spaced_retry
            _spaced_retry(
                lambda: image_edit.edit(
                    Path(tail_frame),
                    "Remove every person and animal from this image; "
                    "keep every fixed element of the scene exactly "
                    "where it is; fill the revealed areas naturally.",
                    washed),
                tag=f"space wash {bg_id} shot{shot_idx}")
        # 独特性验收(与环视入池同法):与池内各视图近似重复 → 不追加
        _mads = [_frame_mad(washed, it.get("path"))
                 for it in views.values() if it.get("path")]
        if _mads and all(m >= 0 for m in _mads) and min(_mads) < 12.0:
            log.info("space bible: washed frame for %s duplicates an "
                     "existing view (min MAD %.1f) — not appended",
                     bg_id, min(_mads))
            return None
        cap = _caption(mllm, washed)
        # 验收清场(2026-08-04 幽灵人物事故的根修):图注还提到人 →
        # 放弃入池,响亮留痕
        if re.search(r"person|people|man|woman|figure|child|boy|girl|"
                     r"人物|男|女|人影|孩", cap, re.IGNORECASE):
            log.warning("space bible: washed frame for %s STILL shows a "
                        "person (caption) — upgrade skipped", bg_id)
            return None
        # 纯追加(2026-08-10 用户裁决:不顶替,只添加 —— 池子膨胀无
        # 关系,挑图按图注匹配,错误顶替才有毒)
        view = f"new_{sum(1 for v in views if v.startswith('new_'))}"
        views[view] = {"path": str(washed), "caption": cap,
                       "src": "frame", "shot_idx": shot_idx}
        storyboard._save()
        log.info("space bible: %s/%s appended from shot %d's real frame",
                 bg_id, view, shot_idx)
        return view
    except Exception as exc:
        log.warning("space bible: frame upgrade failed (%s) — registry "
                    "unchanged", str(exc)[:140])
        return None
