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


def build_space_views(storyboard, image_edit, mllm, out_dir: Path,
                      bg_descs: Optional[dict] = None) -> list:
    """①资产期:每个 bg 从主板派生 left/right/reverse 三视图 + VLM
    图注(以成品为准,不信任指令)。幂等(路径存在即跳过);编辑端
    缺席/失败 → 该视图缺席留痕(master 恒在,绝不断链)。
    返回 decisions 记录。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: list = []
    can_edit = image_edit is not None and \
        type(image_edit).__name__ != "MockImageEditClient"
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
        for view in _VIEWS:
            if view in views:
                continue
            outp = out_dir / f"{bg_id}_{view}.png"
            if not outp.exists():
                if not can_edit:
                    log.warning("space bible: no image-edit client — %s "
                                "view of %s skipped", view, bg_id)
                    continue
                try:
                    from ..cinegraph.first_frame_factory import \
                        _spaced_retry
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
    storyboard._save()
    return decisions


def _caption(mllm, path) -> str:
    if mllm is None:
        return ""
    fn = getattr(mllm, "caption_image", None)
    if fn is None:
        return ""
    try:
        return str(fn(Path(path)) or "").strip()[:400]
    except Exception as exc:
        log.warning("space bible: caption failed (%s)", str(exc)[:100])
        return ""


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
            "opening.\nSHOT OPENING: " + str(shot_opening_desc)[:400]
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
        cap = _caption(mllm, washed)
        # 验收清场(2026-08-04 幽灵人物事故的根修):图注还提到人 →
        # 放弃顶替,响亮留痕
        if re.search(r"person|people|man|woman|figure|child|boy|girl|"
                     r"人物|男|女|人影|孩", cap, re.IGNORECASE):
            log.warning("space bible: washed frame for %s STILL shows a "
                        "person (caption) — upgrade skipped", bg_id)
            return None
        view = _match_view(llm, views, cap)
        if view is None:
            view = f"new_{sum(1 for v in views if v.startswith('new_'))}"
        views[view] = {"path": str(washed), "caption": cap,
                       "src": "frame", "shot_idx": shot_idx}
        storyboard._save()
        log.info("space bible: %s/%s upgraded from shot %d's real frame",
                 bg_id, view, shot_idx)
        return view
    except Exception as exc:
        log.warning("space bible: frame upgrade failed (%s) — registry "
                    "unchanged", str(exc)[:140])
        return None


def _match_view(llm, views: dict, caption: str) -> Optional[str]:
    """定朝向:LLM 拿清场帧图注与各视图图注匹配。自信匹配 → 该视图
    (但 src=frame 的条目只被更新的 frame 顶替);拿不准 → None
    (调用方追加 —— 错误追加无害,错误顶替有害)。"""
    if llm is None or not views or not caption:
        return None
    menu = [{"view": v, "caption": it.get("caption", "")[:200],
             "src": it.get("src")} for v, it in views.items()]
    try:
        raw = llm.complete(
            "Match this newly photographed view of a film location "
            "against the registered views. Answer which registered view "
            "shows the SAME direction, or null if none clearly does.\n"
            "NEW VIEW: " + caption[:300]
            + "\nREGISTERED: " + json.dumps(menu, ensure_ascii=False)
            + '\nSTRICT JSON only: {"view": "<name>"|null, '
              '"confident": true|false}')
        d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        if d.get("confident") and d.get("view") in views:
            return str(d["view"])
    except Exception:
        pass
    return None
