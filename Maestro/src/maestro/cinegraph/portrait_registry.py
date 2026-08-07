"""多视图肖像注册表(ViMax character_portraits_generator 严格移植)。

{角色名: {view: {"path", "description"}}},view ∈ front/side/back。
front = 既有官方肖像(用户钦定图 / 角色库 / flux 生成——上游 §A 已
备好);side/back = 图像编辑从 front 派生(ViMax 三段 prompt 原文)。
选图 agent 按镜头里人物朝向取对应视图——解决"侧脸镜头拿正脸当锚"。
编辑客户端缺席 → 诚实降级 front-only(留痕)。
"""
from __future__ import annotations

from pathlib import Path

from ..logging_utils import get_logger

log = get_logger("maestro.cinegraph")

# ── ViMax prompt 原文移植 ──
_SIDE_PROMPT = (
    "Generate a full-body, side-view portrait of character {identifier} "
    "based on the provided front-view portrait, with a pure white "
    "background. Use a wide 16:9 landscape canvas, not a vertical "
    "portrait canvas. The character should be centered in the image, "
    "occupying the middle of the wide frame with enough horizontal empty "
    "space. Facing left. Standing with arms relaxed at sides.")
_BACK_PROMPT = (
    "Generate a full-body, back-view portrait of character {identifier} "
    "based on the provided front-view portrait, with a pure white "
    "background. Use a wide 16:9 landscape canvas, not a vertical "
    "portrait canvas. The character should be centered in the image, "
    "occupying the middle of the wide frame with enough horizontal empty "
    "space. No facial features should be visible.")


def build_portrait_registry(portraits: dict, image_edit, out_dir: Path,
                            views: tuple = ("front", "side", "back"),
                            ) -> dict:
    """portraits = {角色名: front 肖像路径}(§A 产物)→ 注册表。
    幂等:目标文件已存在即跳过(ViMax skip-if-exists 语义)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry: dict = {}
    can_edit = image_edit is not None and \
        type(image_edit).__name__ != "MockImageEditClient"
    for name, front in (portraits or {}).items():
        front = Path(front)
        if not front.exists():
            log.warning("cinegraph: portrait for %r missing (%s) — "
                        "character excluded from the registry", name, front)
            continue
        # 配文 = ViMax pipeline 723-731 行原文("A front view portrait
        # of X."),选图 agent 靠这几个字识别视图
        entry = {"front": {
            "path": str(front),
            "description": f"A front view portrait of {name}."}}
        for view, tmpl in (("side", _SIDE_PROMPT), ("back", _BACK_PROMPT)):
            if view not in views:
                continue
            outp = out_dir / f"{name}_{view}.png"
            if outp.exists():
                pass                                # 幂等跳过
            elif not can_edit:
                log.warning("cinegraph: no real image-edit client — "
                            "%s view for %r skipped (front-only degrade)",
                            view, name)
                continue
            else:
                try:
                    from .first_frame_factory import _spaced_retry
                    _spaced_retry(
                        lambda: image_edit.edit(
                            front, tmpl.format(identifier=name), outp),
                        tag=f"{view} view for {name}")
                except Exception as exc:
                    log.warning("cinegraph: %s view for %r FAILED (%s) — "
                                "skipped", view, name, str(exc)[:120])
                    continue
            if outp.exists():
                entry[view] = {
                    "path": str(outp),
                    "description": f"A {view} view portrait of {name}."}
        registry[name] = entry
    return registry


def registry_pairs(registry: dict, names: list) -> list:
    """出场角色 → [(path, description), ...](ViMax 装配语义:逐视图
    平铺,选图 agent 每角色至多取一个视图)。"""
    pairs = []
    for n in names:
        for view, item in (registry.get(n) or {}).items():
            pairs.append((item["path"], item["description"]))
    return pairs
