"""用户剧本 JSON 输入(固定形式)的确定性解析。

契约(用户裁决,2026-08-03):
  {"content": "<剧本全文>",
   "role": {"角色名": "<图片路径>", ...}}        # 也兼容顶层平铺 角色→路径

原则:解析是执行器的活,路径永远不进 prompt —— brain 只拿到钦定角色名
与图像打标描述;引用正确性由既有确定性链保证(名字 → <标记> → 肖像
自动附挂 → 槽位清单编号 → 引用闸门)。

坏路径救援链(样例实测 5 条里 3 条坏,数字位数写错):
  原路径 → 同目录同名 → 数字归一匹配(000014 ≡ 00014 ≡ 14)→ 缺失。
每一步救援/缺失都进 notes 响亮留痕;缺失角色照常返回(path=None),
下游落回肖像生成链,绝不拒跑。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..logging_utils import get_logger

log = get_logger("script_input")

_IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _digit_normalize(name: str) -> str:
    """数字段按整数值归一:ComfyUI_000014_.png → comfyui_14_.png。"""
    return re.sub(r"\d+", lambda m: str(int(m.group(0))),
                  name.lower())


def _resolve_image(raw: str, base_dir: Path) -> tuple[str | None, str]:
    """路径救援链 → (解析后路径 | None, 方式)。"""
    p = Path(raw)
    if p.is_file():
        return str(p), "exact"
    cand = base_dir / p.name
    if cand.is_file():
        return str(cand), "basename"
    want = _digit_normalize(p.name)
    hits = [f for f in sorted(base_dir.iterdir())
            if f.is_file() and f.suffix.lower() in _IMG_SUFFIXES
            and _digit_normalize(f.name) == want]
    if len(hits) == 1:
        return str(hits[0]), "digit_rescue"
    return None, "missing"


def parse_script_json(path: Path) -> dict:
    """→ {"content": str, "roles": {name: path|None}, "notes": [dict]}。

    content 缺失/空 → 响亮 raise(没有剧本无从开工);角色图问题一律
    notes 留痕后继续(用户裁决:告警继续)。"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"script json must be an object: {path}")
    content = str(data.get("content", "") or "").strip()
    if not content:
        raise ValueError(f"script json has no non-empty 'content': {path}")

    raw_roles: dict = {}
    if isinstance(data.get("role"), dict):
        raw_roles.update(data["role"])
    # 兼容顶层平铺:除 content/role 外,值形如图片路径的字符串键
    for k, v in data.items():
        if k in ("content", "role") or not isinstance(v, str):
            continue
        if Path(v).suffix.lower() in _IMG_SUFFIXES:
            raw_roles.setdefault(str(k), v)

    roles: dict = {}
    notes: list[dict] = []
    base_dir = path.parent
    for name, raw in raw_roles.items():
        name = str(name).strip()
        if not name:
            continue
        resolved, how = _resolve_image(str(raw), base_dir)
        roles[name] = resolved
        if how == "digit_rescue":
            log.warning("script json: %s's image path was broken (%s) — "
                        "RESCUED by digit-normalized match → %s",
                        name, raw, resolved)
            notes.append({"stage": "script_input", "name": name,
                          "action": "path_rescued", "raw": str(raw),
                          "resolved": resolved})
        elif how == "missing":
            log.warning("script json: %s's image is MISSING (%s, no rescue "
                        "candidate) — that character falls back to the "
                        "portrait-generation chain", name, raw)
            notes.append({"stage": "script_input", "name": name,
                          "action": "image_missing", "raw": str(raw)})
        if name not in content:
            log.warning("script json: role %r never appears in the "
                        "screenplay content — markers cannot bind it "
                        "(continuing per ruling)", name)
            notes.append({"stage": "script_input", "name": name,
                          "action": "name_not_in_content"})
    return {"content": content, "roles": roles, "notes": notes}
