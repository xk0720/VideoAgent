"""Skill 库加载与检索。

三类卡(asset_driven / template / closer)统一结构, Planner 靠 applies_to 检索、
靠 slots 知道每段要填什么 —— **输出契约由卡定义, 不写死在 Planner 里**,
所以加新策略只需加一张 YAML, 不用改代码。
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import PROJECT_ROOT

log = logging.getLogger("viral_studio")
SKILLS_DIR = PROJECT_ROOT / "skills"


class SkillStore:
    def __init__(self, root: Path = SKILLS_DIR):
        self.root = root
        self.skills: Dict[str, dict] = {}
        for p in sorted(root.rglob("*.yaml")):
            card = yaml.safe_load(p.read_text(encoding="utf-8"))
            card["_path"] = str(p.relative_to(root))
            self.skills[card["skill_id"]] = card
        self.rules = self._load_rules()
        log.info("Skill 库: %d 张卡 (%s)", len(self.skills),
                 ", ".join(sorted(self.skills)))

    def _load_rules(self) -> str:
        """INDEX.md 末尾的跨 skill 铁律 —— 每次都要喂给 Planner。"""
        idx = self.root / "INDEX.md"
        if not idx.exists():
            return ""
        text = idx.read_text(encoding="utf-8")
        marker = "## 跨 skill 的实测铁律"
        return text[text.index(marker):] if marker in text else ""

    def get(self, skill_id: str) -> Optional[dict]:
        return self.skills.get(skill_id)

    def candidates(self, category: str, person_count: int,
                   placement: str) -> List[dict]:
        """按商品类目 + hook 人数 + 段落位置筛出可选 skill。"""
        out = []
        for c in self.skills.values():
            a = c.get("applies_to", {})
            if a.get("placement") != placement:
                continue
            cats = a.get("categories", [])
            if cats and category not in cats and "任何" not in " ".join(cats):
                continue
            if person_count not in (a.get("person_count") or [1, 2, 3]):
                continue
            out.append(c)
        return out

    # ── 提供给 Planner 的上下文 ────────────────────────────
    def digest(self, category: str, person_count: int,
           brief: bool = False) -> str:
        """按本次输入筛过的候选卡摘要 —— 含每张卡的 slots 契约与实测告诫。"""
        lines = []
        for placement, label in (("opening", "第一段 开场"),
                                 ("body", "第二段 主体(可多段)"),
                                 ("ending", "第三段 收尾")):
            cands = self.candidates(category, person_count, placement)
            lines.append(f"\n### {label} — 可选 skill {len(cands)} 个")
            if not cands:
                lines.append("  (无可用 skill, 该段跳过)")
                continue
            for c in cands:
                p = c.get("produces", {})
                lines.append(
                    f"\n- **{c['skill_id']}** 「{c.get('name','')}」"
                    f" 时长 {p.get('duration_s', p.get('variants','按变体'))}s"
                    f" | 音频 {p.get('audio_mode','-')}"
                    f" | 背景图 {'需要' if c.get('needs_background') else '不需要'}")
                m = c.get("measured", {})
                if m.get("notes"):
                    lines.append(f"    实测: {str(m['notes']).strip()[:150]}")
                for cav in (m.get("caveats") or [])[:3]:
                    lines.append(f"    ⚠ {cav}")
                sl = c.get("slots") or {}
                if brief:            # 阶段①只选卡, 不需要 slots 细节
                    continue
                if sl:
                    # 直接给 JSON 骨架 —— 展示成"字段清单"会被模型当成值照抄(实测)
                    lines.append('    slots 必须是 JSON 对象, 键固定为下列这些'
                                 '(不多不少, 值替换成你填的内容):')
                    lines.append("    {")
                    items = list(sl.items())
                    for idx, (k, v) in enumerate(items):
                        hint = "en" if v.get("lang") == "en" else (
                            f"zh {v['min_chars']}-{v['max_chars']}字"
                            if v.get("min_chars") else v.get("lang", ""))
                        comma = "," if idx < len(items) - 1 else ""
                        lines.append(f'      "{k}": "<{hint}>"{comma}'
                                     f'   // {v.get("desc","")}')
                    lines.append("    }")
                else:
                    lines.append("    slots 固定为空对象: {}   (prompt 完全写死, 无需填空)")
                for key, label2 in (("action_library", "可用动作库"),
                                    ("scene_by_color", "配色→外景映射"),
                                    ("scene_default", "默认场景")):
                    if key in c:
                        lines.append(f"    {label2}: "
                                     f"{yaml.safe_dump(c[key], allow_unicode=True, width=200).strip()[:400]}")
        return "\n".join(lines)
