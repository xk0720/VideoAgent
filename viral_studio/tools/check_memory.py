#!/usr/bin/env python3
"""记忆库体检: YAML 可解析 + 卡片必填字段 + 媒体文件存在。

写卡片时最常踩的坑是中文"冒号+空格"被 YAML 当成映射键(已踩两次) —— 加卡后跑一次这个。
用法: python tools/check_memory.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MEM = ROOT / "memory"

REQUIRED = {
    "videos": ["video_id", "title", "specs", "structure"],
    "cards": ["asset_id", "kind", "source", "content", "compat", "usage"],
    "patterns": ["pattern_id", "name", "when_to_use", "execution"],
}


def main() -> int:
    errs, n = [], 0
    for path in sorted(MEM.rglob("*.yaml")):
        n += 1
        try:
            card = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:                       # noqa: BLE001
            errs.append(f"{path.relative_to(ROOT)}: YAML 解析失败 — "
                        f"{str(e).splitlines()[0]}(检查中文'冒号+空格')")
            continue
        group = ("cards" if path.parent.name == "cards"
                 else "videos" if path.parent.name == "videos" else "patterns")
        for field in REQUIRED[group]:
            if field not in card:
                errs.append(f"{path.relative_to(ROOT)}: 缺字段 '{field}'")
        for key in ("clip", "bgm"):                  # 资产卡的媒体必须存在
            rel = (card.get("source") or {}).get(key)
            if rel and not (MEM / rel).exists():
                errs.append(f"{path.relative_to(ROOT)}: 媒体不存在 {rel}")

    print(f"检查 {n} 张卡片: " + ("全部通过 ✅" if not errs else f"{len(errs)} 项问题 ❌"))
    for e in errs:
        print("  -", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
