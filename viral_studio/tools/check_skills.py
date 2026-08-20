#!/usr/bin/env python3
"""Skill 卡体检 —— 每次改卡后必跑。

查五件事(都是踩过的坑):
  1. YAML 能否解析(中文"冒号+空格"会被当成映射键)
  2. prompt 正文里的每个 {空位} 是否都在 slots 里声明了 —— 漏声明会导致
     Planner 不填、prompt 带着 {} 原样发出去, 而校验器还报"通过"
  3. pipelines 是否覆盖了 applies_to.person_count 声明的每种人数
  4. prompt_source 指向的字段是否存在
  5. pipeline 里的 @引用 是否指向本段前面已出现的 id
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DERIVED = {"beats_text", "music_prompt", "background_prompt"}


def main() -> int:
    bad = 0
    for f in sorted((ROOT / "skills").rglob("*.yaml")):
        rel = f.relative_to(ROOT)
        try:
            d = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:                                  # noqa: BLE001
            print(f"✗ {rel}: YAML 解析失败 — {str(e).splitlines()[0]}"
                  f"  (检查中文'冒号+空格')"); bad += 1; continue

        sid, errs = d.get("skill_id", "?"), []
        pls = d.get("pipelines") or {}
        slots = set((d.get("slots") or {}).keys())

        # ② prompt 空位 vs slots 声明
        used = set()
        for key in ("prompt_template", "prompt_2p", "prompt_3p"):
            used |= set(re.findall(r"\{(\w+)\}", str(d.get(key, ""))))
        for steps in pls.values():                              # pipeline 里的 {} 也算
            used |= set(re.findall(r"\{(\w+)\}", yaml.safe_dump(steps, allow_unicode=True)))
        for miss in sorted(used - DERIVED - slots):
            errs.append(f"prompt/pipeline 用到 {{{miss}}} 但 slots 未声明")

        # ③ 人数覆盖
        for n in (d.get("applies_to") or {}).get("person_count") or []:
            if str(n) not in pls:
                errs.append(f"applies_to 含 {n} 人但 pipelines 无 \"{n}\"")

        # ④ prompt_source 指向存在
        for n, key in (d.get("prompt_source") or {}).items():
            if key and key not in d:
                errs.append(f'prompt_source["{n}"] 指向不存在的字段 {key}')

        # ⑤ @引用可达
        for n, steps in pls.items():
            seen = []
            for st in steps:
                for ref in re.findall(r"@(\w+)", yaml.safe_dump(st.get("params") or {})):
                    if ref not in seen:
                        errs.append(f'{n}人 pipeline: {st["id"]} 引用 @{ref} 不在前序 {seen}')
                seen.append(st["id"])

        if errs:
            bad += 1
            print(f"✗ {sid}  ({rel})")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"✓ {sid:20s} pipelines={sorted(pls)} slots={len(slots)}")
    print("\n全部通过 ✅" if not bad else f"\n{bad} 张卡有问题 ❌")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
