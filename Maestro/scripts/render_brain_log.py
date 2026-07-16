#!/usr/bin/env python
"""把 brain_calls.jsonl 渲染成人读的 markdown 报告(P3b,2026-07-16)。

每次 brain 调用一节:阶段/镜头、技能装载证据、输入上下文(裁剪展示)、
原始输出全文、解析结果、引用审计。跑完片子一条命令出报告:

    python scripts/render_brain_log.py outputs/attempt2/brain_calls.jsonl \
        [-o docs/analysis_attempt2_brain_io.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _j(x, limit: int = 1600) -> str:
    t = json.dumps(x, ensure_ascii=False, indent=2, default=str)
    return t if len(t) <= limit else t[:limit] + f"\n… <{len(t)} chars total>"


def render(log_path: Path) -> str:
    lines = [f"# Brain I/O 报告 — `{log_path}`", ""]
    records = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    lines.append(f"共 {len(records)} 次 brain 调用。\n")
    for i, r in enumerate(records):
        who = r.get("label") or (f"shot {r.get('shot_idx')}"
                                 if r.get("shot_idx") is not None else "—")
        lines.append(f"## #{i} `{r.get('stage', '?')}`  ({who})")
        lines.append(f"- usable: **{r.get('usable')}**"
                     + (f" · via: {r['via']}" if r.get("via") else "")
                     + (f" · attempt: {r['attempt']}" if r.get("attempt") else ""))
        if r.get("skill"):
            lines.append(f"- skill: `{r['skill']}` "
                         f"({r.get('skill_chars', 0)} chars, "
                         f"loaded={r.get('skill_loaded')})")
        if r.get("ref_audit"):
            lines.append(f"- ref_audit: `{json.dumps(r['ref_audit'], ensure_ascii=False)}`")
        if r.get("menu"):
            lines.append(f"- menu: {', '.join(map(str, r['menu']))}")
        ctx = r.get("context")
        if ctx is not None:
            lines.append("\n<details><summary>输入 context</summary>\n")
            lines.append("```json\n" + _j(ctx) + "\n```\n</details>")
        if r.get("raw"):
            lines.append("\n<details><summary>LLM 原始输出</summary>\n")
            lines.append("```\n" + str(r["raw"])[:3000] + "\n```\n</details>")
        if r.get("parsed") is not None:
            lines.append("\n**解析结果**\n\n```json\n" + _j(r["parsed"]) + "\n```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="brain_calls.jsonl 路径")
    ap.add_argument("-o", "--out", default=None, help="输出 md(默认打印)")
    args = ap.parse_args()
    src = Path(args.log)
    if not src.is_file():
        print(f"❌ 不存在: {src}")
        return 2
    md = render(src)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"✅ 已写入 {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
