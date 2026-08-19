#!/usr/bin/env python3
"""把 Planner 输出的分镜脚本解码成"实际会送进模型的东西"。

Planner 只输出 skill_id + slots(极简)；真正的 prompt 正文、流水线、素材路径
都在 skill 卡上。这个工具把两者合起来渲染, 用于人工审阅——**执行前看到的,
就是执行时真正发送的**。

用法: python tools/decode_storyboard.py outputs/sb_xxx/storyboard.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.skill_store import SkillStore     # noqa: E402


def render(tpl: str, slots: dict, extra: dict) -> str:
    out = tpl
    for k, v in {**slots, **extra}.items():
        out = out.replace("{" + k + "}", str(v))
    return out.strip()


def main() -> int:
    sb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    store = SkillStore()
    print(f"\n{'='*78}\n商品: {sb['product_name']} | 类目: {sb['category']} | "
          f"人物图: {sb['person_count']} 张\n整体思路: {sb['overall_reason']}\n{'='*78}")

    total = 0.0
    for seg in sb["segments"]:
        card = store.get(seg["skill_id"]) or {}
        p = card.get("produces", {})
        var = seg.get("variant")
        dur = p.get("duration_s") or (p.get("variants", {}).get(var, {}) or {}).get("duration_s", 0)
        total += float(dur or 0)
        print(f"\n{'─'*78}\n▶ {seg['seg_id']}  [{seg['part']}]  {dur}s  "
              f"skill={seg['skill_id']}"
              f"{'  variant=' + var if var else ''}"
              f"{'  hook#' + str(seg['hook_index']) if seg.get('hook_index') else ''}")
        print(f"  选卡理由: {seg['reason']}")

        # 流水线(Act Agent 将据此开工单)
        pipe = card.get("pipeline") or []
        print(f"  流水线: " + " → ".join(
            f"{s['id']}({s['tool']}{'|本地' if s.get('local') else ''})" for s in pipe))
        if card.get("needs_background"):
            print(f"  需要背景图: 是 → 先 image_generation 再视频生成")

        # 渲染出真正的 prompt
        tpl = card.get("prompt_template", "")
        if not tpl and var:
            tpl = card.get(f"prompt_{var}p", "")
        if tpl.strip():
            extra = {}
            if var and "variants" in p:
                beats = p["variants"][var]["beats"]
                extra["beats_text"] = ", ".join(f"{b}s" for b in beats[:-1]) + f" and {beats[-1]}s"
            body = render(tpl, seg.get("slots", {}), extra)
            print(f"\n  ── 送进视频模型的 prompt ──")
            for line in body.splitlines():
                print(f"  | {line}")
        else:
            print(f"\n  ── prompt: (无, 该 skill 由参考视频驱动) ──")
            a = card.get("asset", {})
            if a:
                print(f"  | 驱动片段: {a.get('clip')}")
                print(f"  | 自带 BGM: {a.get('bgm')}  ({a.get('bpm')} BPM)")

        # 中文文案单列(要念/要烧字的)
        for k in ("narration", "line1", "line2", "line3", "title"):
            if k in seg.get("slots", {}):
                print(f"  ▸ {k}: {seg['slots'][k]}  ({len(seg['slots'][k])}字)")
        if card.get("music_prompt"):
            print(f"  ▸ 配乐 prompt: {render(card['music_prompt'], {}, extra).strip()[:120]}…")
    print(f"\n{'='*78}\n合计时长 ≈ {total:.0f}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
