#!/usr/bin/env python
"""RL 训练集构建器(2026-08-14 用户令,仿 ViMax bench 构建法):

100 条 = 30 带分镜(bench 同格式,TypeA/B/C 各 10)
       + 30 纯剧本(无分镜无运镜词,考我方分镜)
       + 40 idea(一句话)

流程(逐阶段幂等,断点续跑以文件名为准):
  ①主题清单(1 次调用:60 个主题按覆盖轴配平 + 40 个 idea)→ themes.json
  ②30 bench 式逐条生成(校验 schema:8-16 镜/中文/类型)→ bench/*.json
    并同步产出剧本契约文件(经预分镜适配)→ screenplays/*.screenplay.json
  ③30 纯剧本逐条生成 → story/*.json + 契约文件
  ④汇出 rl/configs/task_pool_train100.yaml(60 剧本 + 40 idea,权重 3:2)

用法:
  python rl/data_gen/build_trainset.py                  # 全量(可断点续)
  python rl/data_gen/build_trainset.py --config configs/bailian.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "rl"))

from env.config import load_dotenv, load_yaml          # noqa: E402

load_dotenv(REPO / ".env")

from reward.judges import OpenAICompatChat             # noqa: E402

# 2026-08-19 用户令(rl/ 自包含):adapt_story 从
# scripts/run_vimax_benchmark.py 逐字拷贝,不再跨目录 import。
def adapt_story(d: dict) -> str:
    """预分镜 benchmark JSON → 我们的剧本 content 文本。

    形状:总览一句 + 逐场逐镜"场景N 镜头M:【开场画面】…【本镜动作】…"
    —— 镜头结构显式可见,scene_write 按预分镜法照抄切分,只做标注。"""
    lines = [f"故事总览:{d.get('story_overview', '').strip()}", ""]
    for sc in d.get("scenes", []):
        n = sc.get("scene_num")
        for sh in sc.get("shots", []):
            lines.append(
                f"场景{n} 镜头{sh.get('shot_id')}:"
                f"【开场画面】{sh.get('first_frame', '').strip()}"
                f"【本镜动作】{sh.get('video_prompt', '').strip()}")
            lines.append("")
    return "\n".join(lines).strip()

OUT = REPO / "rl/trainset"
PROMPT = (Path(__file__).parent / "benchgen_prompt.md").read_text()

_THEME_PROMPT = """为视频生成训练集设计题目清单,只输出 JSON。要求:
1. "bench": 30 个主题,TypeA/TypeB/TypeC 各 10(A=单人跨多场景,
   B=复杂室内单场景,C=2-3人互动);
2. "story": 30 个主题(不限型,标注 type 供故事设计参考);
3. "ideas": 40 条一句话点子(每条 15-30 字,像用户随手输入);
4. 全部中文;主题间不重复不近似;覆盖轴配平:室内/室外、白天/夜晚、
   1/2/3 人、对白多/少、动作强/弱、常见空间先验强/弱(面包店强,
   潜艇舱弱)各占约半;
5. 全部符合:成人角色、着装保守、无暴力无 IP、家庭友好。
输出:{"bench": [{"type": "Type A", "theme_key": "snake_case",
  "theme": "一句话"}], "story": [...同构...], "ideas": ["...", ...]}"""


def _extract_json(text):
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _call(llm, prompt, retries=2):
    for i in range(retries + 1):
        try:
            data = _extract_json(llm.chat(prompt))
            if data:
                return data
        except Exception as exc:
            print(f"  retry {i + 1}: {exc}", flush=True)
        time.sleep(8 * (i + 1))
    raise RuntimeError("LLM call failed after retries")


def _valid_bench(d: dict) -> str:
    shots = [s for sc in d.get("scenes", []) for s in sc.get("shots", [])]
    if not (8 <= len(shots) <= 16):
        return f"shot count {len(shots)} not in 8-16"
    for s in shots:
        if not s.get("first_frame") or not s.get("video_prompt"):
            return f"shot {s.get('shot_id')} missing fields"
    text = json.dumps(d, ensure_ascii=False)
    if len(re.findall(r"[一-鿿]", text)) < len(text) // 20:
        return "output not Chinese"
    return ""


def _valid_story(d: dict) -> str:
    sp = d.get("screenplay", "")
    if len(sp) < 300:
        return "screenplay too short"
    if re.search(r"镜头\s*\d|分镜|shot\s*\d|远景|近景|特写|摇镜|推轨",
                 sp):
        return "camera/storyboard terms leaked into story-only script"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/bailian.yaml"))
    args = ap.parse_args()
    spec = (load_yaml(Path(args.config)).get("models", {})
            .get("llm") or {})
    _BASES = {"qwen-maas": ("https://ws-ox5q19lbmn2u1drg.cn-beijing.maas"
                            ".aliyuncs.com/compatible-mode/v1",
                            "DASHSCOPE_API_KEY"),
              "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1",
                       "QWEN_API_KEY")}
    base, ev = _BASES.get(str(spec.get("name", "qwen")),
                          _BASES["qwen"])
    import os
    llm = OpenAICompatChat(
        spec.get("base_url") or base, spec.get("model", "qwen-max"),
        spec.get("api_key") or os.getenv(ev)
        or os.getenv("DASHSCOPE_API_KEY") or "",
        timeout=int(spec.get("timeout", 600)),
        extra_body=spec.get("extra_body"))
    for sub in ("bench", "story", "screenplays"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # ① 主题清单
    themes_p = OUT / "themes.json"
    if themes_p.exists():
        themes = json.loads(themes_p.read_text())
    else:
        print("== 生成主题清单", flush=True)
        themes = _call(llm, _THEME_PROMPT)
        assert len(themes.get("bench", [])) >= 30 \
            and len(themes.get("story", [])) >= 30 \
            and len(themes.get("ideas", [])) >= 40, "theme list short"
        themes_p.write_text(json.dumps(themes, ensure_ascii=False,
                                       indent=2))
    failed = []

    # ② 30 bench 式(带分镜)
    for i, t in enumerate(themes["bench"][:30]):
        key = f"b{i:02d}_{t['theme_key']}"[:60]
        raw_p = OUT / "bench" / f"{key}.json"
        if raw_p.exists():
            continue
        print(f"== bench {key} ({t['type']})", flush=True)
        try:
            d = _call(llm, PROMPT + "\n\nUSER INPUT:\n"
                      + json.dumps({"MODE": "with_prompts",
                                    "consistency_type": t["type"],
                                    "theme": t["theme"]},
                                   ensure_ascii=False))
            err = _valid_bench(d)
            if err:
                raise RuntimeError(err)
            raw_p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            sp = {"content": adapt_story(d), "role": {}}
            (OUT / "screenplays" / f"{key}.screenplay.json").write_text(
                json.dumps(sp, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"  ❌ {key}: {exc}", flush=True)
            failed.append(key)

    # ③ 30 纯剧本(无分镜)
    for i, t in enumerate(themes["story"][:30]):
        key = f"s{i:02d}_{t['theme_key']}"[:60]
        raw_p = OUT / "story" / f"{key}.json"
        if raw_p.exists():
            continue
        print(f"== story {key} ({t.get('type', '')})", flush=True)
        try:
            d = _call(llm, PROMPT + "\n\nUSER INPUT:\n"
                      + json.dumps({"MODE": "story_only",
                                    "consistency_type": t.get(
                                        "type", "Type A"),
                                    "theme": t["theme"]},
                                   ensure_ascii=False))
            err = _valid_story(d)
            if err:
                raise RuntimeError(err)
            raw_p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            sp = {"content": d["screenplay"],
                  "role": {}}
            (OUT / "screenplays" / f"{key}.screenplay.json").write_text(
                json.dumps(sp, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"  ❌ {key}: {exc}", flush=True)
            failed.append(key)

    # ④ ideas + 任务池
    ideas = [str(x).strip() for x in themes["ideas"][:40] if str(x).strip()]
    (OUT / "ideas.json").write_text(json.dumps(ideas, ensure_ascii=False,
                                               indent=2))
    sps = sorted((OUT / "screenplays").glob("*.screenplay.json"))
    pool = {"mix": {"screenplay_weight": 3, "idea_weight": 2},
            "screenplays": [
                {"file": str(p.relative_to(REPO)),
                 "prompt": json.loads(p.read_text())["content"][:24]
                 .replace("\n", " ")}
                for p in sps],
            "ideas": ideas}
    import yaml
    (REPO / "rl/configs/task_pool_train100.yaml").write_text(
        yaml.safe_dump(pool, allow_unicode=True, sort_keys=False))
    print(f"\nDONE: bench={len(list((OUT / 'bench').glob('*.json')))} "
          f"story={len(list((OUT / 'story').glob('*.json')))} "
          f"ideas={len(ideas)} pool={len(sps)}+{len(ideas)} "
          f"failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
