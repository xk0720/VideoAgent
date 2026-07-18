"""S0 · 日志重放评估器:不花一分生成费,量一个模型的决策质量(2026-07-18)。

拿 eval_holdout.jsonl 里的每道"题"(历史上下文),让被测模型答一遍,
用生产同款确定性校验器打分:

    parse_ok   — 输出能解析出 JSON 吗
    in_menu    — 选的策略/工具在当时的菜单里吗
    refs_ok    — video_prompt 的 @ 引用 ⊆ 当时的槽位清单吗(条件决策)
    agree      — 与当年被记为"好"的决策一致吗(仅 label=true 的题计)
    pass_k     — 同一道题采样 k 次,全部结构合法的比例(稳定性)

用法(对任意 OpenAI 兼容端点,vLLM 服务的 Qwen 就是这么接):
    python scripts/rl/eval_replay.py data/rl/eval_holdout.jsonl \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B -k 4

三条基线先跑再谈收益(报告七-风险 2):原始底座 / 仅 SFT / 仅格式奖励。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from maestro.models.mllm_backends import _extract_json      # noqa: E402
from maestro.pipeline.ref_slots import validate_references  # noqa: E402


def _http_complete_fn(base_url: str, model: str, api_key: str,
                      temperature: float):
    import requests

    def complete(prompt: str) -> str:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
            json={"model": model, "temperature": temperature,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    return complete


def _slots_from_context(ctx: dict, strategy: str) -> list:
    by_strat = ctx.get("slots_by_strategy")
    if isinstance(by_strat, dict) and strategy in by_strat:
        return by_strat[strategy] or []
    return []


def score_one(sample: dict, replies: list[str]) -> dict:
    """一道题的 k 次作答 → 各项指标(每项 = 通过次数/k)。"""
    meta = sample.get("meta", {})
    stage = meta.get("stage", "")
    prompt_text = sample["prompt"][0]["content"]
    ctx = {}
    marker = "THIS TURN (JSON):\n"
    if marker in prompt_text:
        tail = prompt_text.split(marker, 1)[1]
        end = tail.rfind("\n\nSTRICT JSON")
        if end < 0:
            end = tail.rfind("\n\nRespond with STRICT JSON")
        try:
            ctx = json.loads(tail[:end] if end > 0 else tail)
        except json.JSONDecodeError:
            ctx = {}
    menu = ctx.get("menu")
    valid_names = ({m["name"] if isinstance(m, dict) else m for m in menu}
                   if isinstance(menu, list) else None)
    logged_good = None
    if sample.get("label"):
        try:
            logged_good = json.loads(sample["completion"][0]["content"])
        except json.JSONDecodeError:
            logged_good = None

    k = len(replies)
    n_parse = n_menu = n_refs = n_agree = n_struct = 0
    refs_applicable = agree_applicable = 0
    for raw in replies:
        data = _extract_json(raw)
        ok_parse = isinstance(data, dict)
        n_parse += ok_parse
        key = "tool" if stage == "repair" else "strategy"
        choice = str((data or {}).get(key, "")) if ok_parse else ""
        ok_menu = (choice in valid_names) if (ok_parse and valid_names) \
            else ok_parse
        n_menu += ok_menu
        ok_refs = True
        vp = (data or {}).get("video_prompt") if ok_parse else None
        if isinstance(vp, str) and vp.strip():
            slots = _slots_from_context(ctx, choice)
            refs_applicable += 1
            _fixed, audit = validate_references(vp, slots)
            ok_refs = audit["ok"]
            n_refs += ok_refs
        if logged_good is not None and ok_parse:
            agree_applicable += 1
            n_agree += (choice == str(logged_good.get(key, "")))
        n_struct += (ok_parse and ok_menu and ok_refs)
    return {"stage": stage, "k": k,
            "parse_ok": n_parse / k, "in_menu": n_menu / k,
            "refs_ok": (n_refs / refs_applicable) if refs_applicable else None,
            "agree": (n_agree / agree_applicable) if agree_applicable else None,
            "pass_k": 1.0 if n_struct == k else 0.0}


def run_eval(samples: list[dict], complete_fn, k: int = 4) -> dict:
    rows = []
    for s in samples:
        replies = [complete_fn(s["prompt"][0]["content"]) for _ in range(k)]
        rows.append(score_one(s, replies))
    agg: dict = {}
    for stage in sorted({r["stage"] for r in rows}):
        sub = [r for r in rows if r["stage"] == stage]

        def _mean(key):
            vals = [r[key] for r in sub if r[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        agg[stage] = {"n": len(sub), "parse_ok": _mean("parse_ok"),
                      "in_menu": _mean("in_menu"), "refs_ok": _mean("refs_ok"),
                      "agree": _mean("agree"), "pass_k": _mean("pass_k")}
    return {"per_stage": agg, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("-k", "--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    samples = []
    for line in Path(args.dataset).read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    fn = _http_complete_fn(args.base_url, args.model,
                           os.getenv(args.api_key_env, ""), args.temperature)
    result = run_eval(samples, fn, k=args.samples)
    print(json.dumps(result["per_stage"], ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
