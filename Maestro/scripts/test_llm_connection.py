#!/usr/bin/env python
"""LLM 连接诊断:用管线【同一条代码路径】测 openai / gpt-5.6-sol 连通性。

复用 build_llm + OpenAICompatLLM(不另写请求逻辑,测的就是生产用的那条
路),分三步,逐步定位问题在哪一层:

    ① 配置解析   —— base_url / model / key 从哪来、是否在场(掩码显示)
    ② 端点可达   —— GET {base_url}/models(网络/认证问题在这一步现形,
                    和"模型名对不对"解耦)
    ③ 最小补全   —— complete("Reply with exactly: pong"),打印时延与
                    原样回复;HTTP 错误时打印状态码 + 响应体前 500 字
                    (400 的 body 通常直接写明参数错在哪)

用法:
    python scripts/test_llm_connection.py                    # 按 basic.yaml
    python scripts/test_llm_connection.py --model gpt-5.6-sol
    python scripts/test_llm_connection.py --timeout 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import yaml  # noqa: E402

from maestro.models.llm import build_llm  # noqa: E402
from maestro.models.llm_backends import _is_reasoning_model  # noqa: E402


def _mask(key: str | None) -> str:
    if not key:
        return "<MISSING>"
    return f"{key[:6]}...{key[-4:]} (len={len(key)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config",
                    default=str(REPO_ROOT / "configs" / "basic.yaml"))
    ap.add_argument("--provider", default="config",
                    choices=["config", "gemini"],
                    help="config=按 yaml;gemini=走 Google 的 OpenAI 兼容"
                         "端点测 Gemini 文本模型($GEMINI_API_KEY)")
    ap.add_argument("--model", default="", help="覆盖 config 里的模型名")
    ap.add_argument("--base-url", default="", help="覆盖 base_url")
    ap.add_argument("--prompt", default="Reply with exactly: pong")
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()

    if args.provider == "gemini":
        # 原生 generateContent 后端(用户指示 2026-07-29):与 GeminiVLM
        # 同端点形状/同 key/同 $GEMINI_BASE_URL(自定义中转同样生效)——
        # 评审走得通的网络路,brain 就走得通。默认 gemini-3.5-pro
        # (Pro=文本最强档),$GEMINI_TEXT_MODEL / --model 可覆盖。
        llm_cfg = {"name": "gemini"}
        if args.model:
            llm_cfg["model"] = args.model
    else:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        llm_cfg = dict((cfg.get("models") or {}).get("llm") or {})
        if args.model:
            llm_cfg["model"] = args.model
    if args.base_url:
        llm_cfg["base_url"] = args.base_url

    # ── ① 配置解析(和管线同一工厂)─────────────────────────────────
    llm = build_llm(llm_cfg)
    print("── ① 配置解析(build_llm 同路)")
    print(f"   name        = {getattr(llm, 'name', type(llm).__name__)}")
    print(f"   base_url    = {getattr(llm, 'base_url', '<n/a>')}")
    print(f"   model       = {getattr(llm, 'model', '<n/a>')}")
    reasoning = _is_reasoning_model(getattr(llm, "model", ""))
    print(f"   模型族       = {'gpt-5/o 系(max_completion_tokens,无 temperature)' if reasoning else '经典(max_tokens + temperature)'}")
    key = getattr(llm, "api_key", None)
    src = ("$GEMINI_API_KEY" if args.provider == "gemini"
           else "config.api_key" if llm_cfg.get("api_key")
           else "$OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY")
           else "$LLM_API_KEY" if os.getenv("LLM_API_KEY") else "无")
    print(f"   api_key     = {_mask(key)}  来源: {src}")
    if not key:
        need = ("GEMINI_API_KEY" if args.provider == "gemini"
                else "OPENAI_API_KEY")
        print(f"   ✗ 没有 key —— 先解决这个(export {need}=... 或在 .env "
              f"里填上;本仓 .env 存在该行但值为空同样算缺)")
        return 1

    import requests

    # ── ② 端点可达(与模型名无关的网络/认证探针)──────────────────────
    if args.provider == "gemini":
        probe_url = f"{llm.base_url}/v1beta/models"
        probe_headers = {"x-goog-api-key": key}
    else:
        probe_url = f"{llm.base_url}/models"
        probe_headers = {"Authorization": f"Bearer {key}"}
    print(f"── ② 端点可达性  GET {probe_url}")
    t0 = time.time()
    try:
        r = requests.get(probe_url, headers=probe_headers,
                         timeout=args.timeout)
        print(f"   HTTP {r.status_code}  ({time.time() - t0:.1f}s)")
        if r.status_code == 401:
            print("   ✗ 认证失败 —— key 无效或过期")
            return 1
        if r.status_code == 200:
            body = r.json()
            if args.provider == "gemini":     # 原生:{"models":[{"name":"models/…"}]}
                ids = [str(m.get("name", "")).removeprefix("models/")
                       for m in body.get("models", [])]
            else:                              # OpenAI 形:{"data":[{"id":…}]}
                ids = [m.get("id", "") for m in body.get("data", [])]
            hit = getattr(llm, "model", "") in ids
            print(f"   模型列表 {len(ids)} 个;目标模型"
                  f"{'在列 ✓' if hit else ' 不在列 ✗(检查模型名/账号权限)'}")
            if not hit:
                pat = ("pro",) if args.provider == "gemini" \
                    else ("5.6", "gpt-5")
                near = [i for i in ids
                        if any(t in i for t in pat)][:8]
                if near:
                    print(f"   相近可用: {near}")
    except requests.exceptions.RequestException as exc:
        print(f"   ✗ 网络层失败({time.time() - t0:.1f}s): {exc}")
        print("   → 这是连接问题(DNS/代理/防火墙),不是模型或参数问题")
        return 1

    # ── ③ 最小补全(管线真正调用的 complete())────────────────────────
    print(f"── ③ 最小补全  POST /chat/completions  model={llm.model}")
    t0 = time.time()
    try:
        reply = llm.complete(args.prompt, timeout=args.timeout)
        dt = time.time() - t0
        print(f"   ✓ 成功({dt:.1f}s)回复原文: {reply[:200]!r}")
        return 0
    except requests.exceptions.HTTPError as exc:
        resp = exc.response
        print(f"   ✗ HTTP {resp.status_code}({time.time() - t0:.1f}s)")
        print(f"   响应体(前 500 字): {resp.text[:500]}")
        if resp.status_code == 400:
            print("   → 400 多为参数/模型名问题(body 里通常写明);"
                  "注意 gpt-5.x 拒收 max_tokens 与自定义 temperature")
        elif resp.status_code == 429:
            print("   → 限流/配额,稍后重试或查账户用量")
        return 1
    except requests.exceptions.RequestException as exc:
        print(f"   ✗ 网络层失败({time.time() - t0:.1f}s): {exc}")
        return 1
    except RuntimeError as exc:
        print(f"   ✗ {exc}")
        print("   → GeminiLLM 的报错自带 HTTP 状态与响应体前 500 字,"
              "按 body 提示排查(模型名/权限/配额)")
        return 1
    except Exception as exc:
        print(f"   ✗ {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
