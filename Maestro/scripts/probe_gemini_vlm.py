#!/usr/bin/env python
"""Gemini VLM 连通性探针(2026-08-19):在服务器上验证 Gemini 能不能
用 —— 三级递进,逐级报告,哪级断了病灶就在哪级:

  ① 网络层:能否连上 generativelanguage.googleapis.com(TCP/TLS)
  ② 文本层:generateContent 纯文本一问一答(key 有效性/配额)
  ③ 视频层:上传一段本地 mp4(inline base64)让它描述 —— 我们评审
    链真正的用法(原生视频通道)

用法(仓库根):
  python scripts/probe_gemini_vlm.py                       # ①+②
  python scripts/probe_gemini_vlm.py --video path/to.mp4   # ①+②+③
  可选 --model gemini-3.5-flash(默认,与主配置一致)
需要 .env 或环境变量里有 GEMINI_API_KEY(GEMINI_BASE_URL 可选覆盖)。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import requests  # noqa: E402


def step(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"{mark} {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--video", default="", help="本地 mp4(可选,测③)")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    base = (os.getenv("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com").rstrip("/")
    key = os.getenv("GEMINI_API_KEY", "")
    host = urllib.parse.urlparse(base).hostname
    print(f"目标: {base} | 模型: {args.model} | key: "
          f"{'有(' + key[:6] + '****)' if key else '无'}\n")

    # ① 网络层
    t0 = time.time()
    try:
        socket.create_connection((host, 443), timeout=10).close()
        step("① TCP 443 连通", True, f"{time.time() - t0:.1f}s")
    except Exception as exc:
        step("① TCP 443 连通", False, str(exc)[:120])
        print("\n结论:服务器到 Google 的网络不通(与 key 无关)。")
        return 1

    if not key:
        step("② key", False, "GEMINI_API_KEY 缺失(.env 或环境变量)")
        return 1

    url = (f"{base}/v1beta/models/{args.model}:generateContent"
           f"?key={key}")

    # ② 文本层
    t0 = time.time()
    try:
        r = requests.post(url, json={"contents": [{"parts": [
            {"text": "Reply with exactly: PONG"}]}]},
            timeout=args.timeout)
        body = r.json()
        text = (body.get("candidates", [{}])[0].get("content", {})
                .get("parts", [{}])[0].get("text", ""))
        if r.status_code == 200 and text.strip():
            step("② 文本 generateContent", True,
                 f"{time.time() - t0:.1f}s 回复: {text.strip()[:40]}")
        else:
            step("② 文本 generateContent", False,
                 f"HTTP {r.status_code}: "
                 f"{json.dumps(body, ensure_ascii=False)[:300]}")
            return 1
    except Exception as exc:
        step("② 文本 generateContent", False, str(exc)[:200])
        return 1

    # ③ 视频层(可选)
    if args.video:
        vp = Path(args.video)
        if not vp.exists():
            step("③ 视频理解", False, f"文件不存在: {vp}")
            return 1
        mb = vp.stat().st_size / 1e6
        if mb > 18:
            step("③ 视频理解", False,
                 f"文件 {mb:.1f}MB 超 18MB inline 限额,换个短片")
            return 1
        data = base64.b64encode(vp.read_bytes()).decode()
        t0 = time.time()
        try:
            r = requests.post(url, json={"contents": [{"parts": [
                {"inline_data": {"mime_type": "video/mp4",
                                 "data": data}},
                {"text": "用一句中文描述这段视频的画面内容。"}]}]},
                timeout=max(args.timeout, 300))
            body = r.json()
            text = (body.get("candidates", [{}])[0].get("content", {})
                    .get("parts", [{}])[0].get("text", ""))
            if r.status_code == 200 and text.strip():
                step("③ 原生视频理解", True,
                     f"{time.time() - t0:.1f}s({mb:.1f}MB)"
                     f" 回复: {text.strip()[:80]}")
            else:
                step("③ 原生视频理解", False,
                     f"HTTP {r.status_code}: "
                     f"{json.dumps(body, ensure_ascii=False)[:300]}")
                return 1
        except Exception as exc:
            step("③ 原生视频理解", False, str(exc)[:200])
            return 1
    else:
        print("(未传 --video,跳过③;评审链用的是原生视频通道,"
              "建议补测:--video outputs/<某run>/shot000/*.mp4)")

    print("\n结论:Gemini VLM 在本机可用" + ("(含视频通道)"
          if args.video else "(文本通道;视频通道待 --video 补测)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
