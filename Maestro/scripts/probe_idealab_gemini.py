#!/usr/bin/env python
"""idealab 内部 Gemini 网关探针(2026-08-19 用户令:curl 已验证通,
此脚本 = 那条 curl 的代码化 + 我们真正用法的两级加测)。

  ① 文本 chat(key/型号有效性)
  ② 远程 video_url(= 用户那条 curl 原样,证协议对齐)
  ③ 本地 mp4 → base64 data URI video_url(评审链真正的用法 ——
    这一级绿了,IdealabGeminiVLM 的原生视频通道即可用;红了也不慌,
    后端会自动退回抽帧路)

用法(仓库根;.env 需有 IDEALAB_API_KEY):
  python scripts/probe_idealab_gemini.py
  python scripts/probe_idealab_gemini.py --video outputs/<run>/shot000/xx.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from maestro.config import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import requests  # noqa: E402

DEFAULT_BASE = "https://idealab-external.alibaba-inc.com/api/openai/v1"
REMOTE_DEMO = "https://www.w3school.com.cn/example/html5/mov_bbb.mp4"


def step(name, ok, detail=""):
    print(f"{'✅' if ok else '❌'} {name}"
          + (f" — {detail}" if detail else ""), flush=True)
    return ok


def chat(base, key, model, content, timeout=300):
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [
                  {"role": "system", "content": "你是个视频助手"},
                  {"role": "user", "content": content}]},
        timeout=timeout)
    if r.status_code >= 400:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"
    try:
        return r.json()["choices"][0]["message"]["content"], ""
    except Exception:
        return None, f"unexpected body: {r.text[:300]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--video", default="", help="本地 mp4(测③)")
    args = ap.parse_args()
    base = (os.getenv("IDEALAB_BASE_URL") or DEFAULT_BASE).rstrip("/")
    key = os.getenv("IDEALAB_API_KEY", "")
    print(f"目标: {base} | 模型: {args.model} | key: "
          f"{'有(' + key[:6] + '****)' if key else '无'}\n")
    if not key:
        step("key", False, "IDEALAB_API_KEY 缺失(写进 .env)")
        return 1

    # ① 文本
    t0 = time.time()
    text, err = chat(base, key, args.model,
                     "只回复两个字:收到", timeout=120)
    if not step("① 文本 chat", text is not None,
                err or f"{time.time() - t0:.1f}s 回复: {text[:40]}"):
        return 1

    # ② 远程 video_url(用户 curl 原样)
    t0 = time.time()
    text, err = chat(base, key, args.model, [
        {"type": "text", "text": "视频讲了什么?一句话。"},
        {"type": "video_url", "video_url": {"url": REMOTE_DEMO},
         "video_metadata": {"fps": 5}}])
    if not step("② 远程 video_url(=你的 curl)", text is not None,
                err or f"{time.time() - t0:.1f}s 回复: {text[:60]}"):
        return 1

    # ③ 本地 mp4 → base64 data URI(评审链真实用法)
    if args.video:
        vp = Path(args.video)
        if not vp.exists():
            step("③ 本地视频 data URI", False, f"文件不存在: {vp}")
            return 1
        mb = vp.stat().st_size / 1e6
        if mb > 30:
            step("③ 本地视频 data URI", False, f"{mb:.1f}MB 超 30MB,换短片")
            return 1
        data = base64.b64encode(vp.read_bytes()).decode()
        t0 = time.time()
        text, err = chat(base, key, args.model, [
            {"type": "text", "text": "用一句中文描述这段视频的画面内容。"},
            {"type": "video_url",
             "video_url": {"url": f"data:video/mp4;base64,{data}"},
             "video_metadata": {"fps": 5}}])
        ok = step("③ 本地视频 data URI", text is not None,
                  err or f"{time.time() - t0:.1f}s({mb:.1f}MB)"
                         f" 回复: {text[:80]}")
        print("\n结论:" + ("原生视频通道全通,IdealabGeminiVLM 满血可用"
              if ok else "data URI 被拒 —— 后端会自动退抽帧路,评审仍可用,"
                         "但把③的报错发我"))
        return 0 if ok else 1
    print("\n(未传 --video:①②已通;补测 ③ 用 "
          "--video outputs/<某run>/shot000/*.mp4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
