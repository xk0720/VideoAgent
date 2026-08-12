"""百炼 wan2.2-animate-move(图生动作)客户端。

协议照搬本仓已实测的 M0 契约(Maestro/scripts/playground/bailian_kling_probe.py,
2026-08-03 真调用全 SUCCEEDED), 不另起炉灶:

  上传  GET  {BASE}/uploads?action=getPolicy&model=… → OSS 表单直传 → oss://key
  提交  POST {BASE}/services/aigc/image2video/video-synthesis
        头 Authorization: Bearer $DASHSCOPE_API_KEY + X-DashScope-Async: enable
        + X-DashScope-OssResourceResolve: enable(带 oss:// 媒体时)
  轮询  GET  {BASE}/tasks/{task_id}, SUCCEEDED → 下载 video_url(24h 失效, 即下即存)

M0 实战教训一并继承: 本机系统代理会掐断长轮询 → requests 会话
trust_env=False 直连(阿里云国内可达无需代理)。

为什么是 animate-move 而不是 animate-mix: move 的人物**和背景都来自参考图**,
原片只贡献动作 —— 正是"背景与 hook 图一致"的需求; mix 保留的是原片背景。
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

log = logging.getLogger("hook_remake")


class BailianAnimateClient:
    def __init__(self, api_key: str, model: str = "wan2.2-animate-move",
                 mode: str = "wan-std", check_image: bool = True,
                 watermark: bool = False,
                 base_url: str = "https://dashscope.aliyuncs.com/api/v1",
                 poll_interval_s: float = 15.0, timeout_s: float = 900.0):
        if not api_key:
            raise RuntimeError("缺少 API key: 请设置环境变量 DASHSCOPE_API_KEY "
                               "(或仓库根 .env)")
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.check_image = check_image
        self.watermark = watermark
        self.base = base_url.rstrip("/")
        self.synth = f"{self.base}/services/aigc/image2video/video-synthesis"
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._upload_cache: dict = {}          # (resolved, mtime_ns) → oss url
        s = requests.Session()
        s.trust_env = False                    # M0: 系统代理会掐断长轮询
        self._s = s

    # ── 基础 ────────────────────────────────────────────────
    def _headers(self, async_: bool = False, oss: bool = False) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}",
             "Content-Type": "application/json"}
        if async_:
            h["X-DashScope-Async"] = "enable"
        if oss:
            h["X-DashScope-OssResourceResolve"] = "enable"
        return h

    # ── 上传: getPolicy → OSS 表单直传 → oss:// ─────────────
    def upload(self, path: str) -> str:
        p = Path(path).resolve()
        cache_key = (str(p), p.stat().st_mtime_ns)
        if cache_key in self._upload_cache:
            return self._upload_cache[cache_key]

        r = self._s.get(f"{self.base}/uploads",
                        params={"action": "getPolicy", "model": self.model},
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=30)
        r.raise_for_status()
        pol = r.json()["data"]
        key = f"{pol['upload_dir']}/{p.name}"
        with p.open("rb") as f:
            files = {
                "OSSAccessKeyId": (None, pol["oss_access_key_id"]),
                "Signature": (None, pol["signature"]),
                "policy": (None, pol["policy"]),
                "x-oss-object-acl": (None, pol["x_oss_object_acl"]),
                "x-oss-forbid-overwrite": (None, pol["x_oss_forbid_overwrite"]),
                "key": (None, key),
                "success_action_status": (None, "200"),
                "file": (p.name, f),
            }
            up = self._s.post(pol["upload_host"], files=files, timeout=300)
        up.raise_for_status()
        url = f"oss://{key}"
        self._upload_cache[cache_key] = url
        log.info("已上传 %s → %s", p.name, url)
        return url

    # ── 提交 + 轮询 + 下载 ──────────────────────────────────
    def animate(self, image_oss: str, video_oss: str,
                save_to: str) -> Tuple[bool, str, str]:
        """一次完整生成。返回 (ok, task_id, err)。ok=True 时视频已存 save_to。"""
        payload = {
            "model": self.model,
            "input": {"image_url": image_oss, "video_url": video_oss,
                      "watermark": self.watermark},
            "parameters": {"mode": self.mode, "check_image": self.check_image},
        }
        r = self._s.post(self.synth, headers=self._headers(async_=True, oss=True),
                         json=payload, timeout=60)
        body = r.json() if r.text else {}
        task_id = (body.get("output") or {}).get("task_id", "")
        if r.status_code != 200 or not task_id:
            err = f"提交失败 HTTP {r.status_code}: {json.dumps(body, ensure_ascii=False)[:300]}"
            log.error(err)
            return False, task_id, err
        log.info("任务已提交 task_id=%s (mode=%s)", task_id, self.mode)

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            q = self._s.get(f"{self.base}/tasks/{task_id}",
                            headers=self._headers(), timeout=30)
            out = (q.json().get("output") or {})
            st = out.get("task_status", "UNKNOWN")
            if st == "SUCCEEDED":
                url = self._extract_video_url(out)
                if not url:
                    return False, task_id, f"SUCCEEDED 但未找到 video_url: {out}"
                v = self._s.get(url, timeout=600)
                v.raise_for_status()
                Path(save_to).write_bytes(v.content)
                log.info("任务 %s 完成 → %s", task_id, save_to)
                return True, task_id, ""
            if st in ("FAILED", "CANCELED", "UNKNOWN"):
                err = (f"{st}: code={out.get('code')} "
                       f"message={out.get('message')}")
                log.error("任务 %s %s", task_id, err)
                return False, task_id, err
            log.info("任务 %s %s …", task_id, st)
            time.sleep(self.poll_interval_s)
        return False, task_id, f"轮询超时(>{self.timeout_s:.0f}s)"

    @staticmethod
    def _extract_video_url(out: dict) -> Optional[str]:
        """兼容 output.video_url / output.results.video_url /
        output.results[0].{video_url|url} 三种返回形态。"""
        if out.get("video_url"):
            return out["video_url"]
        res = out.get("results")
        if isinstance(res, dict):
            return res.get("video_url") or res.get("url")
        if isinstance(res, list) and res:
            return res[0].get("video_url") or res[0].get("url")
        return None
