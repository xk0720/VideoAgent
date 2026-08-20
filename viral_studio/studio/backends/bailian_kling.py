"""百炼可灵 kling-v3 视频客户端 —— 主力视频后端。

协议 = 本仓 M0 真调用验证过的契约(2026-08-03, 五形态全 SUCCEEDED), 照实现不改:
  提交 POST {BASE}/services/aigc/video-generation/video-synthesis
       头 Authorization + X-DashScope-Async: enable
          (+ X-DashScope-OssResourceResolve: enable, 带 oss:// 媒体时)
  轮询 GET  {BASE}/tasks/{task_id} → SUCCEEDED 即下载(URL 24h 失效, 即下即存)
  上传 GET  /uploads?action=getPolicy&model=… → OSS 表单直传 → oss://key

M0 实测的三条硬契约(决定了 ActAgent 怎么开工单):
  · 模型按有无媒体二选一: 带媒体 → omni; 纯文本 → 标准模型(自动退化为文生视频)
  · media 数组混装 first_frame + refer 合法; **refer 的顺序即 <<<image_N>>> 编号**,
    first_frame 不占编号
  · 无 first_frame 的请求 aspect_ratio 必填
  · duration 域 [3,15] 整数; seed 无此字段(只记日志)
本机系统代理会掐断长轮询 → 会话 trust_env=False 直连。
"""
import json
import logging
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

log = logging.getLogger("viral_studio")


class BailianKlingClient:
    BASE = "https://dashscope.aliyuncs.com/api/v1"
    SYNTH = BASE + "/services/aigc/video-generation/video-synthesis"
    OMNI_MODEL = "kling/kling-v3-omni-video-generation"     # 带媒体
    TEXT_MODEL = "kling/kling-v3-video-generation"          # 纯文本
    DURATION_RANGE = (3, 15)

    def __init__(self, api_key: str, mode: str = "std", aspect_ratio: str = "9:16",
                 poll_interval_s: float = 15.0, timeout_s: float = 900.0):
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.mode = mode
        self.aspect_ratio = aspect_ratio
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._cache: dict = {}
        s = requests.Session()
        s.trust_env = False
        self._s = s

    @staticmethod
    def ref_token(n: int) -> str:
        """可灵引用方言: prompt 里第 N 张 refer 图写作 <<<image_N>>>。"""
        return f"<<<image_{n}>>>"

    def _headers(self, async_: bool = False, oss: bool = False) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if async_:
            h["X-DashScope-Async"] = "enable"
        if oss:
            h["X-DashScope-OssResourceResolve"] = "enable"
        return h

    def upload(self, path: str) -> str:
        if str(path).startswith(("http://", "https://", "oss://")):
            return str(path)
        p = Path(path).resolve()
        key = (str(p), p.stat().st_mtime_ns)
        if key in self._cache:
            return self._cache[key]
        r = self._s.get(f"{self.BASE}/uploads",
                        params={"action": "getPolicy", "model": self.OMNI_MODEL},
                        headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
        r.raise_for_status()
        pol = r.json()["data"]
        oss_key = f"{pol['upload_dir']}/{p.name}"
        with p.open("rb") as f:
            files = {"OSSAccessKeyId": (None, pol["oss_access_key_id"]),
                     "Signature": (None, pol["signature"]),
                     "policy": (None, pol["policy"]),
                     "x-oss-object-acl": (None, pol["x_oss_object_acl"]),
                     "x-oss-forbid-overwrite": (None, pol["x_oss_forbid_overwrite"]),
                     "key": (None, oss_key), "success_action_status": (None, "200"),
                     "file": (p.name, f)}
            up = self._s.post(pol["upload_host"], files=files, timeout=300)
        up.raise_for_status()
        url = f"oss://{oss_key}"
        self._cache[key] = url
        log.info("已上传 %s", p.name)
        return url

    def clamp_duration(self, seconds) -> int:
        lo, hi = self.DURATION_RANGE
        return max(lo, min(hi, int(math.ceil(float(seconds)))))

    def generate(self, prompt: str, duration, save_to: str,
                 first_frame: Optional[str] = None,
                 refer: Optional[List[str]] = None,
                 audio: bool = False,
                 aspect_ratio: Optional[str] = None) -> Tuple[bool, str, str]:
        """t2v / i2v(first_frame) / ref2v(refer×N) / 混装。返回 (ok, task_id, err)。"""
        media: List[dict] = []
        if first_frame:
            media.append({"type": "first_frame", "url": self.upload(first_frame)})
        for img in (refer or []):
            media.append({"type": "refer", "url": self.upload(img)})

        model = self.OMNI_MODEL if media else self.TEXT_MODEL   # 无媒体 → 自动退化为文生视频
        inp: dict = {"prompt": prompt}
        if media:
            inp["media"] = media
        params: dict = {"mode": self.mode, "audio": bool(audio),
                        "duration": self.clamp_duration(duration)}
        if not first_frame:                    # M0 实测: 无 first_frame 时必填
            params["aspect_ratio"] = aspect_ratio or self.aspect_ratio
        payload = {"model": model, "input": inp, "parameters": params}

        r = self._s.post(self.SYNTH, headers=self._headers(async_=True, oss=bool(media)),
                         json=payload, timeout=60)
        body = r.json() if r.text else {}
        task_id = (body.get("output") or {}).get("task_id", "")
        if r.status_code != 200 or not task_id:
            return False, "", (f"提交失败 HTTP {r.status_code}: "
                               f"{json.dumps(body, ensure_ascii=False)[:400]}")
        log.info("kling 已提交 task=%s model=%s media=%d duration=%ds",
                 task_id, model.split("/")[-1], len(media), params["duration"])

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                q = self._s.get(f"{self.BASE}/tasks/{task_id}",
                                headers=self._headers(), timeout=30)
            except requests.exceptions.RequestException as e:
                log.warning("轮询瞬态错误(%s), 继续", str(e)[:80])
                time.sleep(self.poll_interval_s)
                continue
            out = (q.json().get("output") or {})
            st = out.get("task_status", "UNKNOWN")
            if st == "SUCCEEDED":
                url = out.get("video_url") or (out.get("results") or {}).get("video_url")
                if not url:
                    return False, task_id, f"SUCCEEDED 但无 video_url: {out}"
                v = self._s.get(url, timeout=600)
                v.raise_for_status()
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                Path(save_to).write_bytes(v.content)
                return True, task_id, ""
            if st in ("FAILED", "CANCELED", "UNKNOWN"):
                return False, task_id, f"{st}: code={out.get('code')} message={out.get('message')}"
            time.sleep(self.poll_interval_s)
        return False, task_id, f"轮询超时(>{self.timeout_s:.0f}s)"
