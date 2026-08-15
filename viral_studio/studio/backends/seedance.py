"""WaveSpeed seedance-2.0 text-to-video 客户端 —— self_create / vo 路线。

协议 copy 自本仓已验证实现(Maestro WaveSpeedClient, 官方文档 2026-07 核对):
  上传  POST {BASE}/media/upload/binary (multipart file, ≤300MB) → 公网 URL
  提交  POST {BASE}/{model_id} → data.id
  轮询  GET  {BASE}/predictions/{id}/result → status=completed → outputs[0] 下载

为什么走 text-to-video 端点: `reference_images`(≤9 张, prompt 里以 @ImageN 指代)
**只在 t2v 端点上被验证存在**; i2v 端点 schema 只有 image+last_image。
本客户端因此不提供 first_frame —— 身份靠 reference_images, 内容靠 prompt。

时长: 域 4–15 整数秒(用户裁决同样要求整数)。短段按 max(4, ceil) 生成,
执行层再帧级剪回目标时长。
"""
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

log = logging.getLogger("viral_studio")


class SeedanceClient:
    BASE = "https://api.wavespeed.ai/api/v3"
    UPLOAD_URL = BASE + "/media/upload/binary"
    DURATION_RANGE = (4, 15)
    MAX_REFS = 9

    def __init__(self, api_key: str,
                 model_id: str = "bytedance/seedance-2.0/text-to-video",
                 resolution: str = "720p", generate_audio: bool = False,
                 aspect_ratio: str = "9:16", poll_interval_s: float = 5.0,
                 timeout_s: float = 1200.0):
        if not api_key:
            raise RuntimeError("缺少 WAVESPEED_API_KEY")
        self.api_key = api_key
        self.model_id = model_id
        self.resolution = resolution
        self.generate_audio = generate_audio
        self.aspect_ratio = aspect_ratio
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._cache: dict = {}
        s = requests.Session()
        s.trust_env = False
        self._s = s

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def upload(self, path: str) -> str:
        vp = str(path)
        if vp.startswith(("http://", "https://")):
            return vp
        p = Path(vp).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"媒体不存在: {vp}")
        key = (str(p), p.stat().st_mtime_ns)
        if key in self._cache:
            return self._cache[key]
        last: Optional[Exception] = None
        for attempt in range(3):                    # 上传遇 SSL EOF 的实测教训
            try:
                with p.open("rb") as fh:
                    r = self._s.post(self.UPLOAD_URL,
                                     headers={"Authorization": f"Bearer {self.api_key}"},
                                     files={"file": (p.name, fh)}, timeout=300)
                r.raise_for_status()
                url = r.json()["data"]["download_url"]
                self._cache[key] = url
                log.info("已上传 %s", p.name)
                return url
            except Exception as e:                  # noqa: BLE001
                last = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"上传 {p.name} 三次失败: {last}")

    def snap_duration(self, seconds: float) -> int:
        lo, hi = self.DURATION_RANGE
        want = int(seconds) if float(seconds) == int(seconds) else int(seconds) + 1
        return max(lo, min(hi, want))

    def generate(self, prompt: str, duration_s: float, save_to: str,
                 reference_images: Optional[List[str]] = None,
                 seed: Optional[int] = None) -> Tuple[bool, str, str]:
        """返回 (ok, task_id, err); ok=True 时视频已落 save_to。"""
        payload = {
            "prompt": prompt,
            "duration": self.snap_duration(duration_s),
            "resolution": self.resolution,
            "generate_audio": self.generate_audio,
            "aspect_ratio": self.aspect_ratio,
        }
        if seed is not None:
            payload["seed"] = seed
        refs = list(reference_images or [])
        if len(refs) > self.MAX_REFS:
            log.info("参考图 %d 张超上限, 截到 %d 张", len(refs), self.MAX_REFS)
            refs = refs[:self.MAX_REFS]
        if refs:
            payload["reference_images"] = [self.upload(p) for p in refs]

        r = self._s.post(f"{self.BASE}/{self.model_id}", json=payload,
                         headers=self._headers(), timeout=60)
        if r.status_code >= 400:
            # 带上响应体: WaveSpeed 在 body 里说明是哪个字段不对
            return False, "", f"提交失败 HTTP {r.status_code}: {r.text[:600]}"
        task_id = r.json()["data"]["id"]
        log.info("seedance 已提交 task=%s duration=%ds refs=%d",
                 task_id, payload["duration"], len(refs))

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                q = self._s.get(f"{self.BASE}/predictions/{task_id}/result",
                                headers=self._headers(), timeout=30)
            except requests.exceptions.RequestException as e:
                log.warning("轮询瞬态错误(%s), 继续", str(e)[:80])
                time.sleep(self.poll_interval_s)
                continue
            if q.status_code >= 500:
                time.sleep(self.poll_interval_s)
                continue
            if q.status_code >= 400:
                return False, task_id, f"轮询失败 HTTP {q.status_code}: {q.text[:400]}"
            data = q.json()["data"]
            st = data.get("status")
            if st == "completed":
                url = data["outputs"][0]
                try:
                    v = self._s.get(url, timeout=600)
                    v.raise_for_status()
                except requests.exceptions.RequestException as e:
                    log.warning("下载抖动(%s), 重新轮询取新 URL", str(e)[:80])
                    time.sleep(self.poll_interval_s)
                    continue
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                Path(save_to).write_bytes(v.content)
                return True, task_id, ""
            if st == "failed":
                return False, task_id, f"failed: {str(data.get('error', 'unknown'))[:400]}"
            time.sleep(self.poll_interval_s)
        return False, task_id, f"轮询超时(>{self.timeout_s:.0f}s)"
