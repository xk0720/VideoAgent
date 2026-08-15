"""百炼 wan2.2-animate-move(图生动作)客户端 —— reuse_motion 路线。

协议 = 已在 hook_remake 真调用验证过的契约(2026-08-11..13, 两轮 pro/std 全流程):
  上传  GET  {BASE}/uploads?action=getPolicy&model=… → OSS 表单直传 → oss://key
  提交  POST {BASE}/services/aigc/image2video/video-synthesis
        头 Authorization + X-DashScope-Async: enable
        + X-DashScope-OssResourceResolve: enable(带 oss:// 媒体时)
  轮询  GET  {BASE}/tasks/{task_id} → SUCCEEDED 即下载(URL 24h 失效, 即下即存)

为什么是 move 而不是 mix: move 的人物**和背景都来自参考图**, 驱动视频只贡献动作
——正合"人物换成我们的 hook"的需求; mix 保留的是驱动视频的背景。

实测约束(全部由 validate.py 在规划期拦截, 这里只做兜底与如实上报):
  驱动视频 2–30s / ≤200MB / 宽高∈[200,2048]; 参考图 ≤5MB / ∈[200,4096]
  前置检测: 单人 + 完整人脸, 否则 InvalidVideo.NoHuman / InvalidVideo.FullFace
  (确定性, 与 mode 无关 —— 重试同判, 别浪费墙钟)
本机系统代理会掐断长轮询 → 会话 trust_env=False 直连。
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

log = logging.getLogger("viral_studio")

# 前置检测类错误: 确定性拒绝, 重试无意义(实测 pro/std 两轮逐镜同判)
DETERMINISTIC_REJECTS = ("InvalidVideo.NoHuman", "InvalidVideo.FullFace",
                         "InvalidVideo.", "InvalidParameter")


class BailianAnimateClient:
    BASE = "https://dashscope.aliyuncs.com/api/v1"

    def __init__(self, api_key: str, model: str = "wan2.2-animate-move",
                 mode: str = "wan-std", check_image: bool = True,
                 watermark: bool = False, poll_interval_s: float = 15.0,
                 timeout_s: float = 900.0):
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.check_image = check_image
        self.watermark = watermark
        self.synth = f"{self.BASE}/services/aigc/image2video/video-synthesis"
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._cache: dict = {}
        s = requests.Session()
        s.trust_env = False
        self._s = s

    def _headers(self, async_: bool = False, oss: bool = False) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}",
             "Content-Type": "application/json"}
        if async_:
            h["X-DashScope-Async"] = "enable"
        if oss:
            h["X-DashScope-OssResourceResolve"] = "enable"
        return h

    def upload(self, path: str) -> str:
        p = Path(path).resolve()
        key = (str(p), p.stat().st_mtime_ns)
        if key in self._cache:
            return self._cache[key]
        r = self._s.get(f"{self.BASE}/uploads",
                        params={"action": "getPolicy", "model": self.model},
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=30)
        r.raise_for_status()
        pol = r.json()["data"]
        oss_key = f"{pol['upload_dir']}/{p.name}"
        with p.open("rb") as f:
            files = {
                "OSSAccessKeyId": (None, pol["oss_access_key_id"]),
                "Signature": (None, pol["signature"]),
                "policy": (None, pol["policy"]),
                "x-oss-object-acl": (None, pol["x_oss_object_acl"]),
                "x-oss-forbid-overwrite": (None, pol["x_oss_forbid_overwrite"]),
                "key": (None, oss_key),
                "success_action_status": (None, "200"),
                "file": (p.name, f),
            }
            up = self._s.post(pol["upload_host"], files=files, timeout=300)
        up.raise_for_status()
        url = f"oss://{oss_key}"
        self._cache[key] = url
        log.info("已上传 %s", p.name)
        return url

    def animate(self, ref_image: str, driving_video: str,
                save_to: str) -> Tuple[bool, str, str]:
        """一次生成。返回 (ok, task_id, err); ok=True 时视频已落 save_to。"""
        image_oss = self.upload(ref_image)
        video_oss = self.upload(driving_video)
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
            return False, "", (f"提交失败 HTTP {r.status_code}: "
                               f"{json.dumps(body, ensure_ascii=False)[:300]}")
        log.info("animate 已提交 task=%s mode=%s", task_id, self.mode)

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                q = self._s.get(f"{self.BASE}/tasks/{task_id}",
                                headers=self._headers(), timeout=30)
            except requests.exceptions.RequestException as e:   # 轮询抖动不弃任务
                log.warning("轮询瞬态错误(%s), 继续", str(e)[:80])
                time.sleep(self.poll_interval_s)
                continue
            out = (q.json().get("output") or {})
            st = out.get("task_status", "UNKNOWN")
            if st == "SUCCEEDED":
                url = self._video_url(out)
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

    @staticmethod
    def _video_url(out: dict) -> Optional[str]:
        if out.get("video_url"):
            return out["video_url"]
        res = out.get("results")
        if isinstance(res, dict):
            return res.get("video_url") or res.get("url")
        if isinstance(res, list) and res:
            return res[0].get("video_url") or res[0].get("url")
        return None


def is_deterministic_reject(err: str) -> bool:
    """前置检测类拒绝 → 不重试(实测重试同判, 只烧墙钟不烧钱)。"""
    return any(code in (err or "") for code in DETERMINISTIC_REJECTS)
