"""Sonilo 音乐生成(WaveSpeed) —— 卡点开场配乐。

两条路线, 同一套 submit → poll → download 协议(与 seedance 一致):
  text_to_music  POST {BASE}/sonilo/text-to-music   {prompt, duration:1-360}
                 $0.0025/秒; 时长可精确指定
  video_to_music POST {BASE}/sonilo/video-to-music  {video:URL, prompt?}
                 $0.009/秒(按视频长度); 模型看着画面配乐, 卡点由它自己对齐

选型说明: 之前误信仓库里一句 UniVA 移植的旧注释("没有音乐端点")而自造鼓点,
实际平台两条音乐链路都在, 且比手工合成便宜得多(6秒 ≈ 1.5-5 分钱)。
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

log = logging.getLogger("viral_studio")


class SoniloMusicClient:
    BASE = "https://api.wavespeed.ai/api/v3"
    UPLOAD_URL = BASE + "/media/upload/binary"

    def __init__(self, api_key: str, poll_interval_s: float = 3.0,
                 timeout_s: float = 600.0):
        if not api_key:
            raise RuntimeError("缺少 WAVESPEED_API_KEY")
        self.api_key = api_key
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        s = requests.Session()
        s.trust_env = False
        self._s = s

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def upload(self, path: str) -> str:
        if str(path).startswith(("http://", "https://")):
            return str(path)
        p = Path(path).resolve()
        with p.open("rb") as fh:
            r = self._s.post(self.UPLOAD_URL,
                             headers={"Authorization": f"Bearer {self.api_key}"},
                             files={"file": (p.name, fh)}, timeout=300)
        r.raise_for_status()
        return r.json()["data"]["download_url"]

    def _run(self, model_id: str, payload: dict, save_to: str) -> Tuple[bool, str, str]:
        r = self._s.post(f"{self.BASE}/{model_id}", json=payload,
                         headers=self._headers(), timeout=60)
        if r.status_code >= 400:
            return False, "", f"提交失败 HTTP {r.status_code}: {r.text[:400]}"
        task_id = r.json()["data"]["id"]
        log.info("%s 已提交 task=%s", model_id, task_id)

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                q = self._s.get(f"{self.BASE}/predictions/{task_id}/result",
                                headers=self._headers(), timeout=30)
            except requests.exceptions.RequestException as e:
                log.warning("音乐轮询瞬态错误(%s)", str(e)[:80])
                time.sleep(self.poll_interval_s)
                continue
            if q.status_code >= 500:
                time.sleep(self.poll_interval_s)
                continue
            if q.status_code >= 400:
                return False, task_id, f"轮询失败 HTTP {q.status_code}: {q.text[:300]}"
            data = q.json()["data"]
            st = data.get("status")
            if st == "completed":
                url = data["outputs"][0]
                a = self._s.get(url, timeout=300)
                a.raise_for_status()
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                Path(save_to).write_bytes(a.content)
                return True, task_id, ""
            if st in ("failed", "cancelled", "timeout"):
                return False, task_id, f"{st}: {str(data.get('error','unknown'))[:300]}"
            time.sleep(self.poll_interval_s)
        return False, task_id, f"音乐生成超时(>{self.timeout_s:.0f}s)"

    def text_to_music(self, prompt: str, duration_s: int,
                      save_to: str) -> Tuple[bool, str, str]:
        return self._run("sonilo/text-to-music",
                         {"prompt": prompt, "duration": int(duration_s)}, save_to)

    def video_to_music(self, video_path: str, save_to: str,
                       prompt: Optional[str] = None) -> Tuple[bool, str, str]:
        payload = {"video": self.upload(video_path)}
        if prompt:
            payload["prompt"] = prompt
        return self._run("sonilo/video-to-music", payload, save_to)
