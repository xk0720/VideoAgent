"""MiniMax TTS via WaveSpeed —— 旁白解说(不要求音画同步的段落用它)。

协议 copy 自本仓已验证实现(Maestro audio_gen_backends.WaveSpeedAudioClient,
其本身 ported 自 UniVA 已验证的 speech_gen):
  提交 POST {BASE}/minimax/speech-2.6-hd
       payload = {text, voice_id, emotion, speed, pitch, volume,
                  english_normalization}
  轮询 GET  {BASE}/predictions/{id}/result → completed → outputs[0] 下载

相对 seedance 音画同出的好处: 音色/语速可控, 文本 100% 准确(不会像生成模型
那样把中文写错), 且与画面解耦——画面可以完全放开动作。
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

log = logging.getLogger("viral_studio")

# 年轻活泼向音色(用户裁决 2026-08-16); 完整列表见 MiniMax 文档
VOICE_PRESETS = {
    "lively_girl": "Lively_Girl",
    "sweet_girl": "Sweet_Girl_2",
    "young_woman": "Young_Knight",       # 备选, 若前两个不可用
    "wise_woman": "Wise_Woman",          # 后端默认(沉稳), 作兜底
}


class MiniMaxTTSClient:
    BASE = "https://api.wavespeed.ai/api/v3"

    def __init__(self, api_key: str, model_id: str = "minimax/speech-2.6-hd",
                 voice_id: str = "Lively_Girl", poll_interval_s: float = 2.0,
                 timeout_s: float = 300.0):
        if not api_key:
            raise RuntimeError("缺少 WAVESPEED_API_KEY")
        self.api_key = api_key
        self.model_id = model_id
        self.voice_id = voice_id
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        s = requests.Session()
        s.trust_env = False
        self._s = s

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def speak(self, text: str, save_to: str, voice_id: Optional[str] = None,
              speed: float = 1.0, emotion: str = "happy") -> Tuple[bool, str, str]:
        """文本 → 语音文件。返回 (ok, task_id, err)。"""
        payload = {
            "emotion": emotion,
            "english_normalization": False,
            "pitch": 0,
            "speed": speed,
            "text": text,
            "voice_id": voice_id or self.voice_id,
            "volume": 1,
        }
        r = self._s.post(f"{self.BASE}/{self.model_id}", json=payload,
                         headers=self._headers(), timeout=60)
        if r.status_code >= 400:
            return False, "", f"提交失败 HTTP {r.status_code}: {r.text[:400]}"
        task_id = r.json()["data"]["id"]

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                q = self._s.get(f"{self.BASE}/predictions/{task_id}/result",
                                headers=self._headers(), timeout=30)
            except requests.exceptions.RequestException as e:
                log.warning("TTS 轮询瞬态错误(%s)", str(e)[:80])
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
            if st == "failed":
                return False, task_id, f"failed: {str(data.get('error','unknown'))[:300]}"
            time.sleep(self.poll_interval_s)
        return False, task_id, f"TTS 超时(>{self.timeout_s:.0f}s)"
