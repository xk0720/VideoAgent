"""rl/env 精简客户端(2026-08-19 用户令:agent loop 全部重建进 rl/,
不调用文件夹以外的包)。三件套,纯 requests,只保留线协议与实跑教训:

  • TextLLM       —— OpenAI 兼容 chat(冻结 agent / 本地 vLLM 策略共用;
                      extra_body 透传关思考;content 分段列表归一化)
  • KlingClient   —— 百炼 DashScope 可灵(OSS 上传→提交→轮询→下载;
                      trust_env=False 防代理掐长轮询;>1.2MB 图先瘦身;
                      OSS 409 = 已存在 = 成功;下载失败先重取签名 URL)
  • WaveSpeedT2I  —— wavespeed flux t2i(提交→轮询→下载)

刻意不搬的装饰:能力路由、降级链、预算台账、重试梯(env 失败 = 该
候选如实记 None,判官跳过,绝不伪装成功)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests


def _content_text(c):
    """message.content 字符串/分段列表两种合法形态 → str(idealab 等
    网关返回列表;None 保持 None)。"""
    if c is None or isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p if isinstance(p, str)
                       else (p.get("text") or "")
                       for p in c if isinstance(p, (str, dict)))
    return str(c)


class CallLog:
    """env 级调用台账(一行一事件;记录失败不打断正流程)。"""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None

    def write(self, kind: str, **fields):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "kind": kind,
                                    **fields}, ensure_ascii=False,
                                   default=str) + "\n")
        except Exception:
            pass


class TextLLM:
    """OpenAI 兼容文本客户端。transient(429/5xx/断连)重试 2 次。"""

    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout: float = 600, max_tokens: int = 8192,
                 temperature: float = 0.7, extra_body: dict | None = None,
                 log: CallLog | None = None, name: str = "llm"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body if isinstance(extra_body, dict) else {}
        self.log = log or CallLog(None)
        self.name = name

    def complete(self, prompt: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        payload = {**self.extra_body,
                   "model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": int(max_tokens or self.max_tokens),
                   "temperature": float(self.temperature
                                        if temperature is None
                                        else temperature)}
        last = None
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload, timeout=self.timeout)
                if r.status_code in (429,) or r.status_code >= 500:
                    last = RuntimeError(
                        f"HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(10 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    # 4xx 是我们发错了 —— 带 body 立刻炸,不重试
                    raise RuntimeError(
                        f"LLM({self.name}/{self.model}) HTTP "
                        f"{r.status_code}: {r.text[:500]}")
                out = _content_text(
                    r.json()["choices"][0]["message"]["content"]) or ""
                self.log.write("llm", name=self.name, model=self.model,
                               chars=len(out))
                return out
            except RuntimeError:
                raise
            except Exception as exc:
                last = exc
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"LLM({self.name}) failed after retries: {last}")


class KlingClient:
    """百炼可灵(DashScope 异步任务):t2v / ref2v(refer 图)/ i2v
    (first_frame 硬钉,可与 refer 混用)。fps/seed 平台不支持,不发。"""

    BASE = "https://dashscope.aliyuncs.com/api/v1"
    MODEL_T2V = "kling/kling-v3-video-generation"
    MODEL_OMNI = "kling/kling-v3-omni-video-generation"

    def __init__(self, api_key: str, mode: str = "std",
                 aspect_ratio: str = "16:9", poll_interval: float = 15.0,
                 timeout: float = 900, log: CallLog | None = None):
        self.api_key = api_key
        self.mode = mode
        self.aspect_ratio = aspect_ratio
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.log = log or CallLog(None)
        # 本机代理会掐死长轮询(生产同款教训)—— 会话必须绕过 env 代理
        self.session = requests.Session()
        self.session.trust_env = False
        self._upload_cache: dict = {}

    @staticmethod
    def ref_token(n: int) -> str:
        return f"<<<image_{n}>>>"

    # ── OSS 上传 ──────────────────────────────────────────────────
    def _shrink(self, p: Path) -> Path:
        """>1.2MB 或长边 >1280 的图先压 JPEG(2.4MB PNG 曾把 OSS 读
        超时);PIL 缺失 → 原图直传(诚实降级)。"""
        try:
            if p.stat().st_size <= 1_200_000:
                return p
            from PIL import Image
            im = Image.open(p).convert("RGB")
            im.thumbnail((1280, 1280))
            out = p.parent / f".upload_{p.stem}.jpg"
            im.save(out, "JPEG", quality=88)
            return out
        except Exception:
            return p

    def _upload(self, path, model: str) -> str:
        s = str(path)
        if s.startswith(("http://", "https://", "oss://")):
            return s
        p = self._shrink(Path(path))
        ck = (str(p), p.stat().st_mtime, p.stat().st_size, model)
        if ck in self._upload_cache:
            return self._upload_cache[ck]
        r = self.session.get(
            f"{self.BASE}/uploads", params={"action": "getPolicy",
                                            "model": model},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"getPolicy HTTP {r.status_code}: "
                               f"{r.text[:300]}")
        pol = r.json()["data"]
        key = f"{pol['upload_dir']}/{p.name}"
        with open(p, "rb") as fh:
            up = self.session.post(
                pol["upload_host"],
                files={"OSSAccessKeyId": (None, pol["oss_access_key_id"]),
                       "Signature": (None, pol["signature"]),
                       "policy": (None, pol["policy"]),
                       "x-oss-object-acl": (None, pol["x_oss_object_acl"]),
                       "x-oss-forbid-overwrite":
                           (None, pol["x_oss_forbid_overwrite"]),
                       "key": (None, key),
                       "success_action_status": (None, "200"),
                       "file": (p.name, fh)},
                timeout=(10, 300))
        if up.status_code not in (200, 409):     # 409 = 已存在 = 成功
            raise RuntimeError(f"OSS upload HTTP {up.status_code}: "
                               f"{up.text[:300]}")
        url = f"oss://{key}"
        self._upload_cache[ck] = url
        return url

    # ── 生成 ─────────────────────────────────────────────────────
    def generate(self, prompt: str, duration, out_path,
                 first_frame=None, reference_images=None,
                 audio: bool = False) -> Path:
        model = self.MODEL_OMNI if (first_frame or reference_images) \
            else self.MODEL_T2V
        media = []
        if first_frame:
            media.append({"type": "first_frame",
                          "url": self._upload(first_frame, model)})
        for ref in (reference_images or []):
            media.append({"type": "refer",
                          "url": self._upload(ref, model)})
        params: dict = {"mode": self.mode, "audio": bool(audio)}
        if duration is not None:
            d = int(duration) if float(duration) == int(duration) \
                else int(float(duration)) + 1
            params["duration"] = max(3, min(15, d))
        if first_frame is None:
            # 有硬钉首帧时 aspect_ratio 必须省略(M0 实测),其余必填
            params["aspect_ratio"] = self.aspect_ratio
        payload = {"model": model,
                   "input": ({"prompt": prompt, "media": media}
                             if media else {"prompt": prompt}),
                   "parameters": params}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "X-DashScope-Async": "enable"}
        if media:
            headers["X-DashScope-OssResourceResolve"] = "enable"
        self.log.write("kling_submit", model=model, n_media=len(media),
                       audio=bool(audio), duration=params.get("duration"),
                       prompt=prompt[:400])
        r = self.session.post(
            f"{self.BASE}/services/aigc/video-generation/video-synthesis",
            json=payload, headers=headers, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"kling submit HTTP {r.status_code}: "
                               f"{r.text[:500]}")
        task_id = r.json()["output"]["task_id"]
        url = self._poll(task_id)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._download(url, out_path, task_id)
        self.log.write("kling_done", task_id=task_id, out=str(out_path))
        return out_path

    def _poll(self, task_id: str) -> str:
        t0 = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        while True:
            if time.time() - t0 > self.timeout:
                raise RuntimeError(f"kling task {task_id} poll timeout")
            try:
                r = self.session.get(f"{self.BASE}/tasks/{task_id}",
                                     headers=headers, timeout=60)
            except requests.RequestException:
                time.sleep(self.poll_interval)
                continue
            if r.status_code >= 500:
                time.sleep(self.poll_interval)
                continue
            out = r.json().get("output") or {}
            st = out.get("task_status")
            if st == "SUCCEEDED":
                return out["video_url"]
            if st in ("FAILED", "UNKNOWN"):
                raise RuntimeError(
                    f"kling task FAILED code={out.get('code')} "
                    f"message={out.get('message')}")
            time.sleep(self.poll_interval)

    def _download(self, url: str, out_path: Path, task_id: str):
        """任务已 SUCCEEDED = 视频存在;一次 GET 失败先重取签名 URL 再
        试,绝不把 CDN 抖动报成生成失败。"""
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=300)
                if r.status_code < 400 and r.content:
                    out_path.write_bytes(r.content)
                    return
            except Exception:
                pass
            try:
                r2 = self.session.get(
                    f"{self.BASE}/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60)
                url = (r2.json().get("output") or {}).get("video_url", url)
            except Exception:
                pass
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"kling download failed after retries: {url}")


class WaveSpeedT2I:
    """wavespeed flux t2i(肖像/背景板)。"""

    BASE = "https://api.wavespeed.ai/api/v3"
    MODEL = "wavespeed-ai/flux-kontext-pro/text-to-image"

    def __init__(self, api_key: str, aspect_ratio: str = "16:9",
                 poll_interval: float = 2.0, timeout: float = 600,
                 log: CallLog | None = None):
        self.api_key = api_key
        self.aspect_ratio = aspect_ratio
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.log = log or CallLog(None)

    def text_to_image(self, prompt: str, out_path, seed: int = 0) -> Path:
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        self.log.write("t2i_submit", prompt=prompt[:300])
        r = requests.post(
            f"{self.BASE}/{self.MODEL}",
            json={"prompt": prompt, "num_images": 1,
                  "aspect_ratio": self.aspect_ratio,
                  "guidance_scale": 3.5, "safety_tolerance": "5",
                  "seed": int(seed)},
            headers=headers, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"t2i submit HTTP {r.status_code}: "
                               f"{r.text[:300]}")
        task_id = r.json()["data"]["id"]
        t0 = time.time()
        while True:
            if time.time() - t0 > self.timeout:
                raise RuntimeError(f"t2i task {task_id} poll timeout")
            time.sleep(self.poll_interval)
            try:
                pr = requests.get(
                    f"{self.BASE}/predictions/{task_id}/result",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60)
            except requests.RequestException:
                continue
            if pr.status_code >= 500:
                continue
            data = pr.json().get("data") or {}
            st = data.get("status")
            if st == "completed":
                url = data["outputs"][0]
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img = requests.get(url, timeout=300)
                out_path.write_bytes(img.content)
                self.log.write("t2i_done", out=str(out_path))
                return out_path
            if st == "failed":
                raise RuntimeError(f"t2i FAILED: {data.get('error')}")
