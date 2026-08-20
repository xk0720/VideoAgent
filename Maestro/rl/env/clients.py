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
import threading
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
    """env 级调用台账(一行一事件;记录失败不打断正流程)。

    2026-08-20 组内并发:单行可能超过 PIPE_BUF(4096),多线程无锁
    追加会把两行绞在一起 —— 上锁,台账宁可慢一点也不能烂。"""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, kind: str, **fields):
        if self.path is None:
            return
        try:
            line = json.dumps({"ts": time.time(), "kind": kind, **fields},
                              ensure_ascii=False, default=str) + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(line)
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
                 timeout: float = 900, log: CallLog | None = None,
                 t2i=None):
        self.api_key = api_key
        self.mode = mode
        self.aspect_ratio = aspect_ratio
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.log = log or CallLog(None)
        # 生产同款接口面(window_core 的移植体按鸭子类型调用):
        self.generate_audio = False          # driver 按对白镜临时翻转
        self._t2i = t2i                      # wavespeed t2i 代理(生产同构)
        # 本机代理会掐死长轮询(生产同款教训)—— 会话必须绕过 env 代理
        self.session = requests.Session()
        self.session.trust_env = False
        self._upload_cache: dict = {}

    def clone(self) -> "KlingClient":
        """线程私有副本(2026-08-20 组内并发):generate_audio 是【实例
        属性】,driver 逐候选翻转它 —— 四个线程共用一个客户端会互相
        踩(A 候选开了音频,B 候选把它还原)。每个 worker 一个副本 =
        开关线程私有;HTTP 会话各自独立(连接池不共享更稳);上传
        缓存【共享】(同一张肖像只传一次,省钱)。"""
        c = KlingClient(self.api_key, mode=self.mode,
                        aspect_ratio=self.aspect_ratio,
                        poll_interval=self.poll_interval,
                        timeout=self.timeout, log=self.log, t2i=self._t2i)
        c._upload_cache = self._upload_cache      # 有意共享(去重)
        c.generate_audio = self.generate_audio
        return c

    def capabilities(self) -> set:
        """与生产 BailianKlingClient 同款能力申报(菜单门控依据)。"""
        caps = {"t2v", "i2v", "flf2v", "ref_images",
                "first_frame_plus_refs"}
        if self._t2i is not None:
            caps.add("t2i")
        return caps

    def text_to_image(self, prompt: str, out_path, seed: int = 0):
        if self._t2i is None:
            raise RuntimeError("no wavespeed t2i proxy configured")
        return self._t2i.text_to_image(prompt, out_path, seed=seed)

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

    # ── 生成(签名与生产 BailianKlingClient.generate 对齐:fps/seed
    # 接受但平台不支持不发;reference_video 生产同款拒绝)────────────
    def generate(self, prompt: str, duration, out_path, fps=None,
                 seed=None, first_frame=None, reference_images=None,
                 reference_video=None) -> Path:
        if reference_video is not None:
            raise RuntimeError(
                "bailian kling has no reference-video channel "
                "(production parity: raise before any upload)")
        audio = bool(getattr(self, "generate_audio", False))
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

    def frame_to_frame(self, prompt: str, first_frame, last_frame,
                       out_path, duration=None, seed=None,
                       reference_images=None) -> Path:
        """首尾双帧(flf2v):media=[first_frame, last_frame(+refer…)],
        与生产 BailianKlingClient.frame_to_frame 同协议。"""
        model = self.MODEL_OMNI
        media = [{"type": "first_frame",
                  "url": self._upload(first_frame, model)},
                 {"type": "last_frame",
                  "url": self._upload(last_frame, model)}]
        for ref in (reference_images or []):
            media.append({"type": "refer",
                          "url": self._upload(ref, model)})
        params: dict = {"mode": self.mode,
                        "audio": bool(getattr(self, "generate_audio",
                                              False))}
        if duration is not None:
            d = int(duration) if float(duration) == int(duration) \
                else int(float(duration)) + 1
            params["duration"] = max(3, min(15, d))
        payload = {"model": model,
                   "input": {"prompt": prompt, "media": media},
                   "parameters": params}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "X-DashScope-Async": "enable",
                   "X-DashScope-OssResourceResolve": "enable"}
        self.log.write("kling_flf2v", n_media=len(media),
                       duration=params.get("duration"),
                       prompt=prompt[:400])
        r = self.session.post(
            f"{self.BASE}/services/aigc/video-generation/video-synthesis",
            json=payload, headers=headers, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"kling flf2v HTTP {r.status_code}: "
                               f"{r.text[:500]}")
        task_id = r.json()["output"]["task_id"]
        url = self._poll(task_id)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._download(url, out_path, task_id)
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


class WaveSpeedImageEdit:
    """seedream v4 图像编辑(空间圣经派生视图/清场回流用)。协议与
    生产 WaveSpeedImageEditClient 同款:参考图走上传 URL(base64 会
    400),size 按被编辑图长边≈2048 等比 8 对齐(写死方图是已知 bug,
    不复制)。"""

    BASE = "https://api.wavespeed.ai/api/v3"
    MODEL = "bytedance/seedream-v4/edit"

    def __init__(self, api_key: str, poll_interval: float = 2.0,
                 timeout: float = 300, log: CallLog | None = None):
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.log = log or CallLog(None)
        self._upload_cache: dict = {}

    def _upload(self, path) -> str:
        s = str(path)
        if s.startswith(("http://", "https://")):
            return s
        p = Path(path)
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise RuntimeError(f"image-edit ref must be an image: {p}")
        ck = (str(p), p.stat().st_mtime, p.stat().st_size)
        if ck in self._upload_cache:
            return self._upload_cache[ck]
        with open(p, "rb") as fh:
            r = requests.post(
                f"{self.BASE}/media/upload/binary",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (p.name, fh)}, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"media upload HTTP {r.status_code}: "
                               f"{r.text[:300]}")
        url = r.json()["data"]["download_url"]
        self._upload_cache[ck] = url
        return url

    @staticmethod
    def _size_for(src) -> str:
        try:
            from PIL import Image
            w, h = Image.open(src).size
            if w >= h:
                W = 2048
                H = max(8, int(round(2048 * h / w / 8)) * 8)
            else:
                H = 2048
                W = max(8, int(round(2048 * w / h / 8)) * 8)
            return f"{W}*{H}"
        except Exception:
            return "2048*2048"

    def edit(self, src, prompt: str, out_path, references=None) -> Path:
        images = [self._upload(src)] + [self._upload(r)
                                        for r in (references or [])]
        self.log.write("image_edit_submit", n_refs=len(images) - 1,
                       prompt=str(prompt)[:300])
        r = requests.post(
            f"{self.BASE}/{self.MODEL}",
            json={"enable_base64_output": False,
                  "enable_sync_mode": False,
                  "images": images, "prompt": str(prompt),
                  "size": self._size_for(src)},
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"}, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"image-edit submit HTTP {r.status_code}: "
                               f"{r.text[:300]}")
        task_id = r.json()["data"]["id"]
        t0 = time.time()
        while True:
            if time.time() - t0 > self.timeout:
                raise RuntimeError(f"image-edit task {task_id} timeout")
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
                if out_path.suffix.lower() not in (".png", ".jpg",
                                                   ".jpeg", ".webp"):
                    out_path = out_path.with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img = requests.get(url, timeout=300)
                out_path.write_bytes(img.content)
                self.log.write("image_edit_done", out=str(out_path))
                return out_path
            if st == "failed":
                raise RuntimeError(
                    f"image-edit FAILED: {data.get('error')}")


class EnvVLM:
    """RL 环境的 mllm 角色(idealab Gemini,OpenAI 兼容图文 chat)。
    方法集 = 生产 IdealabGeminiVLM 在管线里【实际被用到】的面:
    caption_image(指令原文与生产 OpenAICompatVLM.caption_image 逐字
    一致)。接点实况/空间圣经图注在生产 idealab 配置下都会落到它。"""

    _CAPTION_INSTRUCTION = (
        "Describe this image in ONE short sentence for retrieval: "
        "what/who it shows and the setting. Also start with one "
        "category word from [background, character, object, style] "
        "and a colon. Example: 'background: a cozy living room at "
        "night with a lit fireplace'. No other text.")

    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout: float = 300, log: CallLog | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.log = log or CallLog(None)

    def _chat_image(self, image_path, text: str) -> str:
        import base64 as _b64
        p = Path(str(image_path))
        if not p.exists() or p.suffix.lower() not in (
                ".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            return ""
        try:
            b64 = _b64.b64encode(p.read_bytes()).decode()
        except OSError:
            return ""
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{
                    "role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": text}]}]},
                timeout=self.timeout)
            if r.status_code >= 400:
                self.log.write("vlm_caption_http", code=r.status_code,
                               body=r.text[:200])
                return ""
            return (_content_text(
                r.json()["choices"][0]["message"]["content"])
                or "").strip()
        except Exception as exc:
            self.log.write("vlm_caption_error", error=str(exc)[:150])
            return ""

    def caption_image(self, image_path) -> str:
        return self._chat_image(image_path, self._CAPTION_INSTRUCTION)
