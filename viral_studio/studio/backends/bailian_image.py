"""百炼图像生成 —— 背景图等前置素材(文生图, 可挂参考图)。

与 kling/animate 同源(同一把 DASHSCOPE key、同一套 async 提交+轮询协议), 所以
不引第三方。默认 wan2.5-t2i-preview; 需要多图参考时切 wan2.7-image-pro。

  提交 POST {BASE}/services/aigc/text2image/image-synthesis
       头 Authorization + X-DashScope-Async: enable
  轮询 GET  {BASE}/tasks/{task_id} → SUCCEEDED → output.results[0].url
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

log = logging.getLogger("viral_studio")


class BailianImageClient:
    BASE = "https://dashscope.aliyuncs.com/api/v1"
    SYNTH = BASE + "/services/aigc/text2image/image-synthesis"
    EDIT = BASE + "/services/aigc/image2image/image-synthesis"

    def __init__(self, api_key: str, model: str = "wan2.5-t2i-preview",
                 edit_model: str = "wanx2.1-imageedit",
                 poll_interval_s: float = 5.0, timeout_s: float = 300.0):
        self.edit_model = edit_model
        self._cache: dict = {}
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        s = requests.Session()
        s.trust_env = False
        self._s = s

    def _headers(self, async_: bool = False, oss: bool = False) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if async_:
            h["X-DashScope-Async"] = "enable"
        if oss:
            h["X-DashScope-OssResourceResolve"] = "enable"
        return h

    def upload(self, path: str, model: str) -> str:
        """oss 直传(与 animate 后端同一套 getPolicy 协议), 供图像编辑挂底图。"""
        p = Path(path).resolve()
        key = (str(p), p.stat().st_mtime_ns, model)
        if key in self._cache:
            return self._cache[key]
        r = self._s.get(f"{self.BASE}/uploads",
                        params={"action": "getPolicy", "model": model},
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

    def _poll(self, task_id: str) -> Tuple[bool, str, str]:
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
                res = out.get("results") or []
                url = res[0].get("url") if res else None
                return (True, url, "") if url else (False, "", f"SUCCEEDED 但无 url: {out}")
            if st in ("FAILED", "CANCELED", "UNKNOWN"):
                return False, "", f"{st}: code={out.get('code')} message={out.get('message')}"
            time.sleep(self.poll_interval_s)
        return False, "", f"轮询超时(>{self.timeout_s:.0f}s)"

    def edit(self, base_image: str, prompt: str, save_to: str,
             function: str = "description_edit") -> Tuple[bool, str, str]:
        """指令式图像编辑: 底图不变的部分尽量保留(人物/衣服), 按 prompt 改背景等。

        用于参考图换背景 —— 纯文生图锚不住人物与商品, 编辑模型以底图为准。"""
        url = self.upload(base_image, self.edit_model)
        payload = {"model": self.edit_model,
                   "input": {"function": function, "prompt": prompt,
                             "base_image_url": url},
                   "parameters": {"n": 1}}
        r = self._s.post(self.EDIT, headers=self._headers(async_=True, oss=True),
                         json=payload, timeout=60)
        body = r.json() if r.text else {}
        task_id = (body.get("output") or {}).get("task_id", "")
        if r.status_code != 200 or not task_id:
            return False, "", (f"提交失败 HTTP {r.status_code}: "
                               f"{json.dumps(body, ensure_ascii=False)[:400]}")
        log.info("image-edit 已提交 task=%s model=%s", task_id, self.edit_model)
        ok, img_url, err = self._poll(task_id)
        if not ok:
            return False, task_id, err
        im = self._s.get(img_url, timeout=300)
        im.raise_for_status()
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        Path(save_to).write_bytes(im.content)
        return True, task_id, ""

    def generate(self, prompt: str, save_to: str, size: str = "720*1280",
                 n: int = 1, negative_prompt: Optional[str] = None
                 ) -> Tuple[bool, str, str]:
        """文生图。size 用 '宽*高'(百炼格式); 传入 '720x1280' 会自动转。"""
        size = size.replace("x", "*")
        payload = {"model": self.model,
                   "input": {"prompt": prompt},
                   "parameters": {"size": size, "n": n}}
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt

        r = self._s.post(self.SYNTH, headers=self._headers(async_=True),
                         json=payload, timeout=60)
        body = r.json() if r.text else {}
        task_id = (body.get("output") or {}).get("task_id", "")
        if r.status_code != 200 or not task_id:
            return False, "", (f"提交失败 HTTP {r.status_code}: "
                               f"{json.dumps(body, ensure_ascii=False)[:400]}")
        log.info("image 已提交 task=%s model=%s size=%s", task_id, self.model, size)

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
                res = out.get("results") or []
                url = res[0].get("url") if res else None
                if not url:
                    return False, task_id, f"SUCCEEDED 但无 url: {out}"
                im = self._s.get(url, timeout=300)
                im.raise_for_status()
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                Path(save_to).write_bytes(im.content)
                return True, task_id, ""
            if st in ("FAILED", "CANCELED", "UNKNOWN"):
                return False, task_id, f"{st}: code={out.get('code')} message={out.get('message')}"
            time.sleep(self.poll_interval_s)
        return False, task_id, f"轮询超时(>{self.timeout_s:.0f}s)"
