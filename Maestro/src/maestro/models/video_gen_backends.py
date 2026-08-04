"""Real video-generation backends (v0.4).

These implement the SAME `BaseVideoGenClient.generate(...)` contract as the
mock, so the rest of the data flow is unchanged — flip
`models.video_gen.name` in the config and (for local backends) fill the TODO
bodies.

Conditioning contract (v0.4): prompt + first_frame (C2 keyframe anchor) +
reference_images (E1 identity/style). There is NO physics control signal —
the sketch-as-controller line is dead; physics is VERIFIED from generated
pixels (physics/), never injected into the generator.

Backends:
  • WaveSpeedClient — hosted REST API (submit → poll → download), the same
    service UniVA (2511.08521) uses for all its generation. This one is FULLY
    IMPLEMENTED: with $WAVESPEED_API_KEY set, `maestro` produces real pixels
    with no local GPU — the fastest route to the minimal real chain.
  • BailianKlingClient — 阿里云百炼(DashScope)可灵 kling-v3,
    dev-music-bailian 主后端(t2v/i2v/flf2v/refer,含 first_frame+refer 混用;
    t2i / 音乐按 2026-08-03 裁决经内部 WaveSpeed 代理)。
  • OmniWeavingClient / WanClient — local-weights skeletons (fill + GPU).
  • VeoApiClient — hosted API skeleton.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from .video_gen import BaseVideoGenClient, MockVideoGenClient

log = get_logger(__name__)

_UPLOAD_URL = "https://api.wavespeed.ai/api/v3/media/upload/binary"


def upload_media(api_key: str, path) -> str:
    """Upload a local image/video/audio to WaveSpeed's official media endpoint
    (multipart `file`, ≤300 MB) and return its public URL. http(s) inputs pass
    through. The 2026 models take media by URL — runwayml/gen4-aleph outright
    400s on base64 data URIs. Shared by the video AND audio clients."""
    import requests  # std in our [all] extras; loud ImportError otherwise

    vp = str(path)
    if vp.startswith("http://") or vp.startswith("https://"):
        return vp
    p = Path(vp)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"media not found: {vp}")
    with p.open("rb") as fh:
        resp = requests.post(
            _UPLOAD_URL, headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (p.name, fh)}, timeout=300,
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"WaveSpeed media upload of '{p.name}' failed: HTTP "
            f"{resp.status_code} — {resp.text[:2000]}"
        )
    return resp.json()["data"]["download_url"]


# ─────────────────────────────────────────────────────────────
# WaveSpeed — hosted API backend (no local GPU; UniVA's route)
# ─────────────────────────────────────────────────────────────
class WaveSpeedClient(BaseVideoGenClient):
    """WaveSpeed REST API (https://wavespeed.ai). Protocol (verified against the
    official docs 2026-07, see docs/research/wavespeed_api_reference_2026_07.md):
    POST {BASE}/{model-id} → poll predictions/{id}/result → download outputs[0].

    Defaults target the BEST-quality family on WaveSpeed (cost no object, per
    the user's ruling): bytedance/seedance-2.0 — t2v/i2v (native first+last
    frame via `last_image`), video-edit, video-extend; duration any int 4–15 s;
    resolution up to 4k. Legacy seedance-v1 / wan-flf2v ids keep their old
    schemas ({5,10} s enums) — the payload builder switches per family.

    Media inputs (images/videos) are UPLOADED to the official media endpoint
    and passed by URL: the 2026 models document URL/upload input, and
    runwayml/gen4-aleph outright 400s on base64 data URIs.

    config:
      models.video_gen:
        name: "wavespeed"
        model_id: "bytedance/seedance-2.0/text-to-video"   # i2v id derived
        resolution: "480p"         # 测试期默认最低价;产线再升 720p/1080p/4k
        generate_audio: false      # seedance-2.0 native AV co-generation
        flf2v_model:  "bytedance/seedance-2.0/image-to-video"  # or wavespeed-ai/wan-flf2v
        edit_model:   "bytedance/seedance-2.0/video-edit"
        extend_model: "bytedance/seedance-2.0/video-extend"
        api_key: ...               # or $WAVESPEED_API_KEY
        poll_interval: 2.0
        timeout: 600
        # escape hatches: allowed_durations: [5,10] | duration_range: [4,15]
        # extra_params: {seed: 42, camera_fixed: true, ...}
    """

    BASE = "https://api.wavespeed.ai/api/v3"
    UPLOAD_URL = BASE + "/media/upload/binary"

    def __init__(self, name: str = "wavespeed", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("WAVESPEED_API_KEY")
        self.model_id = self.config.get("model_id", "bytedance/seedance-2.0/text-to-video")
        self.resolution = self.config.get("resolution", "480p")
        self.generate_audio = bool(self.config.get("generate_audio", False))
        self.flf2v_model = self.config.get(
            "flf2v_model", "bytedance/seedance-2.0/image-to-video")
        self.edit_model = self.config.get(
            "edit_model", "bytedance/seedance-2.0/video-edit")
        self.extend_model = self.config.get(
            "extend_model", "bytedance/seedance-2.0/video-extend")
        self.poll_interval = float(self.config.get("poll_interval", 2.0))
        self.timeout = float(self.config.get("timeout", 600))
        self._upload_cache: dict = {}

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "WaveSpeedClient needs an API key: set $WAVESPEED_API_KEY or "
                "models.video_gen.api_key (or switch back to 'mock-video-gen')."
            )
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    @staticmethod
    def _is_range_family(model_id: str) -> bool:
        """seedance-2.0 family takes ANY int duration in 4–15 s; legacy models
        (seedance v1, wan-flf2v) take a {5, 10} enum."""
        return "seedance-2.0" in model_id

    @staticmethod
    def _i2v_variant(model_id: str) -> str:
        """Derive the i2v id from a t2v id — both naming generations."""
        return (model_id
                .replace("-t2v-", "-i2v-")                    # seedance v1 style
                .replace("/text-to-video", "/image-to-video"))  # 2026 3-segment style

    def _snap_duration(self, seconds: float, model_id: Optional[str] = None) -> int:
        """WaveSpeed models constrain `duration` (seconds) and 400 on anything
        else: seedance-2.0 = any int in [4, 15]; legacy = exactly 5 or 10.
        Snap the requested seconds UP into the valid set; callers that need a
        shorter span retime the surplus off (pipeline.timeline._fit_to_seconds)."""
        mid = model_id or self.model_id
        want = int(seconds) if float(seconds) == int(seconds) else int(seconds) + 1
        rng = self.config.get("duration_range") or (
            (4, 15) if self._is_range_family(mid) else None)
        if rng:
            lo, hi = int(rng[0]), int(rng[1])
            return max(lo, min(hi, want))
        allowed = sorted(int(a) for a in self.config.get("allowed_durations",
                                                         (5, 10)))
        for a in allowed:
            if want <= a:
                return a
        return allowed[-1]

    def _upload_media(self, path) -> str:
        """`upload_media` with a loud key check (BEFORE any file IO / network)
        and a per-client cache keyed on (path, mtime, size) so a frame or clip
        reused across repair turns uploads once."""
        self._headers()                              # loud RuntimeError sans key
        vp = str(path)
        if vp.startswith("http://") or vp.startswith("https://"):
            return vp
        p = Path(vp)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"media not found: {vp}")
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
        if key not in self._upload_cache:
            self._upload_cache[key] = upload_media(self.api_key, p)
        return self._upload_cache[key]

    def generate(
        self,
        prompt: str,
        duration: Optional[float],
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
        reference_video: Optional[Path] = None,
    ) -> Path:
        """`reference_video` (seedance-2.0 only) rides the `reference_videos`
        channel — an independent MOTION condition (e.g. a physics-sim reference
        clip, NEWTON-style) that combines with the text prompt. Only offered to
        callers via the "ref_video" capability; a legacy model id raises."""
        # 映射表铁律(docs/CONDITION_MODEL_MAP.md §1):reference_videos 只存在
        # 于 text-to-video 端点;first_frame 会把调用切到 image-to-video(其
        # schema 只有 image+last_image)。这个组合没有已验证的调用方式,丢掉
        # 视频条件又等于偷换策略 —— 所以在任何上传发生之前直接拒绝,调用方
        # 应改用 t2v + reference_images(软锚)或放弃视频条件。
        if reference_video is not None and first_frame is not None:
            raise RuntimeError(
                "generate(): first_frame + reference_video is not a verified "
                "WaveSpeed schema (reference_videos exists on text-to-video "
                "only; image-to-video takes image+last_image). Pass the image "
                "via reference_images on the t2v route instead "
                "(docs/CONDITION_MODEL_MAP.md §1)."
            )
        model_id = self.model_id
        is_i2v = first_frame is not None and (
            str(first_frame).startswith(("http://", "https://"))
            or Path(first_frame).exists())
        if is_i2v:
            model_id = self._i2v_variant(model_id)

        # duration=None = brain 没规划 → 不传字段,API 用模型自然默认
        # (2026-07-14 裁决);有值(brain 规划的 [4,10])→ snap 进模型时长域。
        payload: dict = {"prompt": prompt}
        if duration is not None:
            payload["duration"] = self._snap_duration(duration, model_id)
        if self._is_range_family(model_id):
            payload["resolution"] = self.resolution
            payload["generate_audio"] = self.generate_audio
        else:
            payload["seed"] = seed          # legacy schema (UniVA-verified)
        if is_i2v:
            payload["image"] = self._upload_media(first_frame)
        else:
            # t2v only — in i2v the image defines the ratio
            payload["aspect_ratio"] = self.config.get("aspect_ratio", "16:9")
        if reference_video is not None:
            if not self._is_range_family(model_id):
                raise RuntimeError(
                    f"model '{model_id}' has no reference_videos channel — "
                    "reference-video conditioning needs the seedance-2.0 family")
            vids = (list(reference_video)
                    if isinstance(reference_video, (list, tuple))
                    else [reference_video])
            if len(vids) > 3:
                log.info("reference_videos capped to 3 (got %d) — "
                         "seedance-2.0 documented limit", len(vids))
            payload["reference_videos"] = [self._upload_media(v)
                                           for v in vids[:3]]
        if reference_images:
            # seedance-2.0 reference_images channel. VERIFIED (2026-07, official
            # model pages) ONLY on the TEXT-to-video endpoint: ≤9 images,
            # @Image1…@ImageN prompt mentions, combinable with reference_videos
            # (12 files total). The i2v endpoint's schema lists ONLY
            # image+last_image (its README hints at refs but the schema does
            # not expose the field — UNVERIFIED, so we do NOT hardcode it:
            # docs/research/wavespeed_multi_image_2026_07.md §4).
            if not self._is_range_family(model_id):
                raise RuntimeError(
                    f"model '{model_id}' has no reference_images channel — "
                    "multi-image conditioning needs the seedance-2.0 family "
                    "(or use multi_image_to_video for the Kling route)")
            if is_i2v:
                # Honest degradation, loudly logged: refs are dropped rather
                # than posted against an unverified i2v schema. Callers that
                # NEED refs use the t2v route (no first_frame) or
                # multi_image_to_video.
                log.warning(
                    "reference_images dropped: the seedance-2.0 i2v schema "
                    "exposes only image+last_image (refs verified on t2v only)"
                )
            else:
                refs = list(reference_images)[:9]
                if len(reference_images) > 9:
                    log.info("reference_images capped to 9 (got %d) — "
                             "seedance-2.0 documented limit",
                             len(reference_images))
                payload["reference_images"] = [self._upload_media(p)
                                               for p in refs]
        payload.update(self.config.get("extra_params") or {})

        return self._run_task(model_id, payload, out_path)

    def multi_image_to_video(
        self,
        prompt: str,
        images: list,
        out_path: Path,
        duration: Optional[float] = None,
        seed: int = 0,
        video: Optional[Path] = None,
    ) -> Path:
        """Multi-image REFERENCE-to-video — capability "multi_i2v". One request
        takes an `images` ARRAY and composes ONE video consistent with all of
        them (identity/scene consistency; the prompt drives the motion), e.g.
        [previous shot's last frame, this shot's keyframe, identity anchor].

        Default route kwaivgi/kling-video-o1/reference-to-video (docs-verified
        2026-07: ≤7 images, or ≤4 when a reference `video` is ALSO passed —
        "If the input reference parameters include a video, then the number of
        reference images … will be reduced to 4"). The legacy
        kwaivgi/kling-v1.6-multi-i2v-standard schema (≤4 images, no video) is
        kept behind `multi_i2v_model`. Distinct from `reference_images`
        (steering refs on the seedance t2v route): here the images ARE the
        condition, with no designated first frame."""
        if not images:
            raise ValueError("multi_image_to_video needs at least one image")
        model = self.config.get("multi_i2v_model",
                                "kwaivgi/kling-video-o1/reference-to-video")
        if "kling-v1.6" in model:                       # legacy schema
            imgs = list(images)[:4]
            if len(images) > 4:
                log.info("multi_image_to_video: images capped to 4 (got %d) — "
                         "kling-v1.6 documented limit", len(images))
            payload = {
                "prompt": prompt,
                "images": [self._upload_media(p) for p in imgs],
                "aspect_ratio": self.config.get("aspect_ratio", "16:9"),
            }
            if duration is not None:                    # None → API 默认
                payload["duration"] = self._snap_duration(duration, model)
            return self._run_task(model, payload, out_path)
        cap = 4 if video is not None else 7
        imgs = list(images)[:cap]
        if len(images) > cap:
            log.info("multi_image_to_video: images capped to %d (got %d) — "
                     "kling-video-o1 documented limit%s", cap, len(images),
                     " with a reference video" if video is not None else "")
        payload = {
            "prompt": prompt,
            "images": [self._upload_media(p) for p in imgs],
            "aspect_ratio": self.config.get("aspect_ratio", "16:9"),
            "keep_original_sound": False,
        }
        if duration is not None:                        # {5,10} 枚举,向上 snap
            payload["duration"] = self._snap_duration(duration, model)
        if video is not None:
            payload["video"] = self._upload_media(video)
        return self._run_task(model, payload, out_path)

    # 语义文本字段:审计调用日志时必须全文可读(prompt 被缩写成 "<446 chars>"
    # 的日志毫无审计价值 —— 2026-07-15 attempt_1 现场教训)。上限 4000 只防
    # 极端脏数据;真正要缩写的是 base64/超长 URL 之类的非语义大块头。
    _TEXT_FIELDS = ("prompt", "hint", "edit_instruction", "negative_prompt")

    @classmethod
    def _summarize_payload(cls, payload: dict) -> dict:
        """Payload with blobs shortened — semantic text stays readable."""
        def _one(v, keep_text=False):
            if isinstance(v, str):
                cap = 4000 if keep_text else 200
                return v if len(v) <= cap else f"{v[:cap]}…<{len(v)} chars>"
            if isinstance(v, list):
                return [_one(x) for x in v]
            return v
        return {k: _one(v, keep_text=k in cls._TEXT_FIELDS)
                for k, v in payload.items()}

    def _log_call(self, record: dict) -> None:
        """任务 0(2026-07-14):每次 WaveSpeed 调用可核对——模型名 + 参数打到
        终端 INFO,并逐行追加到 config `call_log` 指定的 JSONL(测试脚本默认
        接到 <out_dir>/wavespeed_calls.jsonl)。日志失败绝不影响调用本身。"""
        try:
            log.info("wavespeed %s → %s %s", record.get("event"),
                     record.get("model"),
                     json.dumps({k: v for k, v in record.items()
                                 if k not in ("event", "model", "ts")},
                                ensure_ascii=False, default=str)[:1500])
        except Exception:
            pass
        path = self.config.get("call_log")
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str)
                        + "\n")
        except OSError as exc:
            log.warning("call_log write failed (%s): %s", path, exc)

    # ── shared submit → poll predictions/{id}/result → download (UniVA protocol) ──
    def _run_task(self, model_id: str, payload: dict, out_path: Path) -> Path:
        import requests  # std in our [all] extras; loud ImportError otherwise

        t0 = time.time()
        self._log_call({"ts": t0, "event": "submit", "model": model_id,
                        "payload": self._summarize_payload(payload),
                        "out": str(out_path)})
        resp = requests.post(
            f"{self.BASE}/{model_id}", json=payload, headers=self._headers(),
            timeout=60,
        )
        if resp.status_code >= 400:
            self._log_call({"ts": time.time(), "event": "failed",
                            "model": model_id,
                            "error": f"HTTP {resp.status_code}: "
                                     f"{resp.text[:500]}"})
            # Surface the API's own explanation — raise_for_status() drops the
            # response body, which is where WaveSpeed says WHICH field is wrong.
            raise RuntimeError(
                f"WaveSpeed submit to '{model_id}' failed: HTTP "
                f"{resp.status_code} — {resp.text[:2000]}\n"
                f"payload: {self._summarize_payload(payload)}"
            )
        task_id = resp.json()["data"]["id"]

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = requests.get(
                f"{self.BASE}/predictions/{task_id}/result",
                headers=self._headers(), timeout=30,
            )
            if r.status_code >= 500:          # transient server hiccup → re-poll
                time.sleep(self.poll_interval)
                continue
            if r.status_code >= 400:
                raise RuntimeError(
                    f"WaveSpeed poll for task {task_id} ('{model_id}') failed: "
                    f"HTTP {r.status_code} — {r.text[:2000]}"
                )
            data = r.json()["data"]
            status = data.get("status")
            if status == "completed":
                url = data["outputs"][0]
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                video = requests.get(url, timeout=120)
                video.raise_for_status()
                out_path.write_bytes(video.content)
                self._log_call({"ts": time.time(), "event": "completed",
                                "model": model_id, "task_id": task_id,
                                "elapsed_s": round(time.time() - t0, 1),
                                "out": str(out_path)})
                return out_path
            if status == "failed":
                self._log_call({"ts": time.time(), "event": "failed",
                                "model": model_id, "task_id": task_id,
                                "error": str(data.get("error",
                                                      "unknown error"))[:500]})
                raise RuntimeError(f"WaveSpeed task {task_id} failed: "
                                   f"{data.get('error', 'unknown error')}")
            time.sleep(self.poll_interval)
        self._log_call({"ts": time.time(), "event": "failed",
                        "model": model_id, "task_id": task_id,
                        "error": f"timeout after {self.timeout}s"})
        raise TimeoutError(f"WaveSpeed task {task_id} did not finish within "
                           f"{self.timeout}s")

    def text_to_image(
        self,
        prompt: str,
        out_path: Path,
        seed: int = 0,
    ) -> Path:
        """Text → ONE still image (capability "t2i"). Used by the window loop's
        keyframe stage (skills/brain_skills/window_generation.md): a shot's
        keyframe can be t2i-generated from its description. Route ported from
        UniVA `text_to_image_generate` (flux-kontext-pro), schema live-verified
        by UniVA's own runs; model id configurable via config `t2i_model`."""
        model = self.config.get(
            "t2i_model", "wavespeed-ai/flux-kontext-pro/text-to-image")
        payload = {
            "prompt": prompt,
            "num_images": 1,
            "aspect_ratio": self.config.get("aspect_ratio", "16:9"),
            "guidance_scale": 3.5,
            "safety_tolerance": "5",
            "seed": seed,
        }
        return self._run_task(model, payload, out_path)

    def frame_to_frame(
        self,
        prompt: str,
        first_frame: Path,
        last_frame: Path,
        out_path: Path,
        duration: Optional[float] = None,
        seed: int = 0,
    ) -> Path:
        """First+last-frame video. Capability "flf2v" (optional; see
        BaseVideoGenClient.capabilities). Two routes by `flf2v_model`:
          • default `bytedance/seedance-2.0/image-to-video` — the best-quality
            first+last model on WaveSpeed (i2v with a native `last_image`);
          • legacy `wavespeed-ai/wan-flf2v` — UniVA's route, old schema.
        Both frames are uploaded and passed by URL."""
        model = self.flf2v_model
        first_url = self._upload_media(first_frame)
        last_url = self._upload_media(last_frame)
        if "wan-flf2v" in model:
            payload = {
                "enable_safety_checker": True,
                "first_image": first_url,
                "guidance_scale": 5,
                "last_image": last_url,
                "negative_prompt": "",
                "num_inference_steps": 30,
                "prompt": prompt,
                "seed": seed,
                "size": "832*480",
            }
        else:
            payload = {
                "prompt": prompt,
                "image": first_url,
                "last_image": last_url,
                "resolution": self.resolution,
                "generate_audio": self.generate_audio,
            }
        if duration is not None:                        # None → API 默认
            payload["duration"] = self._snap_duration(duration, model)
        return self._run_task(model, payload, out_path)

    def edit_video(
        self,
        prompt: str,
        video_path: Path,
        out_path: Path,
        backend: str = "seedance",
        task: str = "depth",
        seed: int = 0,
    ) -> Path:
        """Edit existing footage. Capability "edit" (optional). Three routes —
        the input video is UPLOADED and passed by URL (gen4-aleph documents
        URL-only input; base64 data URIs 400):
          • backend="seedance" → bytedance/seedance-2.0/video-edit (default —
            best-quality prompt edit on WaveSpeed; ≤15 s inputs)
          • backend="runway"   → runwayml/gen4-aleph (UniVA's free-form route)
          • backend="vace"     → wavespeed-ai/wan-2.1-14b-vace (structure-guided;
            `task` is depth/pose/etc. — no wan-2.2 vace exists on WaveSpeed)
        """
        if backend not in ("seedance", "runway", "vace"):
            raise ValueError(
                f"edit_video backend must be 'seedance', 'runway' or 'vace', "
                f"got '{backend}'")
        video_url = self._upload_media(video_path)
        if backend == "seedance":
            payload = {
                "prompt": prompt,
                "video": video_url,
                "resolution": self.resolution,
                "generate_audio": False,   # keep the input audio track
            }
            return self._run_task(self.edit_model, payload, out_path)
        if backend == "runway":
            payload = {
                "aspect_ratio": "16:9",
                "prompt": prompt,
                "video": video_url,
            }
            return self._run_task("runwayml/gen4-aleph", payload, out_path)
        payload = {
            "context_scale": 1,
            "duration": 5,
            "flow_shift": 16,
            "guidance_scale": 5,
            "images": [],              # docs: empty array is the default → accepted
            "negative_prompt": "",
            "num_inference_steps": 40,
            "prompt": prompt,
            "seed": seed,
            "size": "1280*720",
            "task": task,
            "video": video_url,
        }
        return self._run_task("wavespeed-ai/wan-2.1-14b-vace", payload, out_path)

    # ── widened atom palette (v0.4) — thin REAL maps to edit_video tasks,
    #    ported from UniVA's vace_api / runway_video_editing. Each is loud
    #    without a key (edit_video → _upload_media → _headers). ──
    def depth_modify(self, prompt: str, video_path: Path, out_path: Path,
                     seed: int = 0) -> Path:
        """Depth-guided foreground/background edit (UniVA `depth_modify`):
        edit_video(backend='vace', task='depth'). Suitable for a defect that
        is about the BACKGROUND or FOREGROUND being wrong while the rest of the
        content should be preserved — the structure-guided VACE route keeps
        motion/layout and re-renders by depth."""
        return self.edit_video(prompt=prompt, video_path=video_path,
                               out_path=out_path, backend="vace", task="depth",
                               seed=seed)

    def style_transfer(self, prompt: str, video_path: Path, out_path: Path,
                       seed: int = 0) -> Path:
        """Artistic style transfer (UniVA `style_transfer`): a style-framed
        prompt through the free-form runway route (edit_video backend='runway').
        Suitable for a STYLE defect — wrong palette/texture/look while content
        and motion stay. We frame the prompt so the edit reads as a style render."""
        styled = (f"Restyle this video in the following artistic style, keeping "
                  f"the same content and motion: {prompt}")
        return self.edit_video(prompt=styled, video_path=video_path,
                               out_path=out_path, backend="runway", seed=seed)

    def repaint(self, prompt: str, video_path: Path, label: str,
                out_path: Path, seed: int = 0) -> Path:
        """Mask-based object replace/inpaint (UniVA `repainting`). This needs a
        video segmentation backend (Sa2VA/SAM, GPU) to produce the per-object
        mask from `label` BEFORE any edit — that is not wired here, so this is an
        HONEST skeleton that raises rather than faking a mask. Use edit_clip /
        depth_modify for non-mask edits in the meantime."""
        raise RuntimeError(
            "repaint needs a segmentation backend (Sa2VA/SAM) — not wired; "
            "use edit_clip/depth_modify"
        )

    def extend(
        self,
        prompt: str,
        video_path: Path,
        out_path: Path,
        duration: Optional[float] = None,
        seed: int = 0,
        last_image: Optional[Path] = None,
    ) -> Path:
        """Continue an existing clip beyond its last frame. Capability "extend"
        (optional). Default `bytedance/seedance-2.0/video-extend` — schema
        re-verified 2026-07-16(官方模型页):
          prompt(必填,描述接下来的延续)+ video(必填 URL,从【末帧】继续)
          + last_image(可选 URL,目标尾帧 —— 模型从输入末帧插值到这张图)
          + duration(int 4-15,默认 5)+ resolution + generate_audio
          (官方默认 true,这里跟随 config 默认 False 显式关闭)。
        ⚠ 输出 = 原片 + 续段【拼接】(计费只算新段)—— 调用方要裁掉头部
        原片时长才是纯续段(窗口策略 extend_prev 负责裁)。"""
        payload = {
            "prompt": prompt,
            "video": self._upload_media(video_path),   # loud key check inside
            "resolution": self.resolution,
            "generate_audio": self.generate_audio,
        }
        if last_image is not None:                      # 目标尾帧(可选)
            payload["last_image"] = self._upload_media(last_image)
        if duration is not None:                        # None → API 默认
            payload["duration"] = self._snap_duration(duration,
                                                      self.extend_model)
        return self._run_task(self.extend_model, payload, out_path)

    def supported_conditions(self) -> set[str]:
        return {"first_frame"}

    def ref_token(self, n: int) -> str:
        """本后端 prompt 里第 N 张参考图的引用方言:@ImageN(seedance-2.0 /
        kling-o1 官方 prompt 语法)。窗口管线写引用句时问后端要方言,不再
        硬编码 —— 百炼可灵是 <<<image_N>>>(BailianKlingClient.ref_token)。"""
        return f"@Image{n}"

    def capabilities(self) -> set[str]:
        # Phase-2 routing seed: t2v/i2v via generate(), plus the optional
        # frame_to_frame (flf2v), edit_video (edit), and extend (extend) methods.
        # v0.4 widened atom palette: "depth" (depth_modify → vace/depth) and
        # "style" (style_transfer → runway) are real edit_video routes exposed as
        # brain actions; both sit under the same edit-capable backend.
        caps = {"t2v", "i2v", "flf2v", "edit", "extend", "depth", "style",
                "t2i"}   # t2i: text_to_image (keyframe stage of the window loop)
        # "ref_video": generate(reference_video=...) — the seedance-2.0
        # reference_videos channel (motion conditioning, e.g. a physics-sim
        # reference clip) + reference_images channel (≤9 steering images with
        # @ImageN prompt mentions). Legacy v1 model ids have neither.
        if self._is_range_family(self.model_id):
            caps.add("ref_video")
            caps.add("ref_images")
        # multi-image FUSION i2v (images array ≤4) — the Kling multi-i2v route,
        # always addressable on this client (own model id per call).
        caps.add("multi_i2v")
        return caps


# ─────────────────────────────────────────────────────────────
# 百炼可灵 — DashScope hosted API(dev-music-bailian 主后端)
# ─────────────────────────────────────────────────────────────
class BailianKlingClient(BaseVideoGenClient):
    """阿里云百炼(DashScope)可灵 kling-v3 视频后端。协议 = M0 真 API 验证
    (scripts/playground/bailian_kling_probe.py,2026-08-03,五形态全
    SUCCEEDED),照实现,不改契约:

      提交  POST {BASE}/services/aigc/video-generation/video-synthesis
            头 Authorization: Bearer $DASHSCOPE_API_KEY + X-DashScope-Async: enable
      轮询  GET  {BASE}/tasks/{task_id}(默认 15s 间隔 / 15min 超时,可配)
            SUCCEEDED → 下载 output.video_url;FAILED → 响亮 raise(code/message)
      上传  GET /uploads?action=getPolicy&model=… → OSS 表单直传 → oss://key;
            带媒体的请求加头 X-DashScope-OssResourceResolve: enable;
            同文件(路径+mtime)进程内缓存,只传一次。

    M0 实测契约:
      • 无 first_frame 的请求 aspect_ratio 必填;
      • first_frame + refer 可同请求混用(capability "first_frame_plus_refs")
        —— 续接镜 = 硬钉上镜尾帧 + 肖像参考的 API 级硬实现;
      • 引用方言 <<<image_N>>>(见 ref_token());
      • 本机系统代理会掐断长轮询 → requests 会话 trust_env=False 直连;
      • seed:API 无此字段 → 只记日志不传(诚实,不假装可复现)。

    八项裁决(docs/REQUIREMENTS_window_memory.md 2026-08-03):t2i / 图像
    编辑 / 音乐暂留 wavespeed —— config 给 wavespeed 子配置(或环境有
    $WAVESPEED_API_KEY)则内部实例化 WaveSpeedClient 代理,text_to_image /
    _run_task(audio_stage 的音乐)转发给它;没代理时响亮 raise(不装死)。

    config:
      models.video_gen:
        name: "bailian_kling"
        kling_model:      "kling/kling-v3-video-generation"       # 纯文本
        kling_omni_model: "kling/kling-v3-omni-video-generation"  # 带媒体
        mode: "std"                # std | pro
        aspect_ratio: "16:9"       # 无 first_frame 的请求必填(M0)
        generate_audio: false
        api_key: ...               # or $DASHSCOPE_API_KEY
        poll_interval: 15.0
        timeout: 900
        call_log: ...
        wavespeed: {...}           # t2i/音乐代理子配置(api_key 留空走环境)
    """

    BASE = "https://dashscope.aliyuncs.com/api/v1"
    SYNTH = BASE + "/services/aigc/video-generation/video-synthesis"

    def __init__(self, name: str = "bailian_kling",
                 config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = (self.config.get("api_key")
                        or os.getenv("DASHSCOPE_API_KEY"))
        self.kling_model = self.config.get(
            "kling_model", "kling/kling-v3-video-generation")
        self.kling_omni_model = self.config.get(
            "kling_omni_model", "kling/kling-v3-omni-video-generation")
        self.mode = self.config.get("mode", "std")
        self.aspect_ratio = self.config.get("aspect_ratio", "16:9")
        self.generate_audio = bool(self.config.get("generate_audio", False))
        self.poll_interval = float(self.config.get("poll_interval", 15.0))
        self.timeout = float(self.config.get("timeout", 900))
        self._upload_cache: dict = {}     # (resolved path, mtime_ns) → oss url
        self._sess = None                 # 懒建的直连 requests.Session
        # 裁决:t2i / 图像编辑 / 音乐暂留 wavespeed —— 内部代理
        ws_cfg = self.config.get("wavespeed")
        if isinstance(ws_cfg, dict) or os.getenv("WAVESPEED_API_KEY"):
            cfg = dict(ws_cfg or {})
            if self.config.get("call_log") and not cfg.get("call_log"):
                cfg["call_log"] = self.config["call_log"]   # 音乐调用同一本账
            self._wavespeed: Optional[WaveSpeedClient] = WaveSpeedClient(
                name="wavespeed", config=cfg)
        else:
            self._wavespeed = None

    # ── 基础设施 ──────────────────────────────────────────────
    def _session(self):
        """直连会话(M0 实测:本机系统代理会掐断长轮询 → trust_env=False,
        阿里云国内可达无需代理)。整个客户端复用同一条会话。"""
        if self._sess is None:
            import requests   # std in our [all] extras; loud ImportError otherwise
            s = requests.Session()
            s.trust_env = False
            self._sess = s
        return self._sess

    def _headers(self, async_: bool = True, oss: bool = False) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "BailianKlingClient needs an API key: set $DASHSCOPE_API_KEY "
                "or models.video_gen.api_key (or switch back to "
                "'mock-video-gen')."
            )
        h = {"Authorization": f"Bearer {self.api_key}",
             "Content-Type": "application/json"}
        if async_:
            h["X-DashScope-Async"] = "enable"
        if oss:   # 带媒体的请求:让服务端解析 oss:// 临时资源(M0)
            h["X-DashScope-OssResourceResolve"] = "enable"
        return h

    def ref_token(self, n: int) -> str:
        """可灵引用方言:prompt 里第 N 张 refer 图写作 <<<image_N>>>
        (M0 ref2v 实测服从度好;编号 = media 里 refer 的顺序)。"""
        return f"<<<image_{n}>>>"

    @staticmethod
    def _clamp_duration(seconds) -> int:
        """kling duration 域 = [3,15] 整数。小数向上取整后夹取(与 WaveSpeed
        的 snap-up 约定一致:缺的时长补不出来,多的可以裁)。"""
        want = (int(seconds) if float(seconds) == int(seconds)
                else int(seconds) + 1)
        return max(3, min(15, want))

    # call_log 摘要规则与 WaveSpeed 完全一致(语义文本全文保留、长 blob
    # 缩写);kling payload 顶层是 model/input/parameters 三个键,嵌套 dict
    # 原样透传 —— prompt 全文可读,oss:// URL 原样记(很短,天然保留)。
    _summarize_payload = WaveSpeedClient._summarize_payload

    def _log_call(self, record: dict) -> None:
        """call_log 机制与 WaveSpeedClient 同款(任务 0):模型名 + 参数打到
        终端 INFO,并逐行追加到 config `call_log` 指定的 JSONL。日志失败
        绝不影响调用本身。"""
        try:
            log.info("bailian_kling %s → %s %s", record.get("event"),
                     record.get("model"),
                     json.dumps({k: v for k, v in record.items()
                                 if k not in ("event", "model", "ts")},
                                ensure_ascii=False, default=str)[:1500])
        except Exception:
            pass
        path = self.config.get("call_log")
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str)
                        + "\n")
        except OSError as exc:
            log.warning("call_log write failed (%s): %s", path, exc)

    # ── 上传(getPolicy → OSS 表单直传 → oss://key)──────────────
    def _upload_media(self, path, model: Optional[str] = None) -> str:
        """本地文件 → DashScope 临时空间。http(s)/oss:// 直接透传;同文件
        (路径+mtime)进程内缓存,repair 轮反复引用同一帧只传一次。"""
        self._headers(async_=False)          # 无 key 先响亮(任何 IO 之前)
        vp = str(path)
        if vp.startswith(("http://", "https://", "oss://")):
            return vp
        p = Path(vp)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"media not found: {vp}")
        st = p.stat()
        ck = (str(p.resolve()), st.st_mtime_ns)
        if ck in self._upload_cache:
            return self._upload_cache[ck]
        # 上传前降采样(2026-08-03 实跑事故:2.4MB 级 ComfyUI 原图 PNG
        # 连传数张,OSS 读超时 120s → 整条 i2v_first 降级纯 t2v)。参考图
        # 不需要原始体积:长边 >1280 或 >1.2MB 的图片转 JPEG(质量 88),
        # 体积普遍降一个量级;转不动(非图/无 PIL)原样上传,留痕。
        p = self._shrink_for_upload(p)
        model = model or self.kling_omni_model
        sess = self._session()
        r = sess.get(f"{self.BASE}/uploads",
                     params={"action": "getPolicy", "model": model},
                     headers={"Authorization": f"Bearer {self.api_key}"},
                     timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(
                f"bailian_kling getPolicy for '{p.name}' failed: HTTP "
                f"{r.status_code} — {r.text[:2000]}")
        pol = r.json()["data"]
        key = f"{pol['upload_dir']}/{p.name}"
        # 大文件慢链路:读超时 300s + 超时/瞬断重试 3 次(指数退避)
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                with p.open("rb") as f:
                    files = {
                        "OSSAccessKeyId": (None, pol["oss_access_key_id"]),
                        "Signature": (None, pol["signature"]),
                        "policy": (None, pol["policy"]),
                        "x-oss-object-acl": (None,
                                             pol["x_oss_object_acl"]),
                        "x-oss-forbid-overwrite": (
                            None, pol["x_oss_forbid_overwrite"]),
                        "key": (None, key),
                        "success_action_status": (None, "200"),
                        "file": (p.name, f),
                    }
                    up = sess.post(pol["upload_host"], files=files,
                                   timeout=(10, 300))
                if up.status_code >= 500:
                    raise RuntimeError(f"OSS HTTP {up.status_code}")
                if up.status_code == 409:
                    # forbid-overwrite 撞 409 = 同 key 已存在 = 上一次
                    # 尝试实际已落盘(连接在响应前断了)→ 即成功。
                    log.info("bailian_kling: %s already uploaded (a "
                             "previous attempt landed) — using it",
                             p.name)
                    last_exc = None
                    break
                if up.status_code >= 400:
                    raise RuntimeError(
                        f"bailian_kling OSS upload of '{p.name}' failed: "
                        f"HTTP {up.status_code} — {up.text[:2000]}")
                last_exc = None
                break
            except Exception as exc:                    # 超时/瞬断可重试
                if "HTTP 4" in str(exc):
                    raise                               # 4xx 不是瞬断
                last_exc = exc
                log.warning("bailian_kling: upload of %s attempt %d "
                            "failed (%s) — retrying", p.name,
                            attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        if last_exc is not None:
            raise RuntimeError(
                f"bailian_kling OSS upload of '{p.name}' failed after "
                f"3 attempts: {last_exc}")
        url = f"oss://{key}"
        self._upload_cache[ck] = url
        return url

    def _shrink_for_upload(self, p: Path) -> Path:
        """参考图降采样:长边 ≤1280 的 JPEG(质量 88)。结果按源文件缓存
        (同目录 .upload_*.jpg);非图片/PIL 缺失/更小无益 → 原样返回。"""
        try:
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                return p
            if p.stat().st_size <= 1_200_000:
                from PIL import Image
                with Image.open(p) as im:
                    if max(im.size) <= 1280:
                        return p
            out = p.parent / f".upload_{p.stem}.jpg"
            if out.exists() and out.stat().st_mtime >= p.stat().st_mtime:
                return out
            from PIL import Image
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((1280, 1280))
                im.save(out, "JPEG", quality=88)
            log.info("bailian_kling: %s shrunk for upload %.1fMB → %.2fMB",
                     p.name, p.stat().st_size / 1e6,
                     out.stat().st_size / 1e6)
            return out
        except Exception as exc:
            log.warning("bailian_kling: shrink of %s failed (%s) — "
                        "uploading the original", p.name, exc)
            return p

    # ── payload(机械字段确定性补齐,LLM 不碰)─────────────────
    def _build_payload(self, prompt: str, media: list[dict],
                       duration, seed: int) -> tuple[str, dict]:
        """模型按有无媒体二选一(带媒体 → omni,纯文本 → 标准);duration
        None → 不传字段(API 默认);无 first_frame → aspect_ratio 必填
        (M0 实测);audio ← generate_audio 属性;seed API 不支持 → 只记
        日志不传。"""
        model = self.kling_omni_model if media else self.kling_model
        inp: dict = {"prompt": prompt}
        if media:
            inp["media"] = media
        params: dict = {"mode": self.mode,
                        "audio": bool(self.generate_audio)}
        if duration is not None:
            params["duration"] = self._clamp_duration(duration)
        if not any(m.get("type") == "first_frame" for m in media):
            params["aspect_ratio"] = self.aspect_ratio
        if seed:
            log.info("bailian_kling: seed=%s NOT sent — the DashScope kling "
                     "API has no seed field (logged for the record only)",
                     seed)
        return model, {"model": model, "input": inp, "parameters": params}

    # ── 生成入口(窗口管线的调用面)────────────────────────────
    def generate(
        self,
        prompt: str,
        duration: Optional[float],
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
        reference_video: Optional[Path] = None,
    ) -> Path:
        """t2v / i2v(first_frame)/ ref2v(refer×N)/ mix(first_frame+refer,
        M0 已验证可同请求混用)。media 里 refer 的顺序即 <<<image_N>>> 编号。

        reference_video:kling API 没有参考视频通道 → 在任何上传发生之前
        响亮拒绝(诚实,不静默丢条件)。capabilities() 刻意不含 "ref_video",
        能力驱动的菜单根本不会把它路由进来 —— 这里只防直呼。"""
        if reference_video is not None:
            raise RuntimeError(
                "BailianKlingClient has no reference-video channel — the "
                "DashScope kling API takes only first_frame/last_frame/refer "
                "media. Drop the reference_video condition or route this call "
                "to the 'wavespeed' backend (seedance-2.0 reference_videos)."
            )
        media: list[dict] = []
        if first_frame is not None:
            media.append({"type": "first_frame",
                          "url": self._upload_media(first_frame)})
        for img in (reference_images or []):
            media.append({"type": "refer", "url": self._upload_media(img)})
        model, payload = self._build_payload(prompt, media, duration, seed)
        return self._submit_and_poll(model, payload, out_path)

    def frame_to_frame(
        self,
        prompt: str,
        first_frame: Path,
        last_frame: Path,
        out_path: Path,
        duration: Optional[float] = 5,
        seed: int = 0,
    ) -> Path:
        """首尾帧生成(capability "flf2v",M0 flf2v 形态验证)。转场时长由
        调用方传(裁决:转场固定 3s);timeline.py 以 duration= 关键字调用。"""
        media = [
            {"type": "first_frame", "url": self._upload_media(first_frame)},
            {"type": "last_frame", "url": self._upload_media(last_frame)},
        ]
        model, payload = self._build_payload(prompt, media, duration, seed)
        return self._submit_and_poll(model, payload, out_path)

    # ── 提交 → 轮询 → 下载(异步任务协议,M0 原样)──────────────
    def _submit_and_poll(self, model: str, payload: dict,
                         out_path: Path) -> Path:
        needs_oss = bool((payload.get("input") or {}).get("media"))
        t0 = time.time()
        self._log_call({"ts": t0, "event": "submit", "model": model,
                        "payload": self._summarize_payload(payload),
                        "out": str(out_path)})
        sess = self._session()
        resp = sess.post(self.SYNTH, headers=self._headers(oss=needs_oss),
                         json=payload, timeout=60)
        if resp.status_code >= 400:
            self._log_call({"ts": time.time(), "event": "failed",
                            "model": model,
                            "error": f"HTTP {resp.status_code}: "
                                     f"{resp.text[:500]}"})
            raise RuntimeError(
                f"bailian_kling submit to '{model}' failed: HTTP "
                f"{resp.status_code} — {resp.text[:2000]}\n"
                f"payload: {self._summarize_payload(payload)}"
            )
        body = resp.json() if resp.text else {}
        task_id = (body.get("output") or {}).get("task_id")
        if not task_id:
            self._log_call({"ts": time.time(), "event": "failed",
                            "model": model, "error": "no task_id in submit"})
            raise RuntimeError(
                f"bailian_kling submit to '{model}' returned no task_id: "
                f"{json.dumps(body, ensure_ascii=False)[:2000]}"
            )

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = sess.get(f"{self.BASE}/tasks/{task_id}",
                         headers=self._headers(async_=False), timeout=30)
            if r.status_code >= 500:       # transient server hiccup → re-poll
                time.sleep(self.poll_interval)
                continue
            if r.status_code >= 400:
                raise RuntimeError(
                    f"bailian_kling poll for task {task_id} ('{model}') "
                    f"failed: HTTP {r.status_code} — {r.text[:2000]}"
                )
            data = r.json() if r.text else {}
            out = data.get("output") or {}
            status = out.get("task_status")
            if status == "SUCCEEDED":
                url = out.get("video_url")
                if not url:
                    raise RuntimeError(
                        f"bailian_kling task {task_id} SUCCEEDED but has no "
                        f"output.video_url: "
                        f"{json.dumps(out, ensure_ascii=False)[:2000]}")
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                video = sess.get(url, timeout=300)
                if video.status_code >= 400:
                    raise RuntimeError(
                        f"bailian_kling download for task {task_id} failed: "
                        f"HTTP {video.status_code}")
                out_path.write_bytes(video.content)
                self._log_call({"ts": time.time(), "event": "completed",
                                "model": model, "task_id": task_id,
                                "elapsed_s": round(time.time() - t0, 1),
                                "out": str(out_path)})
                return out_path
            if status in ("FAILED", "UNKNOWN"):
                code = out.get("code") or data.get("code")
                msg = out.get("message") or data.get("message")
                self._log_call({"ts": time.time(), "event": "failed",
                                "model": model, "task_id": task_id,
                                "error": f"{status} code={code} "
                                         f"message={str(msg)[:400]}"})
                raise RuntimeError(
                    f"bailian_kling task {task_id} ('{model}') {status}: "
                    f"code={code} message={msg}")
            time.sleep(self.poll_interval)
        self._log_call({"ts": time.time(), "event": "failed", "model": model,
                        "task_id": task_id,
                        "error": f"timeout after {self.timeout}s"})
        raise TimeoutError(
            f"bailian_kling task {task_id} did not finish within "
            f"{self.timeout}s (poll_interval={self.poll_interval}s)")

    # ── WaveSpeed 代理(裁决:t2i / 图像编辑 / 音乐暂留 wavespeed)──
    def _proxy(self, what: str) -> WaveSpeedClient:
        if self._wavespeed is None:
            raise RuntimeError(
                f"BailianKlingClient has no WaveSpeed proxy for {what} — "
                "t2i/图像编辑/音乐暂留 wavespeed(2026-08-03 裁决):配 "
                "models.video_gen.wavespeed 子配置或 $WAVESPEED_API_KEY。"
            )
        return self._wavespeed

    def text_to_image(self, prompt: str, out_path: Path,
                      seed: int = 0) -> Path:
        """t2i 转发内部 WaveSpeed 代理(窗口 keyframe 阶段用);没代理响亮
        raise,不装死。"""
        return self._proxy("text_to_image (t2i)").text_to_image(
            prompt, out_path, seed=seed)

    def _run_task(self, model_id: str, payload: dict, out_path: Path) -> Path:
        """WaveSpeed 原生任务直呼入口(audio_stage 的音乐生成
        `video_gen._run_task(text-to-music)` 走这里)—— 转发内部代理;
        代理带同一本 call_log,音乐调用照常入账。"""
        return self._proxy(f"_run_task('{model_id}')")._run_task(
            model_id, payload, out_path)

    # ── 能力面 ────────────────────────────────────────────────
    def supported_conditions(self) -> set[str]:
        return {"first_frame", "reference_images"}

    def capabilities(self) -> set[str]:
        # 刻意不含 "extend"/"ref_video"/"multi_i2v"/"edit" —— kling API 没有
        # 这些通道,旧菜单项按能力自然消失(能力驱动路由,不硬编码后端名)。
        # "first_frame_plus_refs":M0 实测 first_frame+refer 可同请求混用,
        # 续接镜 = 硬钉上镜尾帧 + 肖像参考,旧 ti2v_prev_plus_keyframe 的
        # 软钉意图从此有 API 级硬实现。
        caps = {"t2v", "i2v", "flf2v", "ref_images",
                "first_frame_plus_refs"}
        # 裁决(2026-08-03):t2i/图像编辑暂留 wavespeed —— 有代理时声明
        # "t2i",肖像/关键帧阶段照常;无代理如实不声明(阶段自然跳过)。
        if getattr(self, "_wavespeed", None) is not None:
            caps.add("t2i")
        return caps


# ─────────────────────────────────────────────────────────────
# OmniWeaving — preferred local backend (multi-condition native)
# ─────────────────────────────────────────────────────────────
class OmniWeavingClient(BaseVideoGenClient):
    """Tencent-Hunyuan/OmniWeaving. Native text + multi-image conditioning:
    first-frame anchor (C2) + reference images (E1) + text."""

    def __init__(self, name: str = "omniweaving", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.weights_path = self.config.get("weights_path") or os.getenv("OMNIWEAVING_WEIGHTS")
        self.device = self.config.get("device", "cuda")
        self._pipe = None  # lazy-loaded

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        # TODO(real): load the real pipeline, e.g.
        #   import torch; from omniweaving import OmniWeavingPipeline
        #   self._pipe = OmniWeavingPipeline.from_pretrained(self.weights_path).to(self.device)
        raise RuntimeError(
            "OmniWeavingClient not wired yet. Set models.video_gen.weights_path "
            "(or $OMNIWEAVING_WEIGHTS) and implement _ensure_loaded()/generate(). "
            "Until then use 'wavespeed' (API) or 'mock-video-gen'."
        )

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        self._ensure_loaded()
        # TODO(real): run the pipeline and write `out_path`, e.g.
        #   frames = self._pipe(prompt=prompt, image=first_frame,
        #                       ref_images=reference_images,
        #                       num_frames=int(duration*fps), seed=seed)
        #   write_video(out_path, frames, fps)
        raise NotImplementedError("OmniWeaving generate() not wired")

    def supported_conditions(self) -> set[str]:
        return {"first_frame", "reference_images"}


# ─────────────────────────────────────────────────────────────
# Wan — open-weight backup
# ─────────────────────────────────────────────────────────────
class WanClient(BaseVideoGenClient):
    """Wan2.x local I2V/T2V with first/last-frame anchoring. Backup to OmniWeaving."""

    def __init__(self, name: str = "wan", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.weights_path = self.config.get("weights_path") or os.getenv("WAN_WEIGHTS")
        self.device = self.config.get("device", "cuda")
        self._pipe = None

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        # TODO(real): load Wan pipeline; use first_frame as the I2V anchor;
        # write `out_path`.
        raise RuntimeError(
            "WanClient not wired yet. Set models.video_gen.weights_path (or "
            "$WAN_WEIGHTS) and implement generate()."
        )

    def supported_conditions(self) -> set[str]:
        return {"first_frame", "reference_images"}


# ─────────────────────────────────────────────────────────────
# API fallback — Veo / Sora (no local weights)
# ─────────────────────────────────────────────────────────────
class VeoApiClient(BaseVideoGenClient):
    """Hosted API fallback (text + optional first image)."""

    def __init__(self, name: str = "veo-api", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("VEO_API_KEY")

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        fps: int = 8,
        first_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        seed: int = 0,
    ) -> Path:
        if not self.api_key:
            raise RuntimeError("VeoApiClient needs $VEO_API_KEY (or config api_key).")
        # TODO(real): call the hosted endpoint, poll, download to out_path.
        raise NotImplementedError("Veo generate() not wired")

    def supported_conditions(self) -> set[str]:
        return {"first_frame"}


# ── 引用方言向后兼容:Mock 与 WaveSpeed 同款 @ImageN ──
# MockVideoGenClient 定义在 video_gen.py(该文件另有人在改,本轮只许动
# 本文件)—— 守卫式注入补齐方言方法;若那边日后正式定义 ref_token,
# 以那边为准(hasattr 守卫自动让位)。
if not hasattr(MockVideoGenClient, "ref_token"):
    def _mock_ref_token(self, n: int) -> str:
        """Mock 的图片引用方言,与 WaveSpeed 相同(@ImageN,向后兼容
        既有 prompt 模板与测试)。"""
        return f"@Image{n}"

    MockVideoGenClient.ref_token = _mock_ref_token


_REGISTRY = {
    "wavespeed": WaveSpeedClient,
    "bailian_kling": BailianKlingClient,
    "bailian": BailianKlingClient,
    "kling": BailianKlingClient,
    "omniweaving": OmniWeavingClient,
    "wan": WanClient,
    "veo": VeoApiClient,
    "veo-api": VeoApiClient,
    "sora": VeoApiClient,
}


def build_real_video_gen(name: str, config: Optional[dict] = None) -> BaseVideoGenClient:
    key = name.split("-")[0].lower() if name else ""
    cls = _REGISTRY.get(name.lower()) or _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown video_gen backend '{name}'. Known: {list(_REGISTRY)}")
    return cls(name=name, config=config)
