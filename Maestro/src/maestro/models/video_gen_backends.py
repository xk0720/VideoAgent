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
  • OmniWeavingClient / WanClient — local-weights skeletons (fill + GPU).
  • VeoApiClient — hosted API skeleton.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from .video_gen import BaseVideoGenClient

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
        resolution: "1080p"        # 480p | 720p | 1080p | 4k (seedance-2.0)
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
        self.resolution = self.config.get("resolution", "1080p")
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
        duration: float,
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
        model_id = self.model_id
        is_i2v = first_frame is not None and (
            str(first_frame).startswith(("http://", "https://"))
            or Path(first_frame).exists())
        if is_i2v:
            model_id = self._i2v_variant(model_id)

        payload: dict = {"prompt": prompt,
                         "duration": self._snap_duration(duration, model_id)}
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
            payload["reference_videos"] = [self._upload_media(reference_video)]
        if reference_images:
            # seedance-2.0 reference_images channel (docs-verified: ≤9 images,
            # referenced in the prompt as @Image1…@ImageN, combinable with
            # image/last_image and reference_videos up to 12 files total).
            if not self._is_range_family(model_id):
                raise RuntimeError(
                    f"model '{model_id}' has no reference_images channel — "
                    "multi-image conditioning needs the seedance-2.0 family "
                    "(or use multi_image_to_video for the Kling multi-i2v route)")
            refs = list(reference_images)[:9]
            if len(reference_images) > 9:
                log.info("reference_images capped to 9 (got %d) — seedance-2.0 "
                         "documented limit", len(reference_images))
            payload["reference_images"] = [self._upload_media(p) for p in refs]
        payload.update(self.config.get("extra_params") or {})

        return self._run_task(model_id, payload, out_path)

    def multi_image_to_video(
        self,
        prompt: str,
        images: list,
        out_path: Path,
        duration: float = 5,
        seed: int = 0,
    ) -> Path:
        """Multi-image FUSION i2v — capability "multi_i2v". One request takes an
        `images` ARRAY (≤4, docs-verified) and composes ONE video consistent
        with all of them (scene composition + character consistency), e.g.
        [previous shot's last frame, this shot's keyframe, identity anchor].
        Default route kwaivgi/kling-v1.6-multi-i2v-standard (the only array-
        input i2v verified on WaveSpeed 2026-07); configurable via
        `multi_i2v_model`. Distinct from `reference_images` (steering refs on
        top of a first-frame anchor): here the images ARE the condition, with
        no designated first frame."""
        if not images:
            raise ValueError("multi_image_to_video needs at least one image")
        model = self.config.get("multi_i2v_model",
                                "kwaivgi/kling-v1.6-multi-i2v-standard")
        imgs = list(images)[:4]
        if len(images) > 4:
            log.info("multi_image_to_video: images capped to 4 (got %d) — "
                     "documented Kling multi-i2v limit", len(images))
        payload = {
            "prompt": prompt,
            "images": [self._upload_media(p) for p in imgs],
            "duration": self._snap_duration(duration, model),   # {5,10} enum
            "aspect_ratio": self.config.get("aspect_ratio", "16:9"),
        }
        return self._run_task(model, payload, out_path)

    @staticmethod
    def _summarize_payload(payload: dict) -> dict:
        """Payload with base64 blobs shortened — safe to put in an error message."""
        return {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200
                    else v)
                for k, v in payload.items()}

    # ── shared submit → poll predictions/{id}/result → download (UniVA protocol) ──
    def _run_task(self, model_id: str, payload: dict, out_path: Path) -> Path:
        import requests  # std in our [all] extras; loud ImportError otherwise

        resp = requests.post(
            f"{self.BASE}/{model_id}", json=payload, headers=self._headers(),
            timeout=60,
        )
        if resp.status_code >= 400:
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
                return out_path
            if status == "failed":
                raise RuntimeError(f"WaveSpeed task {task_id} failed: "
                                   f"{data.get('error', 'unknown error')}")
            time.sleep(self.poll_interval)
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
        duration: float = 5,
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
                "duration": self._snap_duration(duration, model),
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
                "duration": self._snap_duration(duration, model),
                "resolution": self.resolution,
                "generate_audio": self.generate_audio,
            }
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
        duration: float = 5,
        seed: int = 0,
    ) -> Path:
        """Continue an existing clip beyond its last frame. Capability "extend"
        (optional). WaveSpeed now hosts DEDICATED video-extend models — default
        `bytedance/seedance-2.0/video-extend`: the whole input video (uploaded,
        passed by URL) conditions a true continuation, replacing the old
        decode-last-frame → i2v hack (better seam continuity: the model sees
        the motion, not one frame). Loud RuntimeError (no key) fires first."""
        payload = {
            "prompt": prompt,
            "video": self._upload_media(video_path),   # loud key check inside
            "duration": self._snap_duration(duration, self.extend_model),
            "resolution": self.resolution,
            "generate_audio": self.generate_audio,
        }
        return self._run_task(self.extend_model, payload, out_path)

    def supported_conditions(self) -> set[str]:
        return {"first_frame"}

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


_REGISTRY = {
    "wavespeed": WaveSpeedClient,
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
