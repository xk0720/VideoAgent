"""Real MLLM/VLM judge & critic backends (v0.4) — OpenAI-compatible multimodal.

Implements the SAME `BaseMLLMClient` contract as the mock
(`assess_semantic` / `assess_physics` / `compare`), so the ReviewBoard and the
VISTA tournament are unchanged — flip `models.mllm.name` to a real VLM and set
the matching key.

Covers GPT-4o and Qwen-VL (DashScope OpenAI-compat) over the standard
`/chat/completions` multimodal message shape (text + base64 image_url parts).
Raw `requests`, lazy imports, no vendor SDK — same convention as
video_gen_backends.WaveSpeedClient and llm_backends.

────────────────────────────────────────────────────────────────────────────
HONESTY BRANCH — the load-bearing rule (mirrors track_extractor_backends)
────────────────────────────────────────────────────────────────────────────
A VLM judges PIXELS. The mock pipeline writes a TEXT placeholder with a .mp4
name (no decodable frames). `_sample_frames` reuses `_decode_frames` from
track_extractor_backends, which returns None for such non-videos. When there
are NO frames, the VLM has NO EVIDENCE, so it must NOT fabricate a verdict:

  • assess_semantic → [] (no judgment — NOT a fake passed=True "clip present"),
  • assess_physics  → [] (no measured violation),
  • compare         → super().compare (metric fallback, the de-biased default).

This is the project's "no signal from no evidence" rule (see
models/mock_signals.py): a critic that invents a verdict from nothing turns the
self-improve loop into a clock. The decode-None check runs BEFORE the API-key
check, so a mock clip degrades to [] WITHOUT needing a key — exactly the path
the smoke/mock pipeline takes when a real VLM is configured.

Per-CALL inference errors (HTTP failure, unparseable JSON) ALSO degrade safely
(return []/fallback + WARN via logging_utils), never crash — same philosophy as
CoTrackerExtractor. The ONLY loud failure is misconfiguration: a real backend
selected, frames present (real evidence), but no API key.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..logging_utils import get_logger
from ..physics.failure_modes import suggest_intervention
from ..physics.track_extractor_backends import _decode_frames
from ..types import CandidateClip, PhysFailureMode, PhysicsVerdict, ShotSpec
from .mllm import BaseMLLMClient

log = get_logger(__name__)

# Reuse the LLM provider defaults for base_url/key resolution, but with
# vision-capable default model ids.
_VLM_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "openai": ("https://api.openai.com/v1", "gpt-4o", "OPENAI_API_KEY"),
    "gpt-4o": ("https://api.openai.com/v1", "gpt-4o", "OPENAI_API_KEY"),
    "openai-vlm": ("https://api.openai.com/v1", "gpt-4o", "OPENAI_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max", "QWEN_API_KEY"),
    "qwen-vl": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max", "QWEN_API_KEY"),
}


def _extract_json(text: str):
    """Tolerant: pull the first JSON array/object out of a model reply, or None.

    Handles ```json fences and leading prose. Returns the parsed value or None
    (callers degrade to []/fallback on None — never crash on a chatty model)."""
    if not text:
        return None
    # try the whole thing first
    try:
        return json.loads(text)
    except Exception:
        pass
    # first [...] or {...} block (greedy to the matching style of last bracket)
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i = text.find(open_c)
        j = text.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                continue
    return None


def _encode_jpeg_b64(frame) -> Optional[str]:
    """Encode one RGB uint8 frame (H,W,3) as base64 JPEG. None on failure.

    Lazy: tries opencv, then PIL — whichever is importable. The decode path
    that produced the frames already required one of these, so this rarely
    misses, but it degrades to None (skip the frame) rather than crashing."""
    import base64

    try:
        import cv2  # type: ignore

        # frames are RGB (from _decode_frames); cv2 encodes BGR
        ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1])
        if ok:
            return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    try:
        import io

        from PIL import Image  # type: ignore

        bio = io.BytesIO()
        Image.fromarray(frame).save(bio, format="JPEG")
        return base64.b64encode(bio.getvalue()).decode()
    except Exception:
        return None


class OpenAICompatVLM(BaseMLLMClient):
    """A real vision-language judge over OpenAI-compatible multimodal chat.

    config:
      models.mllm:
        name: "gpt-4o"            # or qwen-vl / openai-vlm / openai
        model: "gpt-4o"           # or qwen-vl-max
        api_key: ...              # or OPENAI_API_KEY / QWEN_API_KEY / LLM_API_KEY
        base_url: ...             # default per provider name
        max_tokens: 1024
        n_frames: 4               # frames sampled per clip
    """

    def __init__(self, name: str = "gpt-4o", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        key = name.split("-")[0].lower() if name else ""
        d_base, d_model, env_var = _VLM_DEFAULTS.get(
            name.lower(), _VLM_DEFAULTS.get(key, ("https://api.openai.com/v1", "gpt-4o", "OPENAI_API_KEY"))
        )
        self.base_url = (
            self.config.get("base_url") or os.getenv("LLM_BASE_URL") or d_base
        ).rstrip("/")
        self.model = self.config.get("model", d_model)
        self.api_key = (
            self.config.get("api_key") or os.getenv(env_var) or os.getenv("LLM_API_KEY")
        )
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.n_frames = int(self.config.get("n_frames", 4))

    # ── frame sampling (honesty gate) ──
    def _sample_frames(self, clip: CandidateClip, k: Optional[int] = None):
        """Up to k evenly-spaced frames from the clip, or None if non-decodable.

        None ⇒ the clip is not a real video (e.g. the mock text placeholder):
        the VLM has no pixels to judge and the caller must emit NO verdict."""
        from pathlib import Path

        frames = _decode_frames(Path(clip.video_path))
        if frames is None or len(frames) == 0:
            return None
        k = k or self.n_frames
        n = len(frames)
        if n <= k:
            idxs = list(range(n))
        else:
            step = n / float(k)
            idxs = [min(n - 1, int(round(i * step))) for i in range(k)]
        return [frames[i] for i in idxs]

    def _require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                f"OpenAICompatVLM('{self.name}') needs an API key: set "
                f"models.mllm.api_key or the provider env var, or switch "
                f"models.mllm.name back to 'mock-mllm'."
            )
        return self.api_key

    def _chat(self, frames, text: str) -> Optional[str]:
        """One multimodal chat-completions call (frames + text) → reply str.

        Returns None on any transport/HTTP error (caller degrades safely).
        Raises only for the missing-key misconfiguration (via _require_key)."""
        import requests  # lazy

        key = self._require_key()
        content: list[dict] = [{"type": "text", "text": text}]
        for fr in frames:
            b64 = _encode_jpeg_b64(fr)
            if b64 is None:
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers,
                timeout=float(self.config.get("timeout", 120)),
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # non-fatal: degrade + warn
            log.warning(
                "VLM(%s) inference failed (%d frames): %r — no verdict emitted",
                self.name, len(frames), exc,
            )
            return None

    # ── semantic checklist ──
    def assess_semantic(self, clip: CandidateClip, spec: ShotSpec) -> list[tuple[str, bool, str]]:
        frames = self._sample_frames(clip)
        if frames is None:  # NO EVIDENCE → NO judgment (honesty branch)
            return []
        prompt = (
            "You are a strict video QA judge. The frames below are sampled from a "
            "generated video clip. The clip is meant to depict:\n"
            f"  \"{spec.prompt}\"\n"
            "Check whether the clip clearly shows the prompt's key elements. "
            "Respond with STRICT JSON only: a list of objects "
            "[{\"question\": str, \"passed\": bool, \"fix\": str}], one per key "
            "element. 'fix' is a short instruction to improve the clip when "
            "passed is false, else empty. No prose outside the JSON."
        )
        reply = self._chat(frames, prompt)
        data = _extract_json(reply) if reply is not None else None
        if not isinstance(data, list):
            if reply is not None:
                log.warning("VLM(%s) assess_semantic: unparseable reply, no verdict", self.name)
            return []
        items: list[tuple[str, bool, str]] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            q = str(it.get("question", "Does the clip match the prompt?"))
            passed = bool(it.get("passed", False))
            fix = "" if passed else str(it.get("fix", ""))
            items.append((q, passed, fix))
        return items

    # ── physics verdicts ──
    def assess_physics(self, clip: CandidateClip, spec: ShotSpec, fps: int) -> list[PhysicsVerdict]:
        frames = self._sample_frames(clip)
        if frames is None:  # NO EVIDENCE → NO verdict (honesty branch)
            return []
        expected = (
            spec.physics_annotation.expected_modes
            if spec.physics_annotation and spec.physics_annotation.expected_modes
            else []
        )
        allowed = [m.value for m in expected] if expected else [m.value for m in PhysFailureMode]
        n_frames = max(1, int(round(spec.duration * fps)))
        prompt = (
            "You are a physics-plausibility judge for a generated video clip. "
            "The frames below are sampled in time order. Report physical-law "
            "violations you can see. Respond with STRICT JSON only: a list of "
            "[{\"mode\": str, \"frame_start\": int, \"frame_end\": int, "
            "\"severity\": float, \"intervention\": str}]. 'mode' MUST be one of: "
            f"{allowed}. severity is 0..1 (higher = worse). frame indices are in "
            f"[0, {n_frames}]. Empty list if the motion looks physically plausible. "
            "No prose outside the JSON."
        )
        reply = self._chat(frames, prompt)
        data = _extract_json(reply) if reply is not None else None
        if not isinstance(data, list):
            if reply is not None:
                log.warning("VLM(%s) assess_physics: unparseable reply, no verdict", self.name)
            return []
        verdicts: list[PhysicsVerdict] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            raw_mode = str(it.get("mode", "")).strip().lower()
            try:
                mode = PhysFailureMode(raw_mode)
            except ValueError:
                log.warning("VLM(%s) assess_physics: unknown mode %r skipped", self.name, raw_mode)
                continue
            try:
                start = max(0, min(n_frames, int(it.get("frame_start", 0))))
                end = max(start, min(n_frames, int(it.get("frame_end", n_frames))))
            except (TypeError, ValueError):
                start, end = 0, n_frames
            try:
                severity = max(0.0, min(1.0, float(it.get("severity", 0.5))))
            except (TypeError, ValueError):
                severity = 0.5
            intervention = str(it.get("intervention", "")).strip() or suggest_intervention(mode)
            verdicts.append(
                PhysicsVerdict(
                    mode=mode,
                    frame_range=(start, end),
                    severity=severity,
                    suggested_intervention=intervention,
                    source="vlm",
                )
            )
        return verdicts

    # ── pairwise comparison (VISTA tournament) ──
    def compare(self, a: CandidateClip, b: CandidateClip, spec: ShotSpec) -> int:
        fa = self._sample_frames(a)
        fb = self._sample_frames(b)
        if fa is None or fb is None:  # no pixels on one side → metric fallback
            return super().compare(a, b, spec)
        na = len(fa)
        prompt = (
            "Two generated video clips (A then B) are meant to depict:\n"
            f"  \"{spec.prompt}\"\n"
            f"The first {na} frames are clip A; the rest are clip B. Decide which "
            "better depicts the prompt with plausible motion. Respond with STRICT "
            "JSON only: {\"winner\": \"A\"|\"B\"|\"tie\"}. No prose."
        )
        reply = self._chat(fa + fb, prompt)
        data = _extract_json(reply) if reply is not None else None
        if not isinstance(data, dict):
            return super().compare(a, b, spec)
        winner = str(data.get("winner", "")).strip().lower()
        if winner in ("a", "clip a", "1"):
            return 1
        if winner in ("b", "clip b", "2"):
            return -1
        if winner in ("tie", "draw", "equal", "0"):
            return 0
        return super().compare(a, b, spec)


# name (or its provider prefix) → backend class
_REGISTRY = {
    "gpt-4o": OpenAICompatVLM,
    "openai-vlm": OpenAICompatVLM,
    "openai": OpenAICompatVLM,
    "qwen-vl": OpenAICompatVLM,
    "qwen": OpenAICompatVLM,
}


def build_real_mllm(name: str, config: Optional[dict] = None) -> BaseMLLMClient:
    """Dispatch a real VLM judge by config name. Unknown → ValueError."""
    key = name.split("-")[0].lower() if name else ""
    cls = _REGISTRY.get(name.lower()) or _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"unknown mllm backend '{name}'. known: {sorted(_REGISTRY)} (+ 'mock-mllm')"
        )
    return cls(name=name, config=config)
