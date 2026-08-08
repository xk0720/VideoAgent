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
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from ..physics.failure_modes import suggest_intervention
from ..physics.track_extractor_backends import _decode_frames
from ..types import (
    CandidateClip,
    ChecklistItem,
    PhysFailureMode,
    PhysicsVerdict,
    ShotSpec,
)
from ..language import lang_clause
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


# 钦定角色正典打标(2026-08-04 run7 事故:通用一句话 caption 没写军装
# 颜色 → character_extract 的 LLM 把黑军装脑补成"white royal military
# coat" → 评审拿错误正典扣真实忠于参考图的视频)。正典打标必须逐项带
# 精确颜色;看不见的细节【省略】,绝不猜。
_IDENTITY_CAPTION_INSTRUCTION = (
    "You are writing the appearance canon for a film character from their "
    "official portrait. Describe ONLY what is visible, precisely: face "
    "shape, skin tone, eye color, hair color/length/style, then the attire "
    "item by item WITH EXACT COLORS (e.g. 'black military coat with gold "
    "epaulettes and a red sash, white gloves'). One dense sentence. If a "
    "detail is not visible, OMIT it — never guess. No preamble, no "
    "category prefix.")


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
            if resp.status_code >= 400:
                # keep the response BODY in the warning — that is where the
                # API says WHICH field is wrong (raise_for_status drops it)
                log.warning(
                    "VLM(%s model=%s) HTTP %d: %s — no verdict emitted",
                    self.name, self.model, resp.status_code, resp.text[:500],
                )
                return None
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # non-fatal: degrade + warn
            log.warning(
                "VLM(%s) inference failed (%d frames): %r — no verdict emitted",
                self.name, len(frames), exc,
            )
            return None

    def caption_image(self, image_path) -> str:
        """One-sentence asset label from the REAL VLM (Q-D: fills the middle
        link of user-description > VLM caption > filename). Reads the image
        file directly; any failure returns "" (fall through the chain)."""
        return self._caption_with_text(image_path, (
            "Describe this image in ONE short sentence for retrieval: "
            "what/who it shows and the setting. Also start with one "
            "category word from [background, character, object, style] "
            "and a colon. Example: 'background: a cozy living room at "
            "night with a lit fireplace'. No other text."))

    def caption_identity(self, image_path) -> str:
        """角色正典打标:逐项精确颜色,看不见就省略(绝不猜)。"""
        return self._caption_with_text(
            image_path,
            _IDENTITY_CAPTION_INSTRUCTION + lang_clause("the sentence"))

    def _caption_with_text(self, image_path, text: str) -> str:
        import base64 as _b64
        from pathlib import Path as _P

        p = _P(str(image_path))
        if not p.exists() or p.suffix.lower() not in (
            ".png", ".jpg", ".jpeg", ".webp", ".bmp"
        ):
            return ""
        try:
            b64 = _b64.b64encode(p.read_bytes()).decode()
        except OSError:
            return ""
        mime = "png" if p.suffix.lower() == ".png" else "jpeg"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                {"type": "text", "text": text},
            ]}],
        }
        try:
            import requests  # lazy

            self._require_key()
            resp = requests.post(
                f"{self.base_url}/chat/completions", json=payload,
                headers=headers, timeout=float(self.config.get("timeout", 120)),
            )
            if resp.status_code >= 400:
                log.warning("VLM(%s) caption HTTP %d: %s", self.name,
                            resp.status_code, resp.text[:300])
                return ""
            return str(resp.json()["choices"][0]["message"]["content"]).strip()
        except Exception as exc:  # degrade down the Q-D chain, loudly
            log.warning("VLM(%s) caption_image failed for %s: %r",
                        self.name, p.name, exc)
            return ""

    # ── semantic checklist ──
    def assess_semantic(self, clip: CandidateClip, spec: ShotSpec) -> list[tuple]:
        """Items are (question, passed, fix) or — when the VLM localized the
        failure — (question, passed, fix, (frame_start, frame_end)).
        Q3 ruling: the VLM is REQUIRED to localize every FAILED item to a
        frame span (segment repair needs WHERE); an item it cannot localize
        honestly degrades to the 3-tuple (whole-clip defect), never a made-up
        range. SemanticCritic accepts both shapes."""
        frames = self._sample_frames(clip)
        if frames is None:  # NO EVIDENCE → NO judgment (honesty branch)
            return []
        # Original frame count so the VLM answers in ORIGINAL indices.
        try:
            from pathlib import Path as _P

            decoded = _decode_frames(_P(str(clip.video_path)))
            n_total = len(decoded) if decoded is not None else 0
        except Exception:
            n_total = 0
        prompt = (
            "You are a strict video QA judge. The frames below are sampled IN "
            f"TIME ORDER ({len(frames)} frames evenly spaced across a "
            f"{n_total}-frame clip). The clip is meant to depict:\n"
            f"  \"{spec.prompt}\"\n"
            "Check whether the clip clearly shows the prompt's key elements. "
            "Respond with STRICT JSON only: a list of objects "
            "[{\"question\": str, \"passed\": bool, \"fix\": str, "
            "\"frame_start\": int, \"frame_end\": int}], one per key element. "
            "'fix' is a short instruction to improve the clip when passed is "
            "false, else empty. For every FAILED item you MUST localize the "
            "problem: frame_start/frame_end are original-clip frame indices in "
            f"[0, {max(n_total, 1)}] bounding WHERE the problem is visible — "
            "the failure's own time span, NOT the whole clip unless it truly "
            "spans everything. No prose outside the JSON."
        )
        reply = self._chat(frames, prompt)
        data = _extract_json(reply) if reply is not None else None
        if not isinstance(data, list):
            if reply is not None:
                log.warning("VLM(%s) assess_semantic: unparseable reply, no verdict", self.name)
            return []
        items: list[tuple] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            q = str(it.get("question", "Does the clip match the prompt?"))
            passed = bool(it.get("passed", False))
            fix = "" if passed else str(it.get("fix", ""))
            fr = None
            try:
                fs, fe = int(it["frame_start"]), int(it["frame_end"])
                if fe > fs >= 0:
                    fr = (fs, fe)
            except (KeyError, TypeError, ValueError):
                fr = None      # VLM 没给/给废了 → 诚实的 None(回退整镜级)
            items.append((q, passed, fix, fr) if fr is not None
                         else (q, passed, fix))
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
        n_frames = max(1, int(round((spec.duration or 5.0) * fps)))
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

# ─────────────────────────────────────────────────────────────────────────
# REVIEWER instruction (merged single-call shot review, native video).
# Design goal (user ruling 2026-07-14): the output must be PROBLEM
# LOCALIZATIONS an automated repair Brain can consume — a wrong FRAME, a
# wrong SEGMENT, or a global problem; each with WHERE + reason +
# (suggestion). The five dimensions are the checklist the reviewer works
# through; the LOCALIZED ISSUES are the product.
# ─────────────────────────────────────────────────────────────────────────
_SHOT_REVIEW_INSTRUCTION = """You are the shot REVIEWER of an automated \
video-generation system. Above you were shown THE SHOT VIDEO under review \
and, when provided, the CONDITIONS it was generated from (first/last-frame \
images, reference images, a reference video whose motion it continues). \
The shot is meant to depict:
"{shot_prompt}"{gen_prompt_block}
{conditions_block}{junction_block}

Work through FIVE dimensions:
1. Semantic adherence — does the video show the described facts (objects, \
counts, colors, identities, setting, the action and its ORDER)?
2. Condition adherence — is it consistent with EACH provided condition \
(same subject as the reference images; opens on the first-frame image; \
continues the reference video's motion seamlessly)?
3. Physical correctness — is all motion physically plausible (gravity, \
inertia, momentum, collisions, contact) with no interpenetration, floating, \
teleporting, objects vanishing/appearing, or broken cause-effect order?
4. Temporal consistency — no flicker, no identity/appearance drift, no \
background jumps within the clip.
5. Visual quality — no artifacts, deformed anatomy, smeared textures, \
broken geometry.

Your product is the LOCALIZED ISSUE LIST — it drives an automated repair \
agent, so vague comments are useless. Every issue must say:
- "type": "frame" (one instant is wrong) | "segment" (a span is wrong) | \
"global" (the whole clip);
- "time_start_s"/"time_end_s": WHERE, in seconds from clip start (for \
"frame" use a tight ~0.2 s window; for "global" span the whole clip);
- "category": "semantic" | "condition" | "physics" | "temporal" | "visual";
- "physics_mode" (ONLY when category="physics"): "gravity_inertia" | \
"collision" | "conservation" | "object_permanence" | "penetration" | \
"fluid" | "unexplained";
- "entity": the object/subject concerned ("the glass"; "" if none);
- "severity": 0.0-1.0 — how badly it breaks the shot;
- "problem": ONE concrete sentence — what exactly is wrong THERE;
- "reason": WHY it is wrong — which stated fact, provided condition, or \
physical law it violates;
- "suggestion": what CORRECT looks like there (strongly encouraged);
- "check_ref": the index into `checks` of the failed check this issue \
explains, or -1 if none.

Also return "checks": the concrete yes/no checks you performed — one per \
stated fact of the description and one per provided condition — each \
{{"question": str, "passed": true|false}}. EVERY failed check MUST have a \
matching issue (via check_ref) that localizes it.

Judge ONLY from observable evidence; do not invent requirements; if the \
video is flawless return an empty issues list. You are ONLY a reviewer: \
describe problems — do NOT recommend repair tools or next steps; that \
decision belongs to another agent.

Reply with ONLY a JSON object, no markdown fence:
{{"checks": [{{"question": str, "passed": bool}}],
 "issues": [{{"type": str, "time_start_s": float, "time_end_s": float,
 "category": str, "physics_mode": str, "entity": str, "severity": float,
 "problem": str, "reason": str, "suggestion": str, "check_ref": int}}],
 "summary": "one sentence overall"}}"""


# ─────────────────────────────────────────────────────────────────────────
# VERIFIER instruction (blind A/B, repaired vs original). Fundamentally a
# DIFFERENT question from the reviewer's: not "what is wrong with this
# video" but "did the repair make it BETTER without breaking anything".
# NEWTON mechanics (blind slots, signed score, conservative 0) + our
# additions: per-dimension signed scores and a defect-presence probe.
# ─────────────────────────────────────────────────────────────────────────
_VERIFY_PAIR_INSTRUCTION = """You are an impartial judge in a BLIND A/B \
test of a video REPAIR. One of the two videos above is a repaired revision \
of the other — you are NOT told which; judge purely from what you observe. \
Both attempt the same shot:
"{shot_prompt}"
{target_block}

Score FOUR dimensions, each a signed integer in [-10, +10] = how much \
BETTER Video 2 is than Video 1 on that dimension (0 = no clear difference; \
negative = Video 2 is worse):
- "semantic": which better matches the shot's stated facts (objects, \
counts, identities, setting, action and its order)?
- "physics": which moves more plausibly (gravity, inertia, collisions, \
contact; no interpenetration / floating / teleporting / vanishing)?
- "temporal": which is more internally consistent (no flicker, identity \
drift, background jumps)?
- "visual": which looks cleaner (no artifacts, deformations, smearing)?

Also output the overall "score" in [-10, +10] (same convention), and — if \
a target defect was stated above — "defect_present": whether EACH video \
exhibits that defect.

Be conservative: use 0 when there is no clear difference. A repair that \
fixes one thing but clearly breaks another dimension must show that as a \
negative score on the broken dimension — do not average it away. Judge \
only observable evidence. You are ONLY a judge: score and describe; do NOT \
recommend tools or next steps.

Reply with ONLY a JSON object, no markdown fence:
{{"dim_scores": {{"semantic": int, "physics": int, "temporal": int, \
"visual": int}},
 "notes": {{"semantic": "one sentence", "physics": "one sentence", \
"temporal": "one sentence", "visual": "one sentence"}},
 "score": int,
 "defect_present": {{"video1": bool, "video2": bool}},
 "issues": ["short concrete strings — what the WORSE video still gets wrong"],
 "summary": "one sentence overall"}}"""


class GeminiVLM(OpenAICompatVLM):
    """Gemini as the review VLM over NATIVE VIDEO input (2026-07-14 ruling:
    frame sampling is FORBIDDEN here — gemini-3.5-flash reads video directly).

    Transport = Google generateContent (`POST {base}/v1beta/models/{model}:
    generateContent`), media as labeled text + inline_data parts — exactly
    NEWTON's shape (arXiv:2605.18396 GeminiVerifierCore). Three jobs:

      • review_shot   — ONE merged call per clip (U6 ruling: one upload):
                        the SHOT VIDEO + its generation CONDITIONS (key
                        images / reference video) + the shot text → localized
                        issues (frame/segment/global + reason + suggestion,
                        Brain-consumable) + the yes/no checks performed.
                        assess_semantic / assess_physics SLICE this cached
                        result — the critics' contracts are unchanged.
      • verify_pair   — NEWTON-style BLIND A/B for the Verifier: repaired vs
                        original, randomized slots, 4 dimensions with
                        per-dimension signed scores (our innovation over
                        NEWTON's single score) + a defect-presence probe.
      • compare       — light pairwise pick for the tournament (bidirectional
                        de-biasing happens in the caller).

    config (models.mllm):
      name: "gemini"
      model: "gemini-3.5-flash"     # or $GEMINI_MODEL
      api_key: ...                  # or $GEMINI_API_KEY
      base_url: ...                 # or $GEMINI_BASE_URL (official default)
    """

    # NEWTON's inline budget: request bodies above this get rejected. Videos
    # over the limit are TRANSCODED down (still native video, never frames).
    MAX_INLINE_MB = 18.0

    def __init__(self, name: str = "gemini", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.base_url = (
            self.config.get("base_url") or os.getenv("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        self.model = (self.config.get("model") or os.getenv("GEMINI_MODEL")
                      or "gemini-3.5-flash")
        self.api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.max_retries = int(self.config.get("max_retries", 3))
        # 原生视频的采样帧率(Kevin 2026-07-16):Gemini 官方默认 1.0 fps,
        # 对物理/时序评审和秒级定位太稀 —— 默认提到 5.0(token 相应变多)。
        # config models.mllm.video_fps 可调;官方合法域 0-24。
        self.video_fps = max(0.0, min(24.0,
                                      float(self.config.get("video_fps",
                                                            5.0))))
        # merged-review cache: (path, mtime_ns) → parsed review package.
        # One upload serves BOTH critics (semantic + physics) per clip.
        self._review_cache: dict = {}

    def _require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                f"GeminiVLM('{self.name}') needs an API key: set "
                "models.mllm.api_key or $GEMINI_API_KEY, or switch "
                "models.mllm.name back to 'mock-mllm'."
            )
        return self.api_key

    def _headers(self) -> dict:
        key = self._require_key()
        return {
            "x-goog-api-key": key,  # official Google API
            "Content-Type": "application/json",
        }

    def _generate(self, parts: list, **kwargs) -> Optional[str]:
        """One generateContent call with NEWTON-style retries (backoff) →
        reply text, or None (caller degrades to no-verdict)."""
        import time as _time

        import requests  # lazy

        # ── 视频采样帧率(Kevin 2026-07-16,已收编为配置驱动)────────────
        # Gemini 对视频默认 1.0 fps 采样,评审看不清快动作、秒级定位粗;
        # 给每个视频 Part 挂 videoMetadata.fps(Part 级字段,与 inline_data
        # 平级,REST 收 camelCase)。fps 优先级:调用方 kwargs > config
        # video_fps(默认 5.0)。拷贝 Part 再挂字段 —— 不改写调用方的共享
        # dict(拿别人的对象不动手脚)。
        fps = float(kwargs.get("fps", self.video_fps))
        if fps > 0:
            parts = [
                ({**part, "videoMetadata": {"fps": fps}}
                 if str(part.get("inline_data", {}).get("mime_type", "")
                        ).startswith("video/") else part)
                for part in parts
            ]

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"temperature": 0}}
        headers = self._headers()   # 缺 key = 配置错误,必须响亮 —— 在重试
        # 循环【之外】检查,不许被 except 吞成"无判定"静默降级。
        last = ""
        for attempt in range(max(1, self.max_retries)):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers,
                    timeout=float(self.config.get("timeout", 180)))
                if resp.status_code >= 400:
                    last = f"HTTP {resp.status_code}: {resp.text[:500]}"
                else:
                    cands = resp.json().get("candidates") or []
                    for part in reversed(
                        (cands[0].get("content") or {}).get("parts", [])
                        if cands else []
                    ):
                        if part.get("text"):
                            return str(part["text"])
                    last = "no text part in response"
            except Exception as exc:
                last = repr(exc)
            if attempt + 1 < self.max_retries:
                _time.sleep(min(10, 1 + 2 * attempt))
        log.warning("GeminiVLM(%s model=%s) failed after %d attempt(s): %s — "
                    "no verdict emitted", self.name, self.model,
                    self.max_retries, last)
        return None

    # ── media parts (NEWTON _video_part shape: label text + inline_data) ──
    def _fit_video_for_inline(self, path) -> Optional[str]:
        """Path of an inline-able video: the original when under the budget,
        else an ffmpeg 360p transcode written NEXT TO it (still native video —
        frames are forbidden). None when nothing fits (caller skips the part
        with a loud log; review degrades to no-verdict, never to frames)."""
        import shutil
        import subprocess
        from pathlib import Path as _P

        p = _P(str(path))
        if not p.is_file():
            return None
        if p.stat().st_size / 1e6 <= self.MAX_INLINE_MB:
            return str(p)
        if not shutil.which("ffmpeg"):
            log.warning("GeminiVLM: %s is %.1f MB over the %.0f MB inline "
                        "limit and ffmpeg is missing — cannot review natively",
                        p.name, p.stat().st_size / 1e6, self.MAX_INLINE_MB)
            return None
        out = p.with_name(p.stem + "_inline360.mp4")
        if not out.exists() or out.stat().st_mtime < p.stat().st_mtime:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-vf", "scale=-2:360",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "33",
                 "-an", str(out)], capture_output=True, timeout=600)
            if r.returncode != 0 or not out.exists():
                log.warning("GeminiVLM: transcode of %s failed", p.name)
                return None
        if out.stat().st_size / 1e6 > self.MAX_INLINE_MB:
            log.warning("GeminiVLM: %s still over the inline limit after "
                        "transcode — cannot review natively", p.name)
            return None
        return str(out)

    def _video_part(self, label: str, path) -> list:
        """[label text, inline video] — or [] with a loud log when the file
        cannot be inlined (missing / oversize with no ffmpeg)."""
        import base64 as _b64
        from pathlib import Path as _P

        fit = self._fit_video_for_inline(path)
        if fit is None:
            return []
        data = _b64.b64encode(_P(fit).read_bytes()).decode("ascii")
        return [{"text": f"=== {label} ==="},
                {"inline_data": {"mime_type": "video/mp4", "data": data}}]

    @staticmethod
    def _image_part(label: str, path) -> list:
        import base64 as _b64
        from pathlib import Path as _P

        p = _P(str(path))
        if not p.is_file() or p.suffix.lower() not in (
            ".png", ".jpg", ".jpeg", ".webp", ".bmp"
        ):
            return []
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        try:
            data = _b64.b64encode(p.read_bytes()).decode("ascii")
        except OSError:
            return []
        return [{"text": f"=== {label} ==="},
                {"inline_data": {"mime_type": mime, "data": data}}]

    # ── the merged shot review (ONE upload; both critics slice it) ──
    def _conditioning_parts(self, clip: CandidateClip) -> tuple[list, list[str]]:
        """Inline every generation CONDITION attached to the clip
        (clip.conditioning, set by the window loop) with role-labeled headers.
        Returns (parts, human labels) — labels feed the instruction so the
        reviewer knows what each condition was FOR."""
        cond = getattr(clip, "conditioning", None) or {}
        parts: list = []
        labels: list[str] = []
        role_names = {"first_frame": "FIRST-FRAME image (the shot must open on it)",
                      "first": "FIRST-FRAME image (the shot must open on it)",
                      "last": "LAST-FRAME image (the shot must end on it)",
                      "reference": "REFERENCE image (subject/scene to stay consistent with)",
                      "identity_portrait": "IDENTITY PORTRAIT (the character's "
                      "OFFICIAL look — appearance in the shot must match this "
                      "portrait; judge the cast checks against it)"}
        for i, im in enumerate(cond.get("images") or []):
            label = (f"CONDITION {len(labels) + 1}: "
                     + role_names.get(im.get("role", ""), "REFERENCE image"))
            got = self._image_part(label, im.get("path", ""))
            if got:
                parts += got
                labels.append(label)
        rv = cond.get("reference_video")
        if rv:
            label = (f"CONDITION {len(labels) + 1}: REFERENCE VIDEO (the "
                     "motion this shot continues from)")
            got = self._video_part(label, rv)
            if got:
                parts += got
                labels.append(label)
        return parts, labels

    def review_shot(self, clip: CandidateClip, spec: ShotSpec,
                    fps: int = 24) -> Optional[dict]:
        """ONE native-video review call → {"items": [ChecklistItem...],
        "verdicts": [PhysicsVerdict...], "summary": str}. Cached per
        (path, mtime) so semantic+physics critics share a single upload.
        None ⇒ nothing reviewable (no verdict is ever fabricated)."""
        from pathlib import Path as _P

        from ..physics.track_extractor_backends import _looks_like_video

        p = _P(str(clip.video_path))
        # HONESTY gate(接替旧的"无帧不判"规则):mock 文本桩顶着 .mp4 名字,
        # 魔数嗅探不过 → 无证据、无判定、也【不需要 key】;绝不把垃圾字节
        # 当视频喂给 API。
        if not p.is_file() or not _looks_like_video(p):
            return None
        key = (str(p.resolve()), p.stat().st_mtime_ns)
        if key in self._review_cache:
            return self._review_cache[key]

        video = self._video_part("THE SHOT VIDEO (under review)", p)
        if not video:
            return None                     # not inline-able → honest silence
        cond_parts, cond_labels = self._conditioning_parts(clip)
        cond = getattr(clip, "conditioning", None) or {}
        gen_prompt = str(cond.get("video_prompt") or "").strip()

        # duration/fps for the seconds→frames conversion of localizations
        real_fps = 0.0
        try:
            from ..pipeline.timeline import _probe_fps

            real_fps = _probe_fps(p)
        except Exception:
            real_fps = 0.0
        fps_eff = real_fps if real_fps > 0 else float(fps or 24)

        # 需求 ④(2026-07-15):镜间衔接进评审 —— 开头必须延续上一镜的
        # 真实结束状态,结尾必须落在剧本 end_state;各出一条 check,不符
        # 就是可定位缺陷(头段/尾段 → regenerate_segment 可修)。
        junction_lines = []
        prev_actual = str(cond.get("junction_prev_actual") or "").strip()
        end_state = str(cond.get("end_state") or "").strip()
        if prev_actual:
            junction_lines.append(
                f'The previous shot ACTUALLY ended in this state: '
                f'"{prev_actual}". This shot must OPEN by continuing that '
                f'exact state (position AND motion) — add one check for it.')
        if end_state:
            junction_lines.append(
                f'The script requires this shot to END in this state: '
                f'"{end_state}". Judge the FINAL moment against it (moving '
                f'vs at rest matters) — add one check for it.')
        # 运镜衔接(2026-07-30 用户令):镜头自身也是"运动物体"——
        # 实况/交接棒里提到 camera 时,追加一条"开场运镜是否延续"检查。
        if "camera" in (prev_actual + " " + end_state).lower():
            junction_lines.append(
                "CAMERA CONTINUITY: the states above include the camera's "
                "own motion. Add one check: does this shot OPEN by "
                "continuing that camera state (same move direction and "
                "pace, or a settled camera)? A REVERSED camera direction "
                "at the cut (e.g. push-in ending, pull-back opening) is a "
                "failed check.")
        # 音画同步(2026-07-30 用户令):有台词的镜,评审必须【听】——
        # 台词内容 / 口型同步 / 人声之外是否干净(生成时压制了背景音,
        # 这里正好验证压制话术是否生效;BGM 由后期统一混)。
        dialogue = str(cond.get("dialogue") or "").strip()
        if dialogue:
            junction_lines.append(
                f'This shot has scripted DIALOGUE: "{dialogue}". The video '
                "part includes its AUDIO track — listen to it. Add three "
                "checks: (1) the line is audibly spoken and matches the "
                "script; (2) the speaker's mouth movement is synchronized "
                "with the speech (lip sync); (3) apart from the voice the "
                "audio is clean — no background music or ambience (they "
                "are suppressed at generation and mixed in later; hearing "
                "them here is a defect).")
        # 跨镜一致性(2026-07-17 审计):官方外观描述符 = 一致性标尺。
        # 只对本镜画面里出现的角色判;每个出场角色出一条 check。
        cast = cond.get("cast") or {}
        if isinstance(cast, dict) and cast:
            canon = "; ".join(f'{k}: "{v}"' for k, v in cast.items())
            junction_lines.append(
                "CANONICAL CAST (the movie-wide appearance contract — every "
                f"shot must match it): {canon}. For each cast member VISIBLE "
                "in this shot, add one check: does its appearance match the "
                "canonical descriptor (species/build, coat/wardrobe colors, "
                "distinctive marks)? A drifted look is an issue "
                "(category=consistency-worthy → use category \"semantic\").")
        setting_c = str(cond.get("setting") or "").strip()
        if setting_c:
            junction_lines.append(
                f'CANONICAL SETTING: "{setting_c}". If this shot is in the '
                "same scene, its set dressing and lighting must match — add "
                "one check.")
        instruction = (_SHOT_REVIEW_INSTRUCTION + lang_clause(
            "all free-text fields (question/problem/reason/"
            "suggestion/summary)")).format(
            shot_prompt=spec.prompt,
            gen_prompt_block=(f'\nThe exact generation prompt was:\n"{gen_prompt}"'
                              if gen_prompt and gen_prompt != spec.prompt else ""),
            conditions_block=("Conditions provided above: "
                              + "; ".join(cond_labels) if cond_labels
                              else "No visual conditions were provided — judge "
                                   "against the text alone."),
            junction_block=("\n" + "\n".join(junction_lines)
                            if junction_lines else ""),
        )
        reply = self._generate(video + cond_parts + [{"text": instruction}])
        data = _extract_json(reply) if reply else None
        if not isinstance(data, dict):
            if reply is not None:
                log.warning("GeminiVLM review_shot: unparseable reply — no "
                            "verdict (reply=%.160r)", reply)
            return None
        pkg = self._package_review(data, fps_eff)
        self._review_cache[key] = pkg
        return pkg

    @staticmethod
    def _package_review(data: dict, fps: float) -> dict:
        """Parsed reviewer JSON → pipeline objects. Every failed check /
        localized issue becomes a ChecklistItem or PhysicsVerdict with the
        seconds→frames conversion applied — the DefectReport/repair machinery
        downstream is unchanged."""
        issues = [i for i in (data.get("issues") or []) if isinstance(i, dict)]
        checks = [c for c in (data.get("checks") or []) if isinstance(c, dict)]

        def _range(issue) -> tuple[int, int]:
            try:
                lo = max(0.0, float(issue.get("time_start_s", 0.0)))
                hi = max(lo, float(issue.get("time_end_s", lo)))
            except (TypeError, ValueError):
                lo, hi = 0.0, 0.0
            return (int(round(lo * fps)), max(int(round(lo * fps)) + 1,
                                              int(round(hi * fps))))

        mode_map = {m.value: m for m in PhysFailureMode}
        items: list[ChecklistItem] = []
        verdicts: list[PhysicsVerdict] = []
        used_issue_idx: set[int] = set()

        for ci, c in enumerate(checks):
            q = str(c.get("question", "")).strip() or f"check #{ci}"
            passed = bool(c.get("passed", False))
            fr = None
            fix = ""
            if not passed:
                # the issue explaining this failed check (exact link via
                # check_ref; the prompt requires it for every failure)
                match = next(
                    (k for k, it in enumerate(issues)
                     if int(it.get("check_ref", -1)) == ci), None)
                if match is not None:
                    used_issue_idx.add(match)
                    it = issues[match]
                    fr = _range(it)
                    fix = str(it.get("suggestion") or it.get("problem") or "")
            items.append(ChecklistItem(question=q, kind="semantic",
                                       passed=passed, fix_instruction=fix,
                                       frame_range=fr))

        for k, it in enumerate(issues):
            cat = str(it.get("category", "")).lower()
            try:
                sev = min(1.0, max(0.0, float(it.get("severity", 0.5))))
            except (TypeError, ValueError):
                sev = 0.5
            hint = str(it.get("suggestion") or it.get("problem") or "")
            if cat == "physics":
                mode = mode_map.get(str(it.get("physics_mode", "")).lower(),
                                    PhysFailureMode.UNEXPLAINED)
                verdicts.append(PhysicsVerdict(
                    mode=mode, frame_range=_range(it), severity=sev,
                    suggested_intervention=hint, source="vlm",
                    entity=str(it.get("entity", "") or "")))
            elif k not in used_issue_idx:
                # localized non-physics issue with no matching check —
                # still surfaced (never dropped) as a failed item
                items.append(ChecklistItem(
                    question=str(it.get("problem", "issue"))[:120],
                    kind="semantic", passed=False, fix_instruction=hint,
                    frame_range=_range(it)))
        return {"items": items, "verdicts": verdicts,
                "summary": str(data.get("summary", "")).strip()}

    # ── critic contracts: slice the ONE cached review ──
    def assess_semantic(self, clip: CandidateClip, spec: ShotSpec) -> list[tuple]:
        pkg = self.review_shot(clip, spec)
        if pkg is None:
            return []
        out = []
        for it in pkg["items"]:
            if it.frame_range is not None:
                out.append((it.question, it.passed, it.fix_instruction,
                            it.frame_range))
            else:
                out.append((it.question, it.passed, it.fix_instruction))
        return out

    def assess_physics(self, clip: CandidateClip, spec: ShotSpec,
                       fps: int) -> list[PhysicsVerdict]:
        pkg = self.review_shot(clip, spec, fps=fps)
        return list(pkg["verdicts"]) if pkg else []

    # ── tournament pick (bidirectional de-biasing is the CALLER's job) ──
    def compare(self, a: CandidateClip, b: CandidateClip, spec: ShotSpec) -> int:
        from ..physics.track_extractor_backends import _looks_like_video

        if not (_looks_like_video(Path(str(a.video_path)))
                and _looks_like_video(Path(str(b.video_path)))):
            return BaseMLLMClient.compare(self, a, b, spec)   # 桩→指标回退
        va = self._video_part("Video 1", a.video_path)
        vb = self._video_part("Video 2", b.video_path)
        if not va or not vb:
            return BaseMLLMClient.compare(self, a, b, spec)
        reply = self._generate(va + vb + [{"text": (
            "Both videos attempt the same shot:\n"
            f'"{spec.prompt}"\n\n'
            "Which matches the shot better overall (semantics, physics, "
            "temporal consistency, visual quality)? Reply with ONLY a JSON "
            'object: {"better": 1 | 2 | 0} — 0 means no clear difference. '
            "Be conservative."
        )}])
        data = _extract_json(reply) if reply else None
        try:
            pick = int((data or {}).get("better", 0))
        except (TypeError, ValueError):
            pick = 0
        return 1 if pick == 1 else -1 if pick == 2 else 0

    # ── the Verifier's blind A/B (NEWTON verify_relative + our additions) ──
    def verify_pair(self, candidate_path, baseline_path, shot_prompt: str,
                    repair_context: Optional[dict] = None,
                    seed: Optional[int] = None) -> Optional[dict]:
        """Blind A/B: REPAIRED candidate vs the CURRENT BEST (original).

        NEWTON mechanics kept 1:1 — randomized slots labeled only Video 1/2,
        judge never told which is repaired, signed score is how much better
        Video 2 is, remapped to candidate-vs-baseline afterwards.

        Our additions over NEWTON:
          • FOUR per-dimension signed scores (semantic / physics / temporal /
            visual — user ruling: dims 1,2,4,5; condition adherence belongs
            to the REVIEWER stage, not the pair judge) instead of one blob —
            enables the dimension NON-REGRESSION guard in VerifierAgent
            ("fixed one thing, badly broke another" is auto-reject);
          • a defect-presence probe: the judge is told WHAT the repair tried
            to fix and reports whether each video exhibits that defect →
            target_fixed for the candidate, independent of the overall score.
        Returns the remapped verdict dict, or None (transport/inline failure —
        the caller falls back to the metric gate, loudly)."""
        import random

        from ..physics.track_extractor_backends import _looks_like_video

        if not (_looks_like_video(Path(str(candidate_path)))
                and _looks_like_video(Path(str(baseline_path)))):
            return None                     # 桩视频 → 指标闸兜底(诚实)
        rng = random.Random(seed)
        candidate_is_v2 = rng.random() < 0.5
        v1, v2 = ((baseline_path, candidate_path) if candidate_is_v2
                  else (candidate_path, baseline_path))
        p1 = self._video_part("Video 1", v1)
        p2 = self._video_part("Video 2", v2)
        if not p1 or not p2:
            return None
        ctx = repair_context or {}
        target = ""
        td = ctx.get("target_defect") or {}
        if td:
            target = (f'The repair attempted to fix this defect: '
                      f'"{td.get("note", "")} — {td.get("fix_hint", "")}" '
                      f'(around {td.get("time_range_s", "?")} s, entity: '
                      f'"{td.get("entity", "")}").')
        reply = self._generate(p1 + p2 + [{"text":
            (_VERIFY_PAIR_INSTRUCTION + lang_clause(
                "all notes and issue texts")).format(shot_prompt=shot_prompt,
                                            target_block=target)}])
        data = _extract_json(reply) if reply else None
        if not isinstance(data, dict):
            return None

        def _i(v, lo=-10, hi=10):
            try:
                return max(lo, min(hi, int(round(float(v)))))
            except (TypeError, ValueError):
                return 0

        sign = 1 if candidate_is_v2 else -1
        raw = _i(data.get("score", 0))
        dims_raw = data.get("dim_scores") or {}
        dims = {k: sign * _i(dims_raw.get(k, 0))
                for k in ("semantic", "physics", "temporal", "visual")}
        present = data.get("defect_present") or {}
        cand_slot = "video2" if candidate_is_v2 else "video1"
        base_slot = "video1" if candidate_is_v2 else "video2"
        score = sign * raw
        # accept = strictly better overall AND no dimension badly regressed
        # (the monotonic contract, now dimension-aware).
        conclusion = ("accept" if score >= 1 and min(dims.values()) >= -2
                      else "reject")
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        return {
            "score": score,
            "dim_scores": dims,
            "notes": {k: str((data.get("notes") or {}).get(k, "")).strip()
                      for k in ("semantic", "physics", "temporal", "visual")},
            "target_fixed": (bool(present.get(base_slot, False))
                             and not bool(present.get(cand_slot, False))
                             if td else None),
            "issues": [str(x) for x in issues] if score < 0 else [],
            "summary": str(data.get("summary", "")).strip(),
            "conclusion": conclusion,
            "_order": {"video1": "baseline" if candidate_is_v2 else "candidate",
                       "video2": "candidate" if candidate_is_v2 else "baseline",
                       "raw_v2_minus_v1": raw},
        }

    def caption_image(self, image_path) -> str:
        got = self._image_part("IMAGE", image_path)
        if not got:
            return ""
        self._require_key()
        reply = self._generate([got[1],
            {"text": "Describe this image in ONE short sentence for "
                     "retrieval: what/who it shows and the setting. Start "
                     "with one category word from [background, character, "
                     "object, style] and a colon. Example: 'background: a "
                     "cozy living room at night with a lit fireplace'. No "
                     "other text."},
        ])
        return (reply or "").strip()

    def caption_identity(self, image_path) -> str:
        """角色正典打标(颜色逐项、看不见省略)—— 原生 Gemini 通道。"""
        got = self._image_part("IMAGE", image_path)
        if not got:
            return ""
        self._require_key()
        reply = self._generate([got[1], {
            "text": _IDENTITY_CAPTION_INSTRUCTION
            + lang_clause("the sentence")}])
        return (reply or "").strip()

    def caption_video(self, video_path) -> str:
        """素材视频的【原生视频】打标(2026-07-17 裁决:入库不再抽帧 ——
        某个 shot 可能直接续用用户片段,标签必须描述整段内容(身份词 +
        场景 + 运动),不是某一帧)。假设素材段不长;超 18MB 由
        _video_part 自动 360p 转码。失败/桩文件 → ""(调用方落
        中间帧 caption → 文件名的兜底链,绝不编)。"""
        from ..physics.track_extractor_backends import _looks_like_video

        p = Path(str(video_path))
        if not p.is_file() or not _looks_like_video(p):
            return ""
        got = self._video_part("THE CLIP", p)
        if not got:
            return ""
        self._require_key()
        reply = self._generate(got + [{"text":
            "Describe this video clip in ONE or TWO short sentences for "
            "retrieval and script planning: who/what it shows (identity "
            "words — species/build, coat/wardrobe with colors, distinctive "
            "marks), the setting, and the main motion/camera movement. "
            "Start with one category word from [character, object, "
            "background, style] and a colon. Example: 'character: a young "
            "man in a bright red jacket with a black backpack walks "
            "steadily forward along a seaside boardwalk in golden "
            "afternoon light, camera tracking from behind'. No other "
            "text."}])
        return (reply or "").strip()

    def describe_junction_video(self, video_path, portraits=None) -> str:
        """接点实况·视频版(2026-07-28 用户裁决):看上一镜的【末尾片段】
        —— 单帧只能靠运动模糊猜"动没动",片段才有真实的运动信息(方向/
        快慢/是否仍在动)。失败/桩文件 → ""(调用方回退单帧,绝不编)。
        具名矢量(2026-08-05 用户令):肖像随片发,VLM 按肖像认人 ——
        矢量的 who 直接是角色名,下游绑定不再靠写手猜(shot2 口型
        张冠李戴的根治)。"""
        from ..physics.track_extractor_backends import _looks_like_video

        p = Path(str(video_path))
        if not p.is_file() or not _looks_like_video(p):
            return ""
        got = self._video_part("THE FINAL SECONDS OF THE PREVIOUS SHOT", p)
        if not got:
            return ""
        parts = list(got)
        for name, ppath in sorted((portraits or {}).items()):
            ip = self._image_part(f"OFFICIAL PORTRAIT of {name}", ppath)
            if ip:
                parts += list(ip)
        self._require_key()
        reply = self._generate(parts + [{
            "text": _JUNCTION_VIDEO_INSTRUCTION
            + lang_clause("who/pose/position/direction values")}])
        return (reply or "").strip()

    def describe_junction(self, image_path) -> str:
        """接点实况(2026-07-15 需求 ②):看上一镜的【真实尾帧】,一句话
        说清续接状态 —— 每个关键主体在哪、看起来是动是停、朝什么方向。
        失败返回 ""(调用方诚实跳过,绝不编)。"""
        got = self._image_part("FINAL FRAME", image_path)
        if not got:
            return ""
        self._require_key()
        reply = self._generate([got[1], {
            "text": _JUNCTION_INSTRUCTION
            + lang_clause("who/pose/position/direction values")}])
        return (reply or "").strip()


# 接点实况的共用指令(GeminiVLM 与 LocalQwenVLM 同一份 —— 两种模式同语义)
# 片尾理解报告(2026-08-07 用户令,三条件融合派):结构化出场矢量退役,
# 换两部分理解 —— camera_angle(机位视角)+ character_actions(人物动作)。
# enhancer 拿它润色下一镜开场的承接;具名法保留(2026-08-05:肖像随片
# 发,who 必须是角色名 —— 下游绑定不靠写手猜)。
_JUNCTION_VIDEO_INSTRUCTION = (
    "This is the FINAL few seconds of a video shot, followed by the "
    "cast's OFFICIAL PORTRAITS, each labeled with the character's "
    "name. Report the shot's ENDING STATE in two parts, judged from "
    "the actual motion across the clip (not from blur). For each "
    "person, MATCH them against the portraits: when a person matches "
    "a portrait, `who` MUST be that exact character name (copy the "
    "label verbatim); only a person matching no portrait gets a short "
    "visual handle. STRICT JSON only:\n"
    '{"camera_angle": "<one clause: framing (wide/medium/close-up), '
    'camera height/angle, where the camera sits relative to the '
    'subjects, and its motion state at the last moment>", '
    '"character_actions": [{"who": "<character name>", "position": '
    '"<left/center/right + near/far>", "action": "<one clause: what '
    'they are visibly doing at the last moment — pose, motion or '
    'stillness, direction>"}]}\n'
    "Only what is visible — no speculation. No other text."
)

_JUNCTION_INSTRUCTION = (
    "This is the FINAL frame of a video shot. Report the shot's "
    "ENDING STATE for continuity into the next shot — motion judged "
    "from blur and posture (single-frame fallback; be honest about "
    "uncertainty). STRICT JSON only, same schema:\n"
    '{"camera_angle": "<one clause: framing, camera height/angle, '
    'position relative to subjects, motion state>", '
    '"character_actions": [{"who": "<name or short handle>", '
    '"position": "…", "action": "…"}]}\n'
    "Only what is visible — no speculation. No other text."
)


class LocalQwenVLM(BaseMLLMClient):
    """本地加载的 Qwen2.5-VL(transformers)——用户裁决(2026-07-15 需求
    ②):接点实况/素材打标的第二种 VLM 模式(第一种 = GeminiVLM API)。
    `models.mllm.name: "qwen-local"` 即切换。

    职责范围(诚实):图片级 —— caption_image(素材打标)与
    describe_junction(接点实况)。整镜评审需要原生视频输入,仍归
    GeminiVLM;本类的 assess_* 返回 [] 并警告一次,绝不冒充评审员。

    config(models.mllm):
      name: "qwen-local"
      model: "Qwen/Qwen2.5-VL-7B-Instruct"   # 或本地权重路径
      device: "cuda"
      max_new_tokens: 96
    """

    def __init__(self, name: str = "qwen-local", config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.model_id = self.config.get("model",
                                        "Qwen/Qwen2.5-VL-7B-Instruct")
        self.device = self.config.get("device", "cuda")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 96))
        self._model = None
        self._processor = None
        self._warned_review = False

    def _ensure(self) -> None:
        """惰性加载(构造零开销);缺 torch/transformers → 响亮 RuntimeError,
        绝不静默装死。"""
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration \
                    as _QwenVL
            except ImportError:          # 老/新 transformers 的命名差异
                from transformers import AutoModelForVision2Seq as _QwenVL
        except ImportError as exc:
            raise RuntimeError(
                f"LocalQwenVLM('{self.name}') needs torch+transformers for "
                f"local {self.model_id}: pip install 'torch' 'transformers' "
                f"'qwen-vl-utils' — or switch models.mllm.name to 'gemini' "
                f"(API mode). Import failed: {exc}"
            ) from exc
        log.info("LocalQwenVLM loading %s on %s …", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = _QwenVL.from_pretrained(
            self.model_id, torch_dtype="auto", device_map=self.device)

    def _chat_image(self, image_path, instruction: str) -> str:
        """一图一指令 → 一句回复;文件不可读返回 ""(诚实跳过)。"""
        p = Path(str(image_path))
        if not p.is_file() or p.suffix.lower() not in (
            ".png", ".jpg", ".jpeg", ".webp", ".bmp"
        ):
            return ""
        self._ensure()
        try:
            from PIL import Image
            img = Image.open(p).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instruction}]}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[img],
                                     return_tensors="pt").to(self._model.device)
            out = self._model.generate(**inputs,
                                       max_new_tokens=self.max_new_tokens)
            reply = self._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
            return (reply or "").strip()
        except Exception as exc:
            log.warning("LocalQwenVLM inference failed (%s): %s", p.name, exc)
            return ""

    def caption_image(self, image_path) -> str:
        return self._chat_image(image_path, (
            "Describe this image in ONE short sentence for retrieval: "
            "what/who it shows and the setting. Start with one category "
            "word from [background, character, object, style] and a colon. "
            "No other text."))

    def caption_identity(self, image_path) -> str:
        return self._chat_image(image_path, _IDENTITY_CAPTION_INSTRUCTION
                                + lang_clause("the sentence"))

    def describe_junction(self, image_path) -> str:
        return self._chat_image(image_path, _JUNCTION_INSTRUCTION)

    # 评审职责不归本类 —— 诚实沉默(警告一次),绝不伪造判定。
    def _warn_review(self) -> None:
        if not self._warned_review:
            self._warned_review = True
            log.warning("LocalQwenVLM supports captioning/junction only — "
                        "shot review needs the native-video judge "
                        "(models.mllm.name: 'gemini'); emitting no verdicts")

    def assess_semantic(self, clip, spec):
        self._warn_review()
        return []

    def assess_physics(self, clip, spec, fps):
        self._warn_review()
        return []


_REGISTRY = {
    "gpt-4o": OpenAICompatVLM,
    "openai-vlm": OpenAICompatVLM,
    "openai": OpenAICompatVLM,
    "qwen-vl": OpenAICompatVLM,
    "qwen": OpenAICompatVLM,
    "gemini": GeminiVLM,
    "qwen-local": LocalQwenVLM,
    "local-qwen": LocalQwenVLM,
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
