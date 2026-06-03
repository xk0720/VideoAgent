"""Deterministic text 'embedding' for v0.2.2 (no CLIP needed, CPU-only).

Hash-based bag-of-tokens vector with cosine similarity. Swap for real CLIP /
text encoders behind the same `embed_text` / `cosine` interface in v0.3.

Tokenization mixes two strategies so the same embedder serves both English
and CJK prompts (Maestro is bilingual by design — see prompts in
`physics/failure_modes.py:FAILURE_MODE_KEYWORDS`):
  • ASCII / Latin runs   → word-level via `\\w+` (e.g. "ball thrown")
  • CJK Han / Hiragana / Katakana / Hangul → per-CHARACTER tokens
    (e.g. "水从杯子里倒出来" → ['水','从','杯','子','里','倒','出','来'])

Without the second case, an entire Chinese prompt collapses to one token →
its cosine to every other Chinese prompt is 0 unless they share the literal
string → C4 LessonLibrary retrieval is dead for CJK users. Splitting per
character recovers approximate word-overlap semantics (Chinese morphemes are
~1 char each) and keeps the embedder deterministic + CPU-only.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

DIM = 256

# Unicode CJK ranges we treat as per-character tokens. We use `unicodedata.name`
# bucket prefixes rather than hand-rolled ranges so emoji etc. fall through.
_CJK_PREFIXES = ("CJK UNIFIED", "CJK COMPATIBILITY", "HIRAGANA", "KATAKANA",
                 "HANGUL")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _is_cjk(ch: str) -> bool:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return any(name.startswith(p) for p in _CJK_PREFIXES)


def _tokenize(text: str) -> list[str]:
    """Mixed Latin-word / CJK-char tokenizer. Order-independent (BoW)."""
    text = (text or "").lower()
    tokens: list[str] = []
    # Pull out CJK characters first so they don't get swallowed by `\w+`.
    cjk_chars = [ch for ch in text if _is_cjk(ch)]
    tokens.extend(cjk_chars)
    # Remove CJK chars from the Latin-pass text so Latin tokens aren't
    # contaminated by stray CJK that `\w+` would otherwise capture under
    # Unicode default flags.
    stripped = "".join(" " if _is_cjk(ch) else ch for ch in text)
    tokens.extend(_LATIN_TOKEN_RE.findall(stripped))
    return tokens


def embed_text(text: str, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
