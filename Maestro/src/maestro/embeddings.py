"""Deterministic text 'embedding' for v0.1 (no CLIP needed, CPU-only).

Hash-based bag-of-tokens vector with cosine similarity. Swap for real CLIP/text
encoders behind the same `embed_text` / `cosine` interface in v0.2.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

DIM = 256


def embed_text(text: str, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", (text or "").lower())
    for tok in tokens:
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
