"""The six retrieval metrics, exposed as a standalone tool.

Open-source dependencies:
    • numpy  (always)
    • scipy  (preferred — uses scipy.stats.wasserstein_distance + spearmanr)
              Falls back to pure-numpy implementations of both when scipy is
              not installed, so the metric tool is operational in the v0.1
              base environment.

Definitions follow DIRECT supplementary (and the design doc §7.1):
    m1 prompt_relevance      cosine(CLIP(shot), CLIP(prompt))
    m2 segment_consistency   cosine(CLIP(prev), CLIP(curr))
    m3 motion_continuity     flow-magnitude × flow-direction similarity
    m4 framing               1 - Wasserstein(saliency_prev_end, saliency_curr_start)
    m5 beat_sync             exp(-||cut_time - nearest_beat||^2 / 2σ^2)
    m6 energy_correspondence Spearman(flow_mag_per_shot, music_RMS_per_shot)
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:
    from scipy.stats import spearmanr, wasserstein_distance       # type: ignore
    _HAS_SCIPY = True
except ImportError:                                                # pragma: no cover
    _HAS_SCIPY = False

    def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):  # type: ignore
        """Pure-numpy 1D Wasserstein-1 (CDF-difference) distance.

        We only ever pass equal-length integer supports with the same indexing
        in this codebase (see m4_framing), so the simpler CDF-difference
        formulation is exact.
        """
        u = np.asarray(u_values, dtype="float64")
        v = np.asarray(v_values, dtype="float64")
        if u_weights is None:
            u_w = np.ones_like(u) / max(1, len(u))
        else:
            u_w = np.asarray(u_weights, dtype="float64")
            s = u_w.sum()
            u_w = u_w / s if s > 0 else u_w
        if v_weights is None:
            v_w = np.ones_like(v) / max(1, len(v))
        else:
            v_w = np.asarray(v_weights, dtype="float64")
            s = v_w.sum()
            v_w = v_w / s if s > 0 else v_w
        order_u = np.argsort(u); u = u[order_u]; u_w = u_w[order_u]
        order_v = np.argsort(v); v = v[order_v]; v_w = v_w[order_v]
        all_x = np.unique(np.concatenate([u, v]))
        # CDFs evaluated at all_x[:-1] (right-continuous).
        u_cdf = np.cumsum(u_w[np.searchsorted(u, all_x[:-1], side="right") - 1].clip(min=0))
        v_cdf = np.cumsum(v_w[np.searchsorted(v, all_x[:-1], side="right") - 1].clip(min=0))
        return float(np.sum(np.abs(u_cdf - v_cdf) * np.diff(all_x)))

    def spearmanr(a, b):  # type: ignore
        a = np.asarray(a, dtype="float64")
        b = np.asarray(b, dtype="float64")
        ra = a.argsort().argsort().astype("float64")
        rb = b.argsort().argsort().astype("float64")
        ra -= ra.mean(); rb -= rb.mean()
        denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
        return float(np.dot(ra, rb) / denom), float("nan")

from .base import BaseTool


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = a.flatten().astype("float32"); b = b.flatten().astype("float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def m1_prompt_relevance(shot_emb: np.ndarray, prompt_emb: np.ndarray) -> float:
    return _to_unit(cosine(shot_emb, prompt_emb))


def m2_segment_consistency(prev_emb: Optional[np.ndarray], curr_emb: np.ndarray) -> float:
    if prev_emb is None:
        return 1.0  # first shot has no neighbour penalty
    return _to_unit(cosine(prev_emb, curr_emb))


def m3_motion_continuity(prev_end_flow: Optional[np.ndarray],
                         curr_start_flow: Optional[np.ndarray]) -> float:
    if prev_end_flow is None or curr_start_flow is None:
        return 0.5
    mag_prev = np.linalg.norm(prev_end_flow.reshape(-1, 2), axis=1)
    mag_curr = np.linalg.norm(curr_start_flow.reshape(-1, 2), axis=1)
    mag_sim = 1.0 - np.tanh(abs(mag_prev.mean() - mag_curr.mean()))
    dir_sim = cosine(prev_end_flow.mean(axis=(0, 1)), curr_start_flow.mean(axis=(0, 1)))
    return float(0.5 * mag_sim + 0.5 * _to_unit(dir_sim))


def m4_framing(prev_end_saliency: Optional[np.ndarray],
               curr_start_saliency: Optional[np.ndarray]) -> float:
    if prev_end_saliency is None or curr_start_saliency is None:
        return 0.5
    pa = prev_end_saliency.flatten().astype("float64") + 1e-9
    pb = curr_start_saliency.flatten().astype("float64") + 1e-9
    if pa.size != pb.size:
        # Reshape both to the smaller size via simple block average; in v0.1
        # saliency maps are produced by the same extractor so this branch is
        # essentially dead, but we keep it for robustness.
        n = min(pa.size, pb.size)
        pa = pa[:n]; pb = pb[:n]
    pa = pa / pa.sum(); pb = pb / pb.sum()
    # Closed-form W1 on the same integer support: sum_i |CDF_p[i] - CDF_q[i]|.
    cdf_a = np.cumsum(pa); cdf_b = np.cumsum(pb)
    w = float(np.sum(np.abs(cdf_a - cdf_b)))
    return float(max(0.0, 1.0 - w / max(1, pa.size)))


def m5_beat_sync(cut_time_s: float, beats: Iterable[float], sigma: float = 0.15) -> float:
    beats_arr = np.asarray(list(beats), dtype="float32")
    if beats_arr.size == 0:
        return 0.5
    delta = float(np.min(np.abs(beats_arr - cut_time_s)))
    return float(np.exp(-(delta ** 2) / (2 * sigma ** 2)))


def m6_energy_correspondence(flow_mags: Iterable[float], music_rms: Iterable[float]) -> float:
    fm = np.asarray(list(flow_mags), dtype="float32")
    mr = np.asarray(list(music_rms), dtype="float32")
    if fm.size < 2 or mr.size < 2 or fm.size != mr.size:
        return 0.5
    if np.std(fm) < 1e-6 or np.std(mr) < 1e-6:
        return 0.5
    corr, _ = spearmanr(fm, mr)
    if np.isnan(corr):
        return 0.5
    return float((corr + 1.0) / 2.0)


def _to_unit(x: float) -> float:
    """Map a [-1, 1] cosine to [0, 1]."""
    return float(max(0.0, min(1.0, (x + 1.0) / 2.0)))


class MetricTool(BaseTool):
    name = "compute_metrics"
    description = "Compute the six DIRECT retrieval metrics for a candidate shot."

    def run(self, **kwargs) -> dict[str, float]:
        return {
            "m1": m1_prompt_relevance(kwargs["shot_emb"], kwargs["prompt_emb"]),
            "m2": m2_segment_consistency(kwargs.get("prev_emb"), kwargs["shot_emb"]),
            "m3": m3_motion_continuity(kwargs.get("prev_end_flow"), kwargs.get("curr_start_flow")),
            "m4": m4_framing(kwargs.get("prev_end_saliency"), kwargs.get("curr_start_saliency")),
            "m5": m5_beat_sync(kwargs.get("cut_time_s", 0.0), kwargs.get("beats", [])),
            "m6": m6_energy_correspondence(kwargs.get("flow_mags", []), kwargs.get("music_rms", [])),
        }


__all__ = [
    "MetricTool",
    "m1_prompt_relevance", "m2_segment_consistency", "m3_motion_continuity",
    "m4_framing", "m5_beat_sync", "m6_energy_correspondence",
]
