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


# ─────────────────────────────────────────────────────────────────────
# Arc-level judge (CSA framework, R16) — see docs/CSA_FRAMEWORK.md §4
#
# Operates on the *whole* EditingScript, not on individual segments. This
# is the minimum implementation of the Arc judge slot — a heuristic that
# is good enough to:
#   (a) produce different scores for the same script under a permutation
#       (i.e. C4 falsification holds; ordering matters);
#   (b) compose with the existing m1..m6 (which judge at the Cut scale);
#   (c) be replaceable by a long-context MLLM (Qwen3-VL / InternVL3) when
#       real backends arrive.
# ─────────────────────────────────────────────────────────────────────


def arc_coherence(script, arc_context=None) -> dict[str, float]:
    """Compute Arc-level (whole-script) coherence sub-scores.

    Returns a dict with these keys (each in [0, 1]):

    * ``arc_progression``     — does the sequence of validator scores trace
                                 the intended arc shape (rise / climax /
                                 resolution)? Falls back to checking that
                                 scores aren't a flat line if no
                                 intended_arc is given.
    * ``arc_energy_match``    — does the script's m6 trajectory match the
                                 intended ``energy_curve``? Defaults to a
                                 trivial 0.5 if no curve given.
    * ``arc_character_cover`` — fraction of expected_characters that
                                 actually appear in any retrieval segment.
                                 1.0 if no expectation set.
    * ``arc_continuity``      — fraction of adjacent segments that satisfy
                                 m2 (segment consistency) above a threshold.
                                 This is the only sub-score that's
                                 *order-sensitive* — it's the C4
                                 falsification handle.
    * ``arc_overall``         — weighted mean of the four above.

    Reference: docs/CSA_FRAMEWORK.md §1 challenge C4. This is a heuristic
    placeholder for what a real long-context MLLM judge will eventually
    do. The heuristic is deliberately rule-based so it cannot drift with
    LLM stochasticity; it produces stable numbers for tests.
    """
    from ..types import ArcContext, EditingScript

    if not isinstance(script, EditingScript) or not script.segments:
        return {"arc_progression": 0.5, "arc_energy_match": 0.5,
                "arc_character_cover": 1.0, "arc_continuity": 0.5,
                "arc_overall": 0.5}

    segs = script.segments
    n = len(segs)

    # ── arc_progression ──────────────────────────────────────────────
    vals = [float((s.metric_scores or {}).get("validator", 5.0)) for s in segs]
    intended = (arc_context.intended_arc if arc_context else []) or []
    if intended:
        # Map intended arc names to expected normalized energy levels.
        weights = {
            "setup": 0.4, "rising": 0.6, "climax": 0.95,
            "falling": 0.6, "resolution": 0.5,
            "build": 0.5, "drop": 0.95, "outro": 0.4,
            "verse": 0.5, "chorus": 0.9, "intro": 0.4, "bridge": 0.65,
        }
        # Sample the expected trajectory at len(segs) points.
        if len(intended) == 1:
            target = [weights.get(intended[0], 0.5)] * n
        else:
            ts_int = [i / max(1, len(intended) - 1) for i in range(len(intended))]
            ts_seg = [i / max(1, n - 1) for i in range(n)]
            target: list[float] = []
            for t in ts_seg:
                # Piecewise linear over the intended arc.
                k = 0
                while k + 1 < len(ts_int) and ts_int[k + 1] < t:
                    k += 1
                if k + 1 >= len(ts_int):
                    target.append(weights.get(intended[-1], 0.5))
                else:
                    a, b = weights.get(intended[k], 0.5), weights.get(intended[k + 1], 0.5)
                    frac = (t - ts_int[k]) / max(1e-9, ts_int[k + 1] - ts_int[k])
                    target.append(a * (1 - frac) + b * frac)
        # Normalise validator scores to [0, 1] (they're on a 1..10 scale).
        observed = [max(0.0, min(1.0, (v - 1.0) / 9.0)) for v in vals]
        diffs = [abs(o - t) for o, t in zip(observed, target)]
        arc_progression = 1.0 - sum(diffs) / max(1, len(diffs))
    else:
        # No intended arc — fall back to "is the validator trajectory not
        # constant?". A flat trajectory means the script makes no shape.
        if n <= 1:
            arc_progression = 0.5
        else:
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            # Map variance ∈ [0, 16] (validator ∈ [1,10] so max var ≈ 20) to [0, 1].
            arc_progression = float(max(0.0, min(1.0, var / 8.0)))

    # ── arc_energy_match ──────────────────────────────────────────────
    if arc_context and arc_context.energy_curve:
        # Compare segment m6 (energy correspondence with music) to the
        # piecewise-linear energy curve at each segment's normalised time.
        m6s = [float((s.metric_scores or {}).get("m6", 0.5)) for s in segs]
        diffs2 = []
        for i, m6 in enumerate(m6s):
            ts = i / max(1, n - 1)
            target_e = _piecewise_interp(arc_context.energy_curve, ts)
            diffs2.append(abs(m6 - target_e))
        arc_energy_match = 1.0 - sum(diffs2) / max(1, len(diffs2))
    else:
        arc_energy_match = 0.5

    # ── arc_character_cover ──────────────────────────────────────────
    expected = set(arc_context.expected_characters) if arc_context else set()
    if not expected:
        arc_character_cover = 1.0
    else:
        # We only see character_ids on retrieval segments (mock generation
        # doesn't claim a character). A real-backend version would parse
        # generated frames.
        seen: set[str] = set()
        for s in segs:
            if s.source == "retrieval":
                # EditingSegment doesn't carry character_ids directly; the
                # heuristic uses shot_ids as a proxy for distinct identities.
                # The real version would look up Shot.character_ids in memory.
                for sid in s.shot_ids:
                    seen.add(sid)
        # Tolerant overlap: any expected character whose id substring shows
        # up in any seen shot_id counts as covered.
        covered = sum(
            1 for c in expected
            if any(c in sid for sid in seen)
        )
        arc_character_cover = covered / max(1, len(expected))

    # ── arc_continuity (order-sensitive — C4 falsification handle) ────
    if n <= 1:
        arc_continuity = 1.0
    else:
        m2s = [float((s.metric_scores or {}).get("m2", 0.5)) for s in segs[1:]]
        # Threshold-fraction: how many adjacencies have m2 > 0.5?
        arc_continuity = sum(1 for x in m2s if x > 0.5) / len(m2s)

    # ── arc_overall ──────────────────────────────────────────────────
    overall = 0.35 * arc_progression \
              + 0.20 * arc_energy_match \
              + 0.15 * arc_character_cover \
              + 0.30 * arc_continuity

    return {
        "arc_progression": float(max(0.0, min(1.0, arc_progression))),
        "arc_energy_match": float(max(0.0, min(1.0, arc_energy_match))),
        "arc_character_cover": float(max(0.0, min(1.0, arc_character_cover))),
        "arc_continuity": float(max(0.0, min(1.0, arc_continuity))),
        "arc_overall": float(max(0.0, min(1.0, overall))),
    }


def _piecewise_interp(curve: list[tuple[float, float]], t: float) -> float:
    """Piecewise-linear interpolation of a (time ∈ [0,1], energy) curve at t."""
    if not curve:
        return 0.5
    pts = sorted(curve, key=lambda x: x[0])
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= t <= x1:
            frac = (t - x0) / max(1e-9, x1 - x0)
            return y0 * (1 - frac) + y1 * frac
    return pts[-1][1]


__all__ = [
    "MetricTool",
    "m1_prompt_relevance", "m2_segment_consistency", "m3_motion_continuity",
    "m4_framing", "m5_beat_sync", "m6_energy_correspondence",
    "arc_coherence",
]
