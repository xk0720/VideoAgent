"""Aggregate-metric helpers for benchmark runs.

Open-source dependencies: numpy.

Per-script aggregation strategy mirrors DIRECT supplementary: arithmetic
mean of m1..m6 across all retrieved segments. Generation segments
contribute a fixed prior (configurable) since some metrics are
ill-defined for synthesised footage.
"""
from __future__ import annotations

import numpy as np

from longvideoagent.types import EditingScript


def aggregate_script_metrics(
    script: EditingScript,
    generation_prior: float = 0.5,
) -> dict[str, float]:
    keys = ["m1", "m2", "m3", "m4", "m5", "m6"]
    rows = []
    for seg in script.segments:
        if seg.source == "retrieval":
            rows.append([float(seg.metric_scores.get(k, 0.0)) for k in keys])
        else:
            rows.append([generation_prior] * len(keys))
    if not rows:
        return {k: 0.0 for k in keys}
    arr = np.asarray(rows, dtype="float32")
    return {k: float(arr[:, i].mean()) for i, k in enumerate(keys)}


__all__ = ["aggregate_script_metrics"]
