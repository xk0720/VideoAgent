"""Benchmark module smoke tests."""
from __future__ import annotations

import pytest

from benchmark import (
    CineBenchAdapter,
    MashupBenchAdapter,
    aggregate_script_metrics,
)
from longvideoagent.types import EditingScript, EditingSegment


def test_aggregate_handles_empty_script():
    out = aggregate_script_metrics(EditingScript())
    assert set(out) == {"m1", "m2", "m3", "m4", "m5", "m6"}
    assert all(v == 0.0 for v in out.values())


def test_aggregate_mixes_retrieval_and_generation():
    s1 = EditingSegment(segment_idx=0, source="retrieval", duration=1.0,
                        metric_scores={f"m{i}": 0.8 for i in range(1, 7)})
    s2 = EditingSegment(segment_idx=1, source="generation", duration=1.0)
    out = aggregate_script_metrics(EditingScript(segments=[s1, s2]), generation_prior=0.5)
    # Retrieval gives 0.8, generation gives 0.5 → mean ≈ 0.65 for each metric.
    for v in out.values():
        assert abs(v - 0.65) < 1e-3


def test_adapters_raise_until_v0_2():
    with pytest.raises(NotImplementedError):
        list(MashupBenchAdapter("/tmp").load_cases())
    with pytest.raises(NotImplementedError):
        list(CineBenchAdapter("/tmp").load_cases())
