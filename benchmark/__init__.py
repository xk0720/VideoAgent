"""Evaluation harness.

The DIRECT m1-m6 metrics live in ``longvideoagent.tools.metric_tool`` and are
re-exported here for benchmark-style aggregate scoring.
v0.1 stubs the Mashup-Bench / CineBench adapters — they raise NotImplementedError
on data load until weights/datasets are wired up in v0.2.
"""
from .metrics import aggregate_script_metrics
from .mashup_bench import MashupBenchAdapter
from .cine_bench import CineBenchAdapter

__all__ = ["aggregate_script_metrics", "MashupBenchAdapter", "CineBenchAdapter"]
