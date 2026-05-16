"""Tool layer (called by agents).

Four tools, all subclassing BaseTool:
    RetrievalTool     — beam-search + 6 metrics (DIRECT §4.3 + supplementary)
    GenerationTool    — wrapper around BaseVideoGenClient
    AssemblyTool      — real ffmpeg concat (always real, never mocked)
    MetricTool        — exposes the 6 metrics individually for agent inspection
"""
from .base import BaseTool
from .retrieval_tool import RetrievalTool
from .generation_tool import GenerationTool
from .assembly_tool import AssemblyTool
from .metric_tool import MetricTool

__all__ = ["BaseTool", "RetrievalTool", "GenerationTool", "AssemblyTool", "MetricTool"]
