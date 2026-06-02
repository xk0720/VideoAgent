"""Agent-callable tools.

v0.2.2 onward: tools self-describe via `BaseTool.spec` and register with the
in-process `ToolRegistry` (UniVA MCP-style discovery, without the wire protocol).
Categories: analysis / generation / editing / tracking / physics / metric / retrieval.
"""
from .base import BaseTool, ToolCategory, ToolRegistry, ToolSpec, default_registry
from .metric_tool import MetricTool
from .assembly_tool import AssemblyTool
from .retrieval_tool import RetrievalTool
from .video_probe import VideoProbeTool
from .frame_extract import FrameExtractTool
from .video_concat import VideoConcatTool
from .image_ops import ImageOpsTool
from .captioning import CaptioningTool
from .detection import DetectionTool
from .audio_gen import AudioGenTool

__all__ = [
    "BaseTool", "ToolCategory", "ToolRegistry", "ToolSpec", "default_registry",
    "MetricTool", "AssemblyTool", "RetrievalTool",
    "VideoProbeTool", "FrameExtractTool", "VideoConcatTool", "ImageOpsTool",
    "CaptioningTool", "DetectionTool", "AudioGenTool",
]
