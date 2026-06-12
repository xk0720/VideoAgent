"""Tool ABC + global ToolRegistry (UniVA-inspired MCP-style discovery).

Borrowed pattern (cite): **UniVA — Universal Video Agent**
  • paper: https://arxiv.org/abs/2511.08521
  • code:  https://github.com/univa-agent/univa
UniVA exposes its capabilities through *MCP tool servers* and groups them into
four practical categories (analysis / generation / editing / tracking). We do
NOT implement the MCP wire protocol (overkill for in-process v0.2.2) but we
adopt the same *self-describing tool + categorized registry* abstraction so
new tools plug in without touching agent code — the key UniVA property our
v0.1 monolithic `tools/` lacked.

Our preserved differentiation: the registry sits BELOW Maestro's agentic loop
(C2/C3/C5/C6 still own the high-level review→repair→verify orchestration). UniVA
gives us reach (more tools, uniform calling); we keep the depth (physics-first
self-improvement) on top.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# UniVA's practical tool taxonomy. `physics` and `metric` added for Maestro's
# differentiation surface — we keep them as first-class categories.
ToolCategory = Literal[
    "analysis",     # probe / caption / detect — read-only video understanding
    "generation",   # T2V / I2V / image-edit / audio — produce new media
    "editing",      # ffmpeg cut/concat/trim, PIL ops — deterministic transforms
    "tracking",     # identity / object trajectory across frames
    "physics",      # reference-free physics VERIFICATION: annotation, track
                    # extraction, law checks (Maestro C6, v0.4)
    "metric",       # quantitative scoring (Maestro C3)
    "retrieval",    # asset/lesson lookup (Maestro E1/C4)
]


@dataclass
class ToolSpec:
    """Machine-readable self-description so agents (and the ActAgent / future MCP
    wire layer) can pick the right tool by capability instead of hard-importing.
    """

    name: str
    category: ToolCategory
    description: str = ""
    # JSON-schema-like parameter spec; not validated in v0.2.2 — just declarative
    # documentation that an LLM Planner could read. Real MCP wire-up would lift
    # this directly into the protocol.
    params: dict[str, str] = field(default_factory=dict)
    # If True, calling the tool may produce side effects (write files, hit a
    # paid API). The server / smoke runner can refuse to invoke side-effecting
    # tools in sandbox mode.
    side_effects: bool = False


class BaseTool(ABC):
    """All tools self-describe via `spec` AND remain plain Python callables.

    Existing agents can keep calling `tool.run(...)` directly. The ActAgent +
    server layers use `spec` to discover what's available and what each tool
    expects.
    """

    # Override in subclasses; default fills from class attrs for back-compat
    # with v0.1 tools that only set `name`.
    name: str = "tool"
    category: ToolCategory = "analysis"
    description: str = ""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            category=self.category,
            description=self.description or self.__class__.__doc__ or "",
            params=getattr(self, "params", {}),
            side_effects=getattr(self, "side_effects", False),
        )

    @abstractmethod
    def run(self, *args, **kwargs):
        ...


class ToolRegistry:
    """In-process tool registry. UniVA's MCP server registry equivalent.

    Agents/server look up tools by name OR by category. Multiple instances may
    coexist (e.g. a sandbox registry with mocks vs a production one with real
    backends). Build the default global one with `default_registry()`.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> BaseTool:
        if tool.name in self._by_name:
            # Last-write-wins is intentional: the server can swap mock→real
            # after weights load without restart.
            pass
        self._by_name[tool.name] = tool
        return tool

    def get(self, name: str) -> BaseTool:
        if name not in self._by_name:
            raise KeyError(
                f"unknown tool '{name}'. registered: {sorted(self._by_name)}"
            )
        return self._by_name[name]

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())

    def by_category(self, category: ToolCategory) -> list[BaseTool]:
        return [t for t in self._by_name.values() if t.category == category]

    def list_specs(self) -> list[ToolSpec]:
        """Manifest for the /tools server endpoint (and for an LLM Planner)."""
        return [t.spec for t in self._by_name.values()]

    def call(self, name: str, *args, **kwargs) -> Any:
        return self.get(name).run(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Global default registry — populated lazily so importing a single tool module
# during tests does not pull in the entire universe (and any optional deps it
# may carry, such as PIL / ffprobe).
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def default_registry() -> ToolRegistry:
    """Return (and populate on first call) the in-process default registry.

    Tools that need only stdlib are always registered. Tools that need optional
    deps register themselves but fail at *call* time, not import time, so the
    pipeline degrades gracefully if e.g. ffmpeg or PIL is missing.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY
    reg = ToolRegistry()
    # Local imports to avoid circulars; each tool registers itself.
    from .metric_tool import MetricTool
    from .assembly_tool import AssemblyTool
    from .video_probe import VideoProbeTool
    from .frame_extract import FrameExtractTool
    from .video_concat import VideoConcatTool
    from .image_ops import ImageOpsTool
    from .captioning import CaptioningTool
    from .detection import DetectionTool
    from .audio_gen import AudioGenTool

    reg.register(MetricTool())
    reg.register(AssemblyTool())
    reg.register(VideoProbeTool())
    reg.register(FrameExtractTool())
    reg.register(VideoConcatTool())
    reg.register(ImageOpsTool())
    reg.register(CaptioningTool())
    reg.register(DetectionTool())
    reg.register(AudioGenTool())
    _DEFAULT_REGISTRY = reg
    return reg
