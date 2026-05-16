"""LLM client ABC + canned mock implementation + alias factory.

Design notes:
    • The chat signature follows OpenAI's ``messages = [{"role", "content"}]``
      shape because both Anthropic (``client.messages.create``) and DeepSeek
      already follow that convention, so wrappers can be near-trivial.
    • Function calling is exposed as an optional ``tools`` arg; backends that
      don't support it should raise NotImplementedError, NOT silently ignore.

Open-source libraries (only imported lazily when a real backend is built):
    • openai          https://github.com/openai/openai-python
    • anthropic       https://github.com/anthropics/anthropic-sdk-python
    • tenacity        https://github.com/jd/tenacity (retry/backoff helper)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ...config import configs_dir, load_yaml
from ...logging import logger


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Optional[Any] = None

    def parse_json(self) -> Any:
        """Best-effort JSON parse of ``text`` (strips ```json fences if any)."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            # remove optional language hint after the fence
            if t.lower().startswith("json"):
                t = t[4:]
        return json.loads(t)


class BaseLLMClient(ABC):
    backend_name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def supports_function_calling(self) -> bool:
        return False


class MockLLMClient(BaseLLMClient):
    """Deterministic stub. v0.1 default everywhere when mocks.llm == true.

    Looks for a canned JSON response under
    ``tests/fixtures/mock_llm/<alias>.json`` first; otherwise returns a
    minimal but schema-valid stub keyed on the agent role inferred from the
    first system message.
    """

    backend_name = "mock"

    def __init__(self, alias: str = "default", fixture_root: Optional[Path] = None) -> None:
        self.alias = alias
        self.fixture_root = fixture_root or (Path(__file__).resolve().parents[3]
                                             / ".." / "tests" / "fixtures" / "mock_llm").resolve()

    def supports_function_calling(self) -> bool:
        return True

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=1024, **kwargs) -> LLMResponse:
        text = self._lookup_fixture() or self._stub_for(messages)
        # Maintain a per-instance call counter so successive mock calls can
        # vary their output (e.g. director's semantic_query → no duplicate).
        self._call_idx = getattr(self, "_call_idx", 0) + 1
        return LLMResponse(text=text)

    def _lookup_fixture(self) -> Optional[str]:
        p = self.fixture_root / f"{self.alias}.json"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def _stub_for(self, messages) -> str:
        # Heuristic: peek at the role in the prompt to choose a shape.
        joined = "\n".join(m.get("content", "") if isinstance(m.get("content"), str)
                           else "" for m in messages).lower()
        if "screenwriter" in joined or "global structural plan" in joined:
            # Count how many music sections the prompt advertised (it inlines
            # them as 'idx | name | start..end | energy=...') so the mock can
            # emit a section_plan for every one.
            import re as _re
            n_sections = max(1, len(_re.findall(r"^\s*\d+\s*\|", joined, _re.M)))
            energy_curve = ["medium", "high", "extreme", "medium",
                            "low", "high", "medium", "extreme"]
            tags = [["establishing"], ["action"], ["climax"], ["resolution"],
                    ["transition"], ["build"], ["release"], ["outro"]]
            return json.dumps({"section_plans": [
                {"music_section_idx": i,
                 "energy_level": energy_curve[i % len(energy_curve)],
                 "visual_tags": tags[i % len(tags)],
                 "rationale": "mock"}
                for i in range(n_sections)
            ]})
        if "director" in joined or "semantic query" in joined:
            # Vary the query string across calls so OrchestratorAgent's
            # "no_duplicate_queries" heuristic doesn't trigger on every run.
            idx = getattr(self, "_call_idx", 0)
            vocabulary = [
                "a cinematic establishing shot",
                "a tight close-up of the subject",
                "a sweeping wide shot of the scene",
                "a dynamic tracking shot",
                "an over-the-shoulder reaction",
                "a low-angle hero shot",
                "an aerial overhead view",
                "a handheld point-of-view shot",
            ]
            return json.dumps({
                "semantic_query": vocabulary[idx % len(vocabulary)],
                "editing_heuristic": "default",
                "rhythmic_pacing": [2, 2, 2, 2],
                "cinematography_hints": ["medium", "static"],
                "retrieval_feasibility": 0.8,
            })
        if "orchestrator" in joined or "validate" in joined:
            return json.dumps({"passed": True, "feedback": []})
        if "editor" in joined:
            # If a candidate was rejected by the auto-validator, switch route.
            action = "generate" if "not accepted" in joined or "rejected" in joined \
                else "retrieve"
            return json.dumps({"action": action, "rationale": "mock-default"})
        if "judge" in joined or "validator" in joined or "score" in joined:
            return json.dumps({"score": 7.5, "reasons": [], "accepted": True})
        return "mock-response"


# ─────────────────────────────────────────────────────────────────────
# Real backends (imported lazily by build_llm_from_alias)
# ─────────────────────────────────────────────────────────────────────


def build_llm_from_alias(
    alias: str,
    mocks_enabled: bool = True,
    llm_yaml_path: Optional[Path] = None,
) -> BaseLLMClient:
    """Resolve an alias from configs/models/llm.yaml into a concrete client.

    When ``mocks_enabled`` is True (v0.1 default) we always return a
    MockLLMClient configured with the alias name so it can pick up the
    matching fixture file.
    """
    if mocks_enabled:
        return MockLLMClient(alias=alias)

    cfg = load_yaml(llm_yaml_path or (configs_dir() / "models" / "llm.yaml"))
    spec = cfg["aliases"].get(alias)
    if spec is None:
        raise KeyError(f"LLM alias {alias!r} not present in llm.yaml")
    backend = spec["backend"]
    try:
        if backend == "openai":
            from .openai_client import OpenAIClient
            return OpenAIClient(model=spec["model"])
        if backend == "anthropic":
            from .anthropic_client import AnthropicClient
            return AnthropicClient(model=spec["model"])
        if backend == "deepseek":
            from .deepseek_client import DeepSeekClient
            return DeepSeekClient(model=spec["model"])
        if backend == "vllm":
            from .vllm_local import VLLMLocalClient
            return VLLMLocalClient(model=spec["model"])
    except ImportError as e:
        logger.error(f"Backend {backend} requires an extra; install with "
                     f"`pip install longvideoagent[llm]`. ({e})")
        raise
    raise ValueError(f"Unknown LLM backend: {backend}")


__all__ = ["BaseLLMClient", "LLMResponse", "MockLLMClient", "build_llm_from_alias"]
