"""ScreenwriterAgent — idea -> narrative shot outline (music-driven structure)."""
from __future__ import annotations

from ..types import AssetMemory
from .base import BaseAgent


class ScreenwriterAgent(BaseAgent):
    def run(self, user_prompt: str, asset_memory: AssetMemory) -> list[str]:
        # DESIGN_DECISION: v0.1 deterministic outline. Number of shots = music
        # sections if available, else config n_shots (default 3).
        n = self.config.get("n_shots", 3)
        if asset_memory.music_profile and asset_memory.music_profile.sections:
            n = len(asset_memory.music_profile.sections)
        n = max(1, min(n, self.config.get("max_shots", 6)))

        self.llm.complete(self.prompt_template + "\n" + user_prompt)

        # Split user prompt into shot beats; reuse clauses if present.
        clauses = [c.strip() for c in user_prompt.replace("；", ";").split(";") if c.strip()]
        outline: list[str] = []
        for i in range(n):
            if clauses:
                base = clauses[i % len(clauses)]
            else:
                base = user_prompt
            outline.append(f"Shot {i + 1}: {base}")
        self._log("write_outline", {"user_prompt": user_prompt, "n_shots": n},
                  {"outline": outline})
        return outline
