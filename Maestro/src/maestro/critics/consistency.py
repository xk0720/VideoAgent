"""ConsistencyCritic — identity / style continuity across frames & shots (E1).

v0.4: when the dual-register entity memory (memory/entity_store.py) holds a
COMMITTED state for an entity named in this spec, the critic also checks the
committed attributes against the rendered clip — earlier shots' verified
state becomes part of what later shots are reviewed against (Angle 2's
auditability put to work). SIGNAL SOURCE: in mock mode the check is a
token-level contradiction scan over the prompt echo in the clip body (the
only render-side text the mock produces); a committed value fails only when
its known antonym appears AND the value itself does not. The real backend
swaps in an MLLM frame comparison against the identity reference paths —
same checklist contract. What was compared is logged either way.
"""
from __future__ import annotations

from pathlib import Path

from ..types import ChecklistItem
from .base import BaseCritic

# Antonym pairs mirroring memory/entity_store.py's _STATE_CUES — the mock
# contradiction signal. Small and visible on purpose.
_CONTRADICTIONS = {
    "wet": "dry", "dry": "wet",
    "broken": "intact", "intact": "broken",
    "clean": "dirty", "dirty": "clean",
    "burning": "extinguished",
    "happy": "sad", "sad": "happy",
}


class ConsistencyCritic(BaseCritic):
    kind = "consistency"

    # Wired by pipeline/run.py after MultiLayerMemory opens (the store does
    # not exist yet when build_components constructs the board).
    entity_store = None

    def review(self, clip, spec, asset_memory=None, fps=8) -> None:
        if not spec.identity_refs:
            passed, fix = True, ""
        else:
            # mock: identity locked in once a revision has carried the anchor
            passed = clip.revision >= 1
            fix = "" if passed else "carry identity anchor across frames; re-condition on reference image"
        clip.checklist.items.append(
            ChecklistItem(
                question="Are character/object identities consistent across the clip?",
                kind="consistency", passed=passed, fix_instruction=fix,
            )
        )

        # v0.4 — committed-state continuity: entities with a committed state
        # register must not be contradicted by this shot's render.
        compared = self._check_committed_states(clip, spec)

        self._log({"shot_idx": spec.shot_idx, "revision": clip.revision},
                  {"passed": passed, "has_refs": bool(spec.identity_refs),
                   "committed_state_compared": compared})

    def _check_committed_states(self, clip, spec) -> list[dict]:
        """Compare committed entity attributes against the rendered evidence.
        Returns the comparison record (also logged) — empty when no entity
        store is wired or no named entity has committed state."""
        store = self.entity_store
        if store is None or not getattr(store, "identities", None):
            return []
        evidence = spec.prompt.lower()
        try:
            evidence += "\n" + Path(clip.video_path).read_text(
                encoding="utf-8", errors="ignore").lower()
        except OSError:
            pass
        toks = {w.strip(".,;:!?'\"()") for w in evidence.split()}
        annotated = {
            e.name.lower()
            for e in (spec.physics_annotation.entities
                      if spec.physics_annotation else [])
        }
        compared: list[dict] = []
        for ident in store.identities.values():
            name = ident.name.lower()
            if name not in toks and name not in annotated:
                continue
            st = store.states.get(ident.entity_id)
            if not st or not st.attributes:
                continue
            for fld, val in sorted(st.attributes.items()):
                antonym = _CONTRADICTIONS.get(val)
                contradicted = bool(
                    antonym and antonym in toks and val not in toks
                )
                compared.append({
                    "entity": ident.name, "field": fld, "committed": val,
                    "contradicted_by": antonym if contradicted else "",
                })
                clip.checklist.items.append(ChecklistItem(
                    question=(
                        f"Does '{ident.name}' still show committed state "
                        f"{fld}={val} (entity memory v{st.version})?"
                    ),
                    kind="consistency",
                    passed=not contradicted,
                    fix_instruction=(
                        "" if not contradicted else
                        f"re-condition on entity memory: {ident.name} must be "
                        f"{val}, render shows '{antonym}' — or author an "
                        f"explicit correction transition"
                    ),
                ))
        return compared
