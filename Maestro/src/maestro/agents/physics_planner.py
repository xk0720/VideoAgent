"""PhysicsPlannerAgent — attach a PhysicsAnnotation to each ShotSpec (C6 v0.4).

The planner no longer simulates anything. It records what is physically at
stake in a shot (entities, motion classes, interactions, expected failure
modes) so the verification stack knows what to check and the skill library
knows the shot's physical signature.

On an HSI tier-1 replan it RE-ANNOTATES the shot (entities may change after
prompt fixes) at an UNCHANGED verification bar (strictness 1.0). It does NOT
tighten strictness on a failing shot: tightening the bar on a shot that is
already failing inverts the repair incentive — same-quality regenerations
accrue MORE verdicts, p2 drops, and the monotonic Verifier rejects the very
candidates that would have fixed the defect. Repair guidance instead comes
from the OBSERVED violations (the caller builds anti-violation prompt hints
from the worst verdict). Strictness > 1.0 is reserved for the post-acceptance
hardening pass (`physics.post_accept_strictness`), where a tighter bar is a
log-only quality watchdog, never a rejection.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

from ..physics.annotate import annotate_physics
from ..types import ShotSpec
from .base import BaseAgent


class PhysicsPlannerAgent(BaseAgent):
    def run(self, spec: ShotSpec, cache_dir: Path | None = None, fps: int = 8) -> ShotSpec:
        annotation = annotate_physics(spec)
        spec.physics_annotation = annotation
        self._log(
            "annotate_physics",
            {"shot_idx": spec.shot_idx, "prompt": spec.prompt},
            {
                "entities": [(e.name, e.motion_class) for e in annotation.entities],
                "interactions": [it.kind for it in annotation.interactions],
                "expected_modes": [m.value for m in annotation.expected_modes],
            },
        )
        return spec

    def replan(
        self, spec: ShotSpec, cache_dir: Path | None = None, fps: int = 8,
        strictness: float = 1.0, base_prompt: Optional[str] = None,
    ) -> ShotSpec:
        """HSI Tier-1: re-annotate at an UNCHANGED verification bar.

        `base_prompt` is the shot's ORIGINAL prompt, snapshotted before any
        tier mutated `spec.prompt` (Tier 2 appends '| plan-fix: <hint>' in
        place). Re-annotating from the mutated prompt would keyword-scan the
        hint text and could re-introduce failure modes/entities the hint
        merely names — so annotation always derives from the original prompt
        when one is provided.

        `strictness` defaults to 1.0 and tier-1 callers keep it there: this
        method used to tighten the bar (legacy <1.0 convention mapped to a
        multiplier >1.0), which was self-defeating on a failing shot (see
        module docstring). The parameter remains only so the post-acceptance
        hardening pass can annotate at a tighter, log-only bar.

        The spec's annotation is swapped in place so callers' references stay
        valid."""
        source = spec if base_prompt is None else dataclasses.replace(
            spec, prompt=base_prompt,
        )
        spec.physics_annotation = annotate_physics(source, strictness=strictness)
        self._log(
            "replan_physics",
            {"shot_idx": spec.shot_idx, "strictness": strictness,
             "annotated_from_base_prompt": base_prompt is not None},
            {"entities": [(e.name, e.motion_class)
                          for e in spec.physics_annotation.entities]},
        )
        return spec
