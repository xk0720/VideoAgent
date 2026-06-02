"""MLLM/VLM wrapper used as judge & critic. v0.1 MockMLLMClient is deterministic.

The mock simulates a key property we need to demo: as a clip is revised, its
remaining defects shrink, so the loop converges and Verifier sees monotonic
improvement. Real VLM (Qwen-VL / GPT-4o) plugs in behind the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import CandidateClip, PhysFailureMode, PhysicsVerdict, ShotSpec
from ..physics.failure_modes import detect_expected_modes, suggest_intervention


class BaseMLLMClient(ABC):
    @abstractmethod
    def assess_semantic(self, clip: CandidateClip, spec: ShotSpec) -> list[tuple[str, bool, str]]:
        """Return [(question, passed, fix_instruction), ...]."""

    @abstractmethod
    def assess_physics(self, clip: CandidateClip, spec: ShotSpec, fps: int) -> list[PhysicsVerdict]:
        ...

    def compare(self, a: CandidateClip, b: CandidateClip, spec: ShotSpec) -> int:
        """Judge which candidate is better: +1 if a, -1 if b, 0 tie.

        Used by the VISTA-style tournament. Default ranks by weighted metric total;
        a real MLLM judge overrides this with a perceptual comparison. Critically,
        the tournament calls this BOTH directions (a,b) and (b,a) to de-bias the
        judge (VISTA, arXiv:2510.15831), so this method only needs a single view.
        """
        sa = a.metric_scores.get("weighted_total", 0.0)
        sb = b.metric_scores.get("weighted_total", 0.0)
        if abs(sa - sb) < 1e-9:
            return 0
        return 1 if sa > sb else -1


class MockMLLMClient(BaseMLLMClient):
    def __init__(self, name: str = "mock-mllm", strictness: float = 1.0):
        self.name = name
        self.strictness = strictness

    # ── semantic checklist ──
    def assess_semantic(self, clip: CandidateClip, spec: ShotSpec) -> list[tuple[str, bool, str]]:
        # Deterministic: a couple of semantic checks that pass once revised enough.
        rev = clip.revision
        tokens = [t for t in spec.prompt.lower().split() if len(t) > 3][:3]
        items: list[tuple[str, bool, str]] = []
        for i, tok in enumerate(tokens):
            # earlier tokens get fixed first
            passed = rev > i
            fix = "" if passed else f"strengthen depiction of '{tok}'"
            items.append((f"Does the clip clearly show '{tok}'?", passed, fix))
        if not items:
            items.append(("Does the clip match the prompt?", rev >= 1, "align to prompt"))
        return items

    # ── physics verdicts (localized, per failure mode) ──
    def assess_physics(self, clip: CandidateClip, spec: ShotSpec, fps: int) -> list[PhysicsVerdict]:
        expected = (
            spec.physics_sketch.expected_modes
            if spec.physics_sketch and spec.physics_sketch.expected_modes
            else detect_expected_modes(spec.prompt)
        )
        verdicts: list[PhysicsVerdict] = []
        n_frames = max(1, int(round(spec.duration * fps)))
        for j, mode in enumerate(expected):
            # severity decays with revisions -> loop converges; resolved when < 0.3
            base = 0.8 - 0.3 * clip.revision - 0.1 * j
            severity = max(0.0, round(base * self.strictness, 3))
            if severity >= 0.3:
                start = min(j, max(0, n_frames - 1))
                end = min(n_frames, start + max(1, n_frames // 3))
                verdicts.append(
                    PhysicsVerdict(
                        mode=mode,
                        frame_range=(start, end),
                        severity=severity,
                        suggested_intervention=suggest_intervention(mode),
                    )
                )
        return verdicts


def build_mllm(spec: str | dict | None) -> BaseMLLMClient:
    name = "mock-mllm"
    if isinstance(spec, dict):
        name = spec.get("name", name)
    elif isinstance(spec, str):
        name = spec
    # DESIGN_DECISION: real Qwen-VL/GPT-4o judge plugs in here.
    return MockMLLMClient(name=name)
