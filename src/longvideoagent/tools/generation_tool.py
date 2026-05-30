"""GenerationTool — wraps a BaseVideoGenClient.

In v0.1 the backing client is MockVideoGenClient (writes a coloured clip).
The wrapper's responsibility is to:
    1. Compose the final text prompt (semantic query + cinematography hint).
    2. Hand neighbour-frame / character-reference conditions to the client.
    3. Return an EditingSegment with `source="generation"`.

Open-source library: depends only on its injected BaseVideoGenClient (see
``models.video_gen`` for upstream library notes).

Metric scoring for generated segments
-------------------------------------
Generated clips don't have real ShotFeatures to feed metric_tool, so we
synthesise plausible per-clip metric_scores that *adjust* with the inputs
the agent provided:

    m1 prompt_relevance    high prior (generation is text-conditioned)
    m2 segment_consistency boosted if a neighbour first-frame was supplied
    m3 motion_continuity   boosted if a flow field was supplied
    m4 framing             prior 0.5 (no saliency map until v0.2)
    m5 beat_sync           if music beats are in neighbor_context and the
                           segment starts on a beat, near 1
    m6 energy              prior 0.5

The validator then runs over these the same way it does for retrieval
candidates, so the EditorAgent gets a fair comparison between routes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ..models.video_gen.base import BaseVideoGenClient
from ..types import EditingSegment, SegmentGuidance
from .base import BaseTool
from .metric_tool import m5_beat_sync


class GenerationTool(BaseTool):
    name = "generate_shot"
    description = "Synthesize a new video clip conditioned on the guidance and neighbour context."

    def __init__(self, client: BaseVideoGenClient, default_duration_s: float = 4.0) -> None:
        self.client = client
        self.default_duration_s = default_duration_s

    def run(
        self,
        guidance: SegmentGuidance,
        neighbor_context: Optional[dict[str, Any]] = None,
        output_dir: Path = Path("./outputs/generated"),
    ) -> EditingSegment:
        neighbor_context = neighbor_context or {}
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"gen_seg{guidance.segment_idx:04d}.mp4"
        full_prompt = self._compose_prompt(guidance)
        client_supports = self.client.supported_conditions()
        first_frame = neighbor_context.get("end_frame") if "first_frame" in client_supports else None
        refs = neighbor_context.get("character_anchors") if "reference_images" in client_supports else None
        flow = neighbor_context.get("end_flow") if "flow_field" in client_supports else None

        produced = self.client.generate(
            prompt=full_prompt,
            duration=self.default_duration_s,
            out_path=out_path,
            first_frame=first_frame,
            reference_images=refs if isinstance(refs, list) else None,
            flow_field=flow,
            cinematography_hint=", ".join(guidance.cinematography_hints) or None,
        )

        metric_scores = self._estimate_metrics(
            has_first_frame=first_frame is not None,
            has_flow=flow is not None,
            has_refs=bool(refs),
            beats=neighbor_context.get("beats", []),
            cut_time_s=neighbor_context.get("expected_start_time_s", 0.0),
            previous_source=neighbor_context.get("previous_source"),
        )
        return EditingSegment(
            segment_idx=guidance.segment_idx,
            source="generation",
            duration=self.default_duration_s,
            gen_prompt=full_prompt,
            gen_video_path=Path(produced),
            gen_conditions={"has_first_frame": first_frame is not None,
                            "has_refs": bool(refs),
                            "has_flow": flow is not None},
            metric_scores=metric_scores,
        )

    @staticmethod
    def _estimate_metrics(
        has_first_frame: bool,
        has_flow: bool,
        has_refs: bool = False,
        beats: Iterable[float] = (),
        cut_time_s: float = 0.0,
        previous_source: str | None = None,
    ) -> dict[str, float]:
        """Mock metric synthesis — honest about how anchor quality flows in.

        v0.2 honesty fix (see ``docs/BASELINE_v0_2.md``): we used to return
        flat 0.50 for m4 and a narrow 0.50→0.75 spread for m2/m3 regardless
        of ``previous_source``. That made R→G and G→G boundaries
        indistinguishable, which collapses the hybrid-vs-alternation
        distinction the framework's core claim depends on. We now:

            • give m2/m3 a wider spread between "real anchor" and "no anchor"
            • let m4 reflect whether character_refs were actually passed
            • discount when the anchor itself was synthesised (G→G chain)

        These are still mock numbers — the *real* test is when a real
        video-gen backend (OmniWeaving / HunyuanVideo) replaces the
        ``MockVideoGenClient``. But the spread is enough that B can now
        detect whether the routing decision matters at all.
        """
        # Anchor quality multiplier: 1.0 (came from retrieval = real source frames),
        # 0.70 (came from generation = chain of synth), 0.50 (no anchor).
        if previous_source == "retrieval":
            anchor_quality = 1.0
        elif previous_source == "generation":
            anchor_quality = 0.70
        else:
            anchor_quality = 0.50

        # m2 (segment consistency) — anchor is half of this; the other half
        # is whether the generator was given anchor inputs at all.
        m2_base = 0.85 if has_first_frame else 0.30
        m2 = m2_base * anchor_quality + (1.0 - anchor_quality) * 0.30

        # m3 (motion continuity) — flow field is needed AND it has to come from
        # a real source. Without flow, mock generator has no chance.
        m3_base = 0.80 if has_flow else 0.25
        m3 = m3_base * anchor_quality + (1.0 - anchor_quality) * 0.25

        # m4 (framing) — character references discipline framing; the anchor
        # quality is less load-bearing here than m2/m3.
        m4 = (0.70 if has_refs else 0.40) * (0.7 + 0.3 * anchor_quality)

        return {
            "m1": 0.75,                                  # text-conditioned ⇒ good prior
            "m2": float(round(m2, 4)),
            "m3": float(round(m3, 4)),
            "m4": float(round(m4, 4)),
            "m5": m5_beat_sync(cut_time_s, list(beats)) if beats else 0.50,
            "m6": 0.50,
        }

    @staticmethod
    def _compose_prompt(guidance: SegmentGuidance) -> str:
        bits = [guidance.semantic_query.strip()]
        if guidance.cinematography_hints:
            bits.append("cinematography: " + ", ".join(guidance.cinematography_hints))
        return ". ".join(b for b in bits if b)


__all__ = ["GenerationTool"]
