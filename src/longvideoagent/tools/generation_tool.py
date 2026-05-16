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
            beats=neighbor_context.get("beats", []),
            cut_time_s=neighbor_context.get("expected_start_time_s", 0.0),
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
        beats: Iterable[float],
        cut_time_s: float,
    ) -> dict[str, float]:
        return {
            "m1": 0.75,                                            # text-conditioned ⇒ good prior
            "m2": 0.75 if has_first_frame else 0.50,               # neighbour anchor helps
            "m3": 0.70 if has_flow else 0.45,                      # explicit flow condition helps
            "m4": 0.50,                                            # no saliency until v0.2
            "m5": m5_beat_sync(cut_time_s, beats) if beats else 0.50,
            "m6": 0.50,
        }

    @staticmethod
    def _compose_prompt(guidance: SegmentGuidance) -> str:
        bits = [guidance.semantic_query.strip()]
        if guidance.cinematography_hints:
            bits.append("cinematography: " + ", ".join(guidance.cinematography_hints))
        return ". ".join(b for b in bits if b)


__all__ = ["GenerationTool"]
