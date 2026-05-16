"""Stage 3 — composition via EditorAgent + AssemblyTool."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..agents import EditorAgent, ValidatorAgent
from ..config import Config
from ..memory.retriever import MemoryRetriever
from ..memory.store import MemoryStore
from ..models.llm import build_llm_from_alias
from ..models.reward import MLLMJudge, MockRewardModel
from ..models.video_gen import build_video_gen_from_config
from ..tools import AssemblyTool, GenerationTool, RetrievalTool
from ..types import EditingScript, NarrativeMemory, SegmentGuidance
from ..utils.trajectory import TrajectoryLogger


def compose(
    memory: NarrativeMemory,
    guidances: list[SegmentGuidance],
    config: Config,
    output_path: Path,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    memory_store: Optional[MemoryStore] = None,
) -> EditingScript:
    output_path = Path(output_path)
    gen_dir = (config.output_root / "generated").resolve()
    gen_dir.mkdir(parents=True, exist_ok=True)

    if memory_store is None:
        memory_store = MemoryStore(config.cache_root)
    retriever = MemoryRetriever(memory_store, embed_dim=config.preprocess.feature_extractor.embed_dim)

    ed_llm = build_llm_from_alias("editor", mocks_enabled=config.mocks.llm)
    video_gen = build_video_gen_from_config(
        config.compose.generation.backend, mocks_enabled=config.mocks.video_gen,
    )

    retrieval_tool = RetrievalTool(
        retriever, beam_width=config.compose.retrieval.beam_width,
        top_k_pool=config.compose.retrieval.top_k_pool,
        sliding_stride=config.compose.retrieval.sliding_stride,
    )
    gen_tool = GenerationTool(video_gen, default_duration_s=config.compose.generation.duration_default)
    reward_model = MockRewardModel(accept_threshold=config.compose.validator_threshold) \
        if config.mocks.reward else MLLMJudge(accept_threshold=config.compose.validator_threshold)
    validator = ValidatorAgent(reward_model=reward_model, trajectory_logger=trajectory_logger)

    editor = EditorAgent(
        llm_client=ed_llm,
        retrieval_tool=retrieval_tool,
        generation_tool=gen_tool,
        validator=validator,
        trajectory_logger=trajectory_logger,
        max_steps=config.compose.max_editor_steps,
        feasibility_threshold=config.compose.generation.fallback_threshold,
    )

    state = {
        "memory": memory,
        "segment_guidances": guidances,
        "output_dir": gen_dir,
    }
    result = editor.run(state)
    script: EditingScript = result["script"]

    # Final assembly — real ffmpeg.
    assembly = AssemblyTool(config.assembly)
    assembly.run(script, output_path)
    return script


__all__ = ["compose"]
