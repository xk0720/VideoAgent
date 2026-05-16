"""Stage 3 — composition via EditorAgent + AssemblyTool.

v0.2 enhancements wired in here:
    • EnsembleRewardModel  — 2 mock judges (semantic-priority weighted +
                              motion-priority weighted) vote, disagreement
                              flagged. Replaces single MockRewardModel by
                              default. Closes the "single-judge bias" gap.
    • PreferenceLogger     — when EditorAgent sees ≥2 candidates per
                              segment, writes (winner, losers) to a JSONL
                              file ready for DPO / IPO training (v0.3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..agents import EditorAgent, ValidatorAgent
from ..config import Config
from ..memory.retriever import MemoryRetriever
from ..memory.store import MemoryStore
from ..models.llm import build_llm_from_alias
from ..models.reward import (
    BaseRewardModel, EnsembleRewardModel, MLLMJudge, MockRewardModel,
)
from ..models.video_gen import build_video_gen_from_config
from ..tools import AssemblyTool, GenerationTool, RetrievalTool
from ..types import EditingScript, NarrativeMemory, SegmentGuidance
from ..utils.preferences import PreferenceLogger
from ..utils.trajectory import TrajectoryLogger


def _build_reward_model(config: Config) -> BaseRewardModel:
    """Build the v0.2 multi-judge ensemble.

    Even in pure-mock mode we use ≥2 MockRewardModels with different metric
    weight vectors — this is the simplest non-trivial ensemble and it
    surfaces a non-zero disagreement signal for ``EnsembleResult``.
    """
    threshold = config.compose.validator_threshold

    # Judge 1: balanced (default heuristic weights).
    judge_balanced = MockRewardModel(accept_threshold=threshold)
    # Judge 2: semantically biased (m1 dominates) — emulates a "content-first" critic.
    judge_semantic = MockRewardModel(
        accept_threshold=threshold,
        weights={"m1": 0.50, "m2": 0.30, "m3": 0.05, "m4": 0.05, "m5": 0.05, "m6": 0.05},
        name="SemanticCriticRM",
    )
    # Judge 3: motion biased (m3 + m5 dominate) — emulates a "motion-first" critic.
    judge_motion = MockRewardModel(
        accept_threshold=threshold,
        weights={"m1": 0.10, "m2": 0.10, "m3": 0.40, "m4": 0.10, "m5": 0.20, "m6": 0.10},
        name="MotionCriticRM",
    )
    judges: list[BaseRewardModel] = [judge_balanced, judge_semantic, judge_motion]

    # If real reward path is on, append the MLLMJudge so it counts as one
    # more vote (and EnsembleResult.disagreement still works).
    if not config.mocks.reward:
        judges.append(MLLMJudge(accept_threshold=threshold))

    return EnsembleRewardModel(judges, accept_threshold=threshold)


def compose(
    memory: NarrativeMemory,
    guidances: list[SegmentGuidance],
    config: Config,
    output_path: Path,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    memory_store: Optional[MemoryStore] = None,
    preference_log_path: Optional[Path] = None,
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
    reward_model = _build_reward_model(config)
    validator = ValidatorAgent(reward_model=reward_model, trajectory_logger=trajectory_logger)

    pref_logger: Optional[PreferenceLogger] = None
    if preference_log_path is not None:
        pref_logger = PreferenceLogger(preference_log_path)

    editor = EditorAgent(
        llm_client=ed_llm,
        retrieval_tool=retrieval_tool,
        generation_tool=gen_tool,
        validator=validator,
        trajectory_logger=trajectory_logger,
        max_steps=config.compose.max_editor_steps,
        feasibility_threshold=config.compose.generation.fallback_threshold,
        preference_logger=pref_logger,
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
