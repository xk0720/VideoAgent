"""End-to-end driver — preprocess → plan → compose → critic.

v0.2 enhancements wired here:
    • LessonBook       — persistent JSONL of cross-run reflections; loaded
                          before planning so Screenwriter sees relevant lessons;
                          written to by CriticAgent after the run completes.
    • CriticAgent      — post-hoc scan of the run's trajectory; identifies
                          suspect decisions and emits Lessons into the LessonBook.
    • PreferenceLogger — opt-in pairwise DPO/IPO data collection during compose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..agents.critic import CriticAgent
from ..config import Config, load_config
from ..logging import configure_logging, logger
from ..memory.lessons import LessonBook
from ..memory.store import MemoryStore
from ..types import EditingScript
from ..utils.trajectory import TrajectoryLogger
from .compose import compose
from .plan import plan
from .preprocess import preprocess


def run_pipeline(
    source_videos: list[Path],
    user_prompt: str,
    output_path: Path,
    music: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    trajectory_log_path: Optional[Path] = None,
    overrides: Optional[dict] = None,
    log_level: str = "INFO",
    lesson_book_path: Optional[Path] = None,
    preference_log_path: Optional[Path] = None,
    self_consistency_k: int = 1,
    run_critic: bool = True,
) -> EditingScript:
    configure_logging(level=log_level)
    config: Config = load_config(config_path, overrides=overrides)

    cache_dir = Path(cache_dir or config.cache_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_log_path = Path(trajectory_log_path or (output_path.with_suffix(".trajectory.jsonl")))

    # LessonBook: a persistent cross-run store. Default location keeps it
    # next to the cache, so blowing away .cache also resets the lesson book —
    # surprise-free for users.
    lesson_book_path = Path(lesson_book_path or (cache_dir / "lessons.jsonl"))
    lesson_book = LessonBook(lesson_book_path)
    logger.info(f"[run] lesson_book={lesson_book_path} (existing: {len(lesson_book)} lessons)")

    with TrajectoryLogger(trajectory_log_path,
                          redact_large_tensors=config.trajectory.redact_large_tensors) as traj:
        logger.info(f"[run] cache={cache_dir} → output={output_path}")
        memory = preprocess(source_videos, music, cache_dir, config)
        store = MemoryStore(cache_dir)
        guidances = plan(
            memory, user_prompt, config,
            trajectory_logger=traj, memory_store=store,
            lesson_book=lesson_book, self_consistency_k=self_consistency_k,
        )
        logger.info(f"[run] planning yielded {len(guidances)} segment guidances")
        script = compose(
            memory, guidances, config, output_path,
            trajectory_logger=traj, memory_store=store,
            preference_log_path=preference_log_path,
        )
        store.close()
        logger.info(f"[run] script duration={script.total_duration:.2f}s, "
                    f"segments={len(script.segments)}, output={script.output_path}")

    # CriticAgent: read the now-flushed trajectory, write Lessons into the book.
    if run_critic:
        critic = CriticAgent(lesson_book)
        new_lessons = critic.review(
            trajectory_log_path, user_prompt=user_prompt,
            run_context={"memory_size": len(memory.shots),
                         "music_bpm": memory.music_profile.bpm if memory.music_profile else 0.0,
                         "source_video_count": len(source_videos)},
        )
        logger.info(f"[run] CriticAgent wrote {len(new_lessons)} new lesson(s).")

    return script


__all__ = ["run_pipeline"]
