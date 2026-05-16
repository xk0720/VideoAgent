"""End-to-end driver — preprocess → plan → compose."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Config, load_config
from ..logging import configure_logging, logger
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
) -> EditingScript:
    configure_logging(level=log_level)
    config: Config = load_config(config_path, overrides=overrides)

    cache_dir = Path(cache_dir or config.cache_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_log_path = Path(trajectory_log_path or (output_path.with_suffix(".trajectory.jsonl")))

    with TrajectoryLogger(trajectory_log_path,
                          redact_large_tensors=config.trajectory.redact_large_tensors) as traj:
        logger.info(f"[run] cache={cache_dir} → output={output_path}")
        memory = preprocess(source_videos, music, cache_dir, config)
        # Recreate a store handle the downstream stages can share.
        store = MemoryStore(cache_dir)
        guidances = plan(memory, user_prompt, config, trajectory_logger=traj, memory_store=store)
        logger.info(f"[run] planning yielded {len(guidances)} segment guidances")
        script = compose(memory, guidances, config, output_path,
                         trajectory_logger=traj, memory_store=store)
        store.close()
        logger.info(f"[run] script duration={script.total_duration:.2f}s, "
                    f"segments={len(script.segments)}, output={script.output_path}")
        return script


__all__ = ["run_pipeline"]
