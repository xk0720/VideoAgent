"""Implementation of the console entry points; the thin scripts/ wrappers
also import from here so both paths share code.

Open-source dependency: stdlib ``argparse``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .logging import configure_logging, logger
from .memory.store import MemoryStore
from .pipeline.preprocess import preprocess
from .pipeline.run import run_pipeline


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None,
                   help="Path to configs/default.yaml (defaults to repo root copy)")
    p.add_argument("--log-level", default="INFO")


def preprocess_main() -> int:
    p = argparse.ArgumentParser(prog="lva-preprocess",
                                description="Stage 1 preprocessing: source video → narrative memory")
    p.add_argument("--source", nargs="+", required=True, type=Path)
    p.add_argument("--music", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, required=True)
    _add_common(p)
    args = p.parse_args()
    configure_logging(args.log_level)
    cfg = load_config(args.config)
    preprocess(args.source, args.music, args.cache_dir, cfg)
    logger.info("preprocess finished")
    return 0


def build_memory_main() -> int:
    p = argparse.ArgumentParser(prog="lva-build-memory",
                                description="(Re-)materialize NarrativeMemory from a cache dir")
    p.add_argument("--cache-dir", type=Path, required=True)
    _add_common(p)
    args = p.parse_args()
    configure_logging(args.log_level)
    store = MemoryStore(args.cache_dir)
    mem = store.load_full_memory(load_features=False)
    print(json.dumps({
        "n_shots": len(mem.shots),
        "n_events": len(mem.events),
        "n_stories": len(mem.stories),
        "n_characters": len(mem.characters),
        "has_music": mem.music_profile is not None,
    }, indent=2))
    store.close()
    return 0


def run_pipeline_main() -> int:
    p = argparse.ArgumentParser(prog="lva-run",
                                description="End-to-end LongVideoEditAgent pipeline")
    p.add_argument("--source", nargs="+", type=Path,
                   help="Source videos (omit if --cache-dir already populated)")
    p.add_argument("--music", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--user-prompt", required=True, type=str)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trajectory-log", type=Path, default=None)
    p.add_argument("--lesson-book", type=Path, default=None,
                   help="Path to cross-run LessonBook JSONL (default: <cache>/lessons.jsonl)")
    p.add_argument("--preference-log", type=Path, default=None,
                   help="If set, log (winner, loser) pairs to this file for DPO/IPO training")
    p.add_argument("--self-consistency-k", type=int, default=1,
                   help="Run ScreenwriterAgent K times and majority-vote (default 1 = off)")
    p.add_argument("--no-critic", action="store_true",
                   help="Skip the post-run CriticAgent pass (default: run it)")
    _add_common(p)
    args = p.parse_args()
    run_pipeline(
        source_videos=args.source or [],
        user_prompt=args.user_prompt,
        output_path=args.output,
        music=args.music,
        cache_dir=args.cache_dir,
        config_path=args.config,
        trajectory_log_path=args.trajectory_log,
        lesson_book_path=args.lesson_book,
        preference_log_path=args.preference_log,
        self_consistency_k=args.self_consistency_k,
        run_critic=not args.no_critic,
        log_level=args.log_level,
    )
    return 0


def eval_main() -> int:
    p = argparse.ArgumentParser(prog="lva-eval",
                                description="Run a benchmark adapter (stub in v0.1)")
    p.add_argument("--benchmark", choices=["mashup-bench", "cine-bench"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    _add_common(p)
    args = p.parse_args()
    configure_logging(args.log_level)
    logger.info(f"(stub) eval {args.benchmark} → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"benchmark": args.benchmark, "status": "stub-not-implemented"}, indent=2))
    return 0


def viz_trajectory_main() -> int:
    p = argparse.ArgumentParser(prog="lva-viz",
                                description="Print a trajectory JSONL log as a Rich table")
    p.add_argument("--log", type=Path, required=True)
    _add_common(p)
    args = p.parse_args()
    configure_logging(args.log_level)
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title=str(args.log))
        for col in ["timestamp", "agent", "action", "reward"]:
            table.add_column(col)
        for line in args.log.read_text().splitlines():
            rec = json.loads(line)
            table.add_row(
                f"{rec['timestamp']:.3f}",
                rec.get("agent_name", ""),
                rec.get("action", ""),
                str(rec.get("reward", "")),
            )
        console.print(table)
    except ImportError:
        # rich not installed — fall back to plain text.
        header = f"{'timestamp':<14} {'agent':<14} {'action':<20} reward"
        print(header); print("-" * len(header))
        for line in args.log.read_text().splitlines():
            rec = json.loads(line)
            print(f"{rec['timestamp']:<14.3f} {rec.get('agent_name',''):<14} "
                  f"{rec.get('action',''):<20} {rec.get('reward','')}")
    return 0
