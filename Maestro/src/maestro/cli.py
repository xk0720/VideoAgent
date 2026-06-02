"""Maestro CLI — single entry for smoke / serve / run-once.

    python -m maestro.cli smoke                       # CPU healthcheck (no keys, no GPU)
    python -m maestro.cli serve --port 8000           # FastAPI on UniVA-compatible /health
    python -m maestro.cli run-once --prompt "..." -o out.mp4

Why a CLI instead of three scripts: one entry point in `pyproject.toml` makes
`pip install -e .` give you a `maestro` console command on a server with no
extra wiring. UniVA wires its entry points the same way.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _cmd_smoke(args) -> int:
    """Build a temporary pipeline, run end-to-end with mocks, fail loud on any
    exception. The single command an operator runs after deploying to verify
    "the box is alive and the package is wired right".
    """
    from .config import load_config
    from .pipeline.run import run_maestro
    from .tools.base import default_registry

    print("[smoke] tool registry:")
    reg = default_registry()
    by_cat: dict[str, list[str]] = {}
    for spec in reg.list_specs():
        by_cat.setdefault(spec.category, []).append(spec.name)
    for cat in sorted(by_cat):
        print(f"  {cat}: {', '.join(sorted(by_cat[cat]))}")

    with tempfile.TemporaryDirectory(prefix="maestro_smoke_") as td:
        td = Path(td)
        out = td / "smoke.mp4"
        result = run_maestro(
            user_prompt=args.prompt,
            output_path=out,
            config=load_config(),
            cache_dir=td / "cache",
            trajectory_path=out.with_suffix(".trajectory.jsonl"),
            lesson_path=td / "lessons.jsonl",
        )
        report = result["report"]
        print(f"\n[smoke] OK: n_shots={report['n_shots']} "
              f"lessons={report['lessons_learned']}")
        for s in report["shots"]:
            print(f"  shot {s['shot_idx']}: revisions={s['revisions_used']} "
                  f"tier_used={s['tier_used']} escalations={s['escalations']}")
        return 0


def _cmd_serve(args) -> int:
    """Bring up the FastAPI server. Mirrors UniVA's `python univa_server.py`
    one-shot entry — defaults match `.env.example`."""
    try:
        import uvicorn
    except ImportError:
        print("[serve] uvicorn not installed. `pip install '.[server]'`",
              file=sys.stderr)
        return 2
    host = args.host or os.getenv("MAESTRO_HOST", "0.0.0.0")
    port = int(args.port or os.getenv("MAESTRO_PORT", "8000"))
    print(f"[serve] http://{host}:{port}  (health: /health, tools: /tools)")
    uvicorn.run("maestro.server:app", host=host, port=port,
                log_level=os.getenv("MAESTRO_LOG_LEVEL", "info").lower())
    return 0


def _cmd_run_once(args) -> int:
    """Headless single-shot generation. Useful for cron / batch."""
    from .config import load_config
    from .pipeline.run import run_maestro

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run_maestro(
        user_prompt=args.prompt,
        output_path=out,
        source_videos=[Path(p) for p in args.source] or None,
        images=[Path(p) for p in args.image] or None,
        music=Path(args.music) if args.music else None,
        config=load_config(args.config),
        cache_dir=out.parent / "cache",
        trajectory_path=out.with_suffix(".trajectory.jsonl"),
        lesson_path=out.parent / "lessons.jsonl",
    )
    print(json.dumps({
        "output": str(result["output_path"]),
        "report": str(result["report_path"]),
        "n_shots": result["report"]["n_shots"],
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="maestro")
    sub = ap.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke", help="CPU healthcheck (mock pipeline)")
    smoke.add_argument("--prompt", default="a ball is thrown and bounces")
    smoke.set_defaults(func=_cmd_smoke)

    serve = sub.add_parser("serve", help="run the FastAPI server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", default=None)
    serve.set_defaults(func=_cmd_serve)

    run = sub.add_parser("run-once", help="single-shot generation")
    run.add_argument("--prompt", required=True)
    run.add_argument("--source", nargs="*", default=[])
    run.add_argument("--image", nargs="*", default=[])
    run.add_argument("--music", default=None)
    run.add_argument("--output", "-o", default="outputs/demo.mp4")
    run.add_argument("--config", default=None)
    run.set_defaults(func=_cmd_run_once)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":              # pragma: no cover
    raise SystemExit(main())
