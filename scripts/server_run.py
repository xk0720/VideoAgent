#!/usr/bin/env python
"""scripts/server_run.py — manifest-driven batch runner for server boxes (R19).

Inspired by `CutClaw/local_run.py` (https://github.com/GVCLab/CutClaw) — a
single CLI entry that takes one or more (video, audio, instruction) samples
and runs the full pipeline per sample. Extended versus CutClaw with:

  • our hybrid *retrieve+generation* pathway (CutClaw is retrieval-only):
    each sample may set ``generation.enabled`` and ``fallback_threshold``.
  • CSA Arc context (see ``docs/CSA_FRAMEWORK.md``): each sample may attach
    ``intended_arc`` + ``energy_curve`` + ``expected_characters``. After
    the pipeline returns an ``EditingScript`` the runner evaluates it with
    ``tools.metric_tool.arc_coherence`` so the per-sample ``report.json``
    contains an Arc-scale judgement, not just per-segment m1..m6 means.
  • dotted ``--config.NAME=VALUE`` overrides (matches CutClaw's CLI ergonomics).

Two modes:

    # Batch (CutClaw `app.py` analogue)
    python scripts/server_run.py --manifest configs/instructions.example.yaml
    python scripts/server_run.py --manifest configs/instructions.example.yaml \\
                                 --only dark_knight_demo

    # Single-shot (CutClaw `local_run.py` analogue)
    python scripts/server_run.py --single \\
        --video data/videos/x.mp4 --audio data/audio/y.mp3 \\
        --instruction "Make a 10-second high-energy montage." \\
        --output outputs/x.mp4 \\
        --config.compose.generation.fallback_threshold=0.3 \\
        --config.preprocess.parallel.num_workers=8

Outputs per sample:
    <output>.mp4                — the rendered edit
    <output>.trajectory.jsonl   — agent decisions
    <output>.report.json        — per-segment metrics + Arc-scale scores

Plus a top-level ``manifest_summary.json`` aggregating across samples.

Open-source dependencies: stdlib ``argparse`` + ``PyYAML``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

# Make ``longvideoagent`` importable when run as a script.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402  PyYAML — already a hard dep via pydantic + configs

from longvideoagent.pipeline.run import run_pipeline  # noqa: E402
from longvideoagent.tools.metric_tool import arc_coherence  # noqa: E402
from longvideoagent.types import ArcContext, EditingScript  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Dotted overrides — `compose.generation.fallback_threshold=0.3`
# matches CutClaw `--config.PARAM VALUE`.
# ─────────────────────────────────────────────────────────────────────


def _coerce_scalar(s: str) -> Any:
    """Best-effort coerce a CLI string to int / float / bool / None / list."""
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    # JSON for lists / dicts: e.g. '[0.4, 0.8]'
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """`d`, `'a.b.c'`, `value` → ``d['a']['b']['c'] = value`` (creates dicts)."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"Override path {dotted_key!r} collides with "
                             f"non-dict node at key {k!r}.")
    cur[keys[-1]] = value


def _parse_dotted_overrides(items: list[str]) -> dict[str, Any]:
    """Parse a list of ``key.path=value`` strings into a nested override dict.

    Accepts both ``--config.a.b=1`` (single token) and ``--config a.b=1``
    (two tokens after argparse). We canonicalise on the ``a.b=v`` form here.
    """
    out: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Override {raw!r} must be `dotted.key=value`.")
        key, val = raw.split("=", 1)
        key = key.strip()
        if key.startswith("config."):
            key = key[len("config."):]
        _set_nested(out, key, _coerce_scalar(val))
    return out


def _deep_merge(base: dict, ext: dict) -> dict:
    """Deep merge `ext` into a copy of `base` (ext values win)."""
    out = deepcopy(base)
    for k, v in ext.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-sample reporting
# ─────────────────────────────────────────────────────────────────────


def _segment_means(script: EditingScript) -> dict[str, float]:
    segs = list(script.segments)
    if not segs:
        return {k: 0.0 for k in ["m1", "m2", "m3", "m4", "m5", "m6", "validator"]}
    out: dict[str, float] = {}
    for key in ["m1", "m2", "m3", "m4", "m5", "m6", "validator"]:
        vals = [float(s.metric_scores.get(key, 0.0)) for s in segs]
        out[f"mean_{key}"] = sum(vals) / len(vals)
    out["n_segments"] = len(segs)
    out["n_retrieval"] = sum(1 for s in segs if s.source == "retrieval")
    out["n_generation"] = sum(1 for s in segs if s.source == "generation")
    out["accepted_rate"] = sum(1 for s in segs if s.accepted_by_validator) / len(segs)
    out["total_duration_s"] = float(script.total_duration)
    return out


def _build_arc_context(sample_arc: dict[str, Any] | None,
                       instruction: str) -> Optional[ArcContext]:
    """Build an ArcContext from the YAML `arc:` block; None if no shape claimed."""
    if not sample_arc:
        return None
    intended = list(sample_arc.get("intended_arc", []) or [])
    energy = sample_arc.get("energy_curve", []) or []
    energy_curve = [(float(t), float(e)) for t, e in energy] if energy else []
    expected = list(sample_arc.get("expected_characters", []) or [])
    if not intended and not energy_curve and not expected:
        return None
    return ArcContext(
        user_prompt=instruction,
        intended_arc=intended,
        energy_curve=energy_curve,
        expected_characters=expected,
    )


# ─────────────────────────────────────────────────────────────────────
# Sample execution
# ─────────────────────────────────────────────────────────────────────


def _resolve_path(p: str | Path, repo_root: Path) -> Path:
    """Resolve relative paths against the repo root, leave absolutes alone."""
    pp = Path(p)
    return pp if pp.is_absolute() else (repo_root / pp).resolve()


def run_one_sample(
    sample: dict[str, Any],
    *,
    defaults: dict[str, Any],
    global_overrides: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Execute a single manifest entry; return the per-sample report dict."""
    name = sample.get("name") or "unnamed"
    instruction = sample.get("instruction")
    if not instruction:
        raise ValueError(f"sample {name!r}: missing required field `instruction`")

    sources = sample.get("sources") or []
    if not sources:
        raise ValueError(f"sample {name!r}: missing required field `sources` (list of paths)")
    source_paths = [_resolve_path(s, repo_root) for s in sources]
    for sp in source_paths:
        if not sp.exists():
            raise FileNotFoundError(f"sample {name!r}: source video not found: {sp}")

    audio_raw = sample.get("audio")
    audio_path = _resolve_path(audio_raw, repo_root) if audio_raw else None
    if audio_path and not audio_path.exists():
        raise FileNotFoundError(f"sample {name!r}: audio not found: {audio_path}")

    # Cache: per-sample subdir keeps multi-sample runs from clobbering each other.
    cache_root = Path(defaults.get("cache_root", "./.cache"))
    if not cache_root.is_absolute():
        cache_root = (repo_root / cache_root).resolve()
    cache_dir = cache_root / name

    output_block = sample.get("output") or {}
    output_root = Path(defaults.get("output_root", "./outputs/server_run"))
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mp4_path = Path(output_block.get("mp4") or (output_root / f"{name}.mp4"))
    if not mp4_path.is_absolute():
        mp4_path = (repo_root / mp4_path).resolve()
    traj_path = Path(output_block.get("trajectory") or mp4_path.with_suffix(".trajectory.jsonl"))
    if not traj_path.is_absolute():
        traj_path = (repo_root / traj_path).resolve()
    report_path = Path(output_block.get("report") or mp4_path.with_suffix(".report.json"))
    if not report_path.is_absolute():
        report_path = (repo_root / report_path).resolve()

    # Overrides layer order: global manifest defaults → CLI --config.X
    # → per-sample `config_overrides` → sample's generation block.
    per_sample = dict(sample.get("config_overrides") or {})
    per_sample_nested: dict[str, Any] = {}
    for dotted, val in per_sample.items():
        _set_nested(per_sample_nested, dotted, val)

    overrides: dict[str, Any] = {}
    overrides = _deep_merge(overrides, defaults.get("config_overrides_nested", {}))
    overrides = _deep_merge(overrides, global_overrides)
    overrides = _deep_merge(overrides, per_sample_nested)

    gen_block = sample.get("generation") or {}
    if "enabled" in gen_block:
        overrides.setdefault("compose", {}).setdefault("generation", {})["enabled"] = bool(
            gen_block["enabled"])
    if "fallback_threshold" in gen_block:
        overrides.setdefault("compose", {}).setdefault("generation", {})["fallback_threshold"] = float(
            gen_block["fallback_threshold"])

    log_level = sample.get("log_level") or defaults.get("log_level", "INFO")
    self_consistency_k = int(sample.get("self_consistency_k",
                                        defaults.get("self_consistency_k", 1)))
    run_critic = bool(sample.get("run_critic", defaults.get("run_critic", True)))

    print(f"[server_run] ▶ {name}: instruction={instruction!r}")
    print(f"[server_run]   sources={[str(p) for p in source_paths]}")
    print(f"[server_run]   audio={audio_path}")
    print(f"[server_run]   output={mp4_path}")
    print(f"[server_run]   overrides={json.dumps(overrides, sort_keys=True)}")

    t0 = time.perf_counter()
    err: Optional[str] = None
    script: Optional[EditingScript] = None
    try:
        script = run_pipeline(
            source_videos=source_paths,
            user_prompt=instruction,
            output_path=mp4_path,
            music=audio_path,
            cache_dir=cache_dir,
            trajectory_log_path=traj_path,
            overrides=overrides or None,
            log_level=log_level,
            self_consistency_k=self_consistency_k,
            run_critic=run_critic,
        )
    except Exception as e:  # pragma: no cover  — surface server crashes loudly
        err = repr(e)
        print(f"[server_run] ✗ {name} crashed: {err}")
    elapsed = time.perf_counter() - t0

    # ── Build per-sample report ───────────────────────────────────────
    report: dict[str, Any] = {
        "name": name,
        "instruction": instruction,
        "sources": [str(p) for p in source_paths],
        "audio": str(audio_path) if audio_path else None,
        "output_mp4": str(mp4_path),
        "trajectory": str(traj_path),
        "elapsed_s": elapsed,
        "error": err,
    }

    if script is not None:
        report.update(_segment_means(script))
        arc_ctx = _build_arc_context(sample.get("arc"), instruction)
        report["arc_context_provided"] = arc_ctx is not None
        report["arc"] = arc_coherence(script, arc_ctx)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[server_run] ✓ {name}: report → {report_path}")
    return report


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="server_run",
        description=("Batch / single-shot LongVideoEditAgent runner for servers. "
                     "Mirrors CutClaw's local_run.py ergonomics."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("Open-source dependencies:")[0],
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path,
                      help="YAML manifest with a `samples:` list.")
    mode.add_argument("--single", action="store_true",
                      help="Single-shot mode; requires --video / --instruction / --output.")

    # Batch-only filter
    p.add_argument("--only", nargs="+", default=None,
                   help="When --manifest is given, only run samples with these names.")

    # Single-shot inputs
    p.add_argument("--video", action="append", type=Path, dest="single_videos",
                   help="(single mode) Source video. Repeat for multiple sources.")
    p.add_argument("--audio", type=Path, default=None,
                   help="(single mode) Music track. Optional.")
    p.add_argument("--instruction", type=str, default=None,
                   help="(single mode) Editing instruction.")
    p.add_argument("--output", type=Path, default=None,
                   help="(single mode) Output mp4 path. trajectory + report sit beside it.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="(single mode) Cache dir for perception artifacts.")

    # Common knobs
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--no-critic", action="store_true",
                   help="Skip post-run CriticAgent.")
    p.add_argument("--self-consistency-k", type=int, default=1,
                   help="ScreenwriterAgent self-consistency K.")

    # CutClaw-style `--config.PARAM VALUE` accepts either `--config-override key=val`
    # (canonical) or the legacy split form. We expose the canonical form here;
    # README/SERVER_RUN docs show users how to write either.
    p.add_argument("--config-override", action="append", default=[],
                   metavar="dotted.key=value",
                   help="Nested config override; repeatable. "
                        "Example: --config-override compose.generation.fallback_threshold=0.3")
    return p


def _argparse_with_dotted_passthrough(argv: list[str]) -> argparse.Namespace:
    """Translate ``--config.a.b=v`` (CutClaw style) into ``--config-override a.b=v``
    before handing off to argparse, so users can use either form.
    """
    rewritten: list[str] = []
    for tok in argv:
        if tok.startswith("--config.") and "=" in tok:
            rewritten.append("--config-override")
            rewritten.append(tok[len("--config."):])
        else:
            rewritten.append(tok)
    return _build_parser().parse_args(rewritten)


# ─────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    args = _argparse_with_dotted_passthrough(argv if argv is not None else sys.argv[1:])
    repo_root = _REPO
    cli_overrides = _parse_dotted_overrides(args.config_override)

    if args.single:
        if not args.single_videos or not args.instruction or not args.output:
            print("--single requires --video, --instruction, --output.", file=sys.stderr)
            return 2
        synthetic_sample = {
            "name": args.output.stem,
            "sources": [str(p) for p in args.single_videos],
            "audio": str(args.audio) if args.audio else None,
            "instruction": args.instruction,
            "output": {"mp4": str(args.output)},
            "log_level": args.log_level,
            "self_consistency_k": args.self_consistency_k,
            "run_critic": not args.no_critic,
        }
        defaults: dict[str, Any] = {
            "cache_root": str(args.cache_dir) if args.cache_dir else "./.cache",
            "output_root": str(args.output.parent if args.output else "./outputs/server_run"),
        }
        run_one_sample(synthetic_sample, defaults=defaults,
                       global_overrides=cli_overrides, repo_root=repo_root)
        return 0

    # Batch mode
    manifest_path: Path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    data = yaml.safe_load(manifest_path.read_text()) or {}
    defaults = dict(data.get("defaults") or {})
    # Pre-flatten `defaults.config_overrides` (dotted) → nested dict that
    # `run_one_sample` will deep-merge.
    flat_def = defaults.pop("config_overrides", {}) or {}
    nested_def: dict[str, Any] = {}
    for k, v in flat_def.items():
        _set_nested(nested_def, k, v)
    defaults["config_overrides_nested"] = nested_def

    samples = data.get("samples") or []
    if args.only:
        wanted = set(args.only)
        samples = [s for s in samples if s.get("name") in wanted]
        if not samples:
            print(f"--only filter matched 0 samples from {manifest_path}", file=sys.stderr)
            return 2

    reports: list[dict[str, Any]] = []
    for sample in samples:
        try:
            report = run_one_sample(sample, defaults=defaults,
                                    global_overrides=cli_overrides,
                                    repo_root=repo_root)
        except Exception as e:  # pragma: no cover
            name = sample.get("name", "<unnamed>")
            print(f"[server_run] ✗ {name}: {e!r}", file=sys.stderr)
            report = {"name": name, "error": repr(e)}
        reports.append(report)

    # Aggregate summary
    output_root = Path(defaults.get("output_root", "./outputs/server_run"))
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "manifest_summary.json"
    summary = {
        "manifest": str(manifest_path),
        "n_samples": len(reports),
        "n_errors": sum(1 for r in reports if r.get("error")),
        "reports": reports,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[server_run] summary → {summary_path}")
    return 0 if summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
