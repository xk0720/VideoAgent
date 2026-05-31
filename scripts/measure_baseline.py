#!/usr/bin/env python
"""scripts/measure_baseline.py — v0.2 baseline measurement (B in the roadmap).

What this script proves (or disproves):

    The framework's central architectural claim is **hybrid retrieval + generation
    outperforms either alone**, because generation is conditioned on neighbour
    end-frame / flow / character anchors from retrieved shots (design doc §0,
    §5.3, §7.2). This script puts that claim to the test by sweeping
    ``compose.generation.fallback_threshold`` to coerce the EditorAgent into
    three regimes — R-only, G-only, Hybrid — and comparing the resulting
    ``EditingScript`` quality with the metrics we already have.

We deliberately do NOT add new metrics, new agents, or new training stages.
The whole purpose of B is to *measure with what we have* before adding more.

Output:
    outputs/baseline/
        runs.jsonl                 — one line per (regime, seed) with per-run
                                      metrics, cross-segment-boundary metrics,
                                      and the EditingScript summary
        report.json                — aggregated means / stds per regime
        report.md                  — human-readable findings (this becomes
                                      docs/BASELINE_v0_2.md after a manual review)

Usage:
    python scripts/measure_baseline.py \\
        --source tests/fixtures/tiny_clip.mp4 \\
        --output-dir outputs/baseline \\
        --prompts \\
            "Make a 4-second high-energy montage" \\
            "Make a calm cinematic establishing sequence" \\
            "Make a fast action cut with motion continuity" \\
        --seeds 0 1 2

References (design-doc lines we are *measuring against*):
    • §0: hybrid retrieval+generation is the core mechanism
    • §5.3: GenerationTool conditioned on neighbour anchor frames + flow
    • §7.2: OmniWeaving chosen for free-form text+multi-image+video conditioning
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Make ``longvideoagent`` importable when run as a script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from longvideoagent.pipeline.run import run_pipeline
from longvideoagent.types import EditingScript, EditingSegment


# ─────────────────────────────────────────────────────────────────────
# Regime configuration
#
# The three regimes are produced by sweeping ONLY the fallback_threshold.
# This isolates the routing decision from every other knob.
# ─────────────────────────────────────────────────────────────────────

REGIMES: dict[str, dict[str, Any]] = {
    "R_only": {
        # threshold = -1 ⇒ no guidance feasibility ever drops below it ⇒
        # EditorEnv's first_action is always "retrieve" and the LLM almost
        # never picks "generate".
        "compose": {"generation": {"fallback_threshold": -1.0}},
    },
    "G_only": {
        # threshold = 2.0 ⇒ feasibility (∈ [0,1]) always below it ⇒
        # first_action is always "generate".
        "compose": {"generation": {"fallback_threshold": 2.0}},
    },
    "Hybrid": {
        # Default — feasibility-aware routing. Set explicitly so we never
        # mistake a default-change for an experiment intervention.
        "compose": {"generation": {"fallback_threshold": 0.4}},
    },
}


# ─────────────────────────────────────────────────────────────────────
# Per-run measurement
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SegmentBoundary:
    """A boundary between two consecutive segments in the script.

    The hybrid claim's most-loadbearing prediction lives here: if generation
    is properly conditioned on the previous segment's end-frame/flow,
    retrieve→generate boundaries should *not* be markedly worse than
    retrieve→retrieve boundaries on motion (m3) and framing (m4).
    """
    idx: int
    prev_source: str
    next_source: str
    transition_type: str            # "R→R" / "R→G" / "G→R" / "G→G"
    # We don't have the actual cross-segment metric numbers post-hoc (they're
    # computed *within* RetrievalTool's beam search). The next best proxy
    # available without re-running perception is the metric_scores of the
    # SECOND segment, which by m2/m3/m4 definition depends on the previous
    # segment's tail features. So m2..m4 of segment[i+1] is our best signal
    # for the "i→i+1" transition quality.
    next_segment_m1: float
    next_segment_m2: float
    next_segment_m3: float
    next_segment_m4: float


@dataclass
class RunStats:
    regime: str
    seed: int
    user_prompt: str
    n_segments: int
    n_retrieval: int
    n_generation: int
    duration_s: float
    elapsed_s: float
    # Per-segment metric averages
    mean_m1: float
    mean_m2: float
    mean_m3: float
    mean_m4: float
    mean_m5: float
    mean_m6: float
    mean_validator: float
    accepted_rate: float            # fraction of segments accepted by validator
    # Boundary breakdown
    boundaries: list[SegmentBoundary] = field(default_factory=list)
    # Hybrid claim probes (means)
    mean_R_to_R_m3: float | None = None
    mean_R_to_G_m3: float | None = None
    mean_R_to_G_m4: float | None = None
    mean_G_to_R_m3: float | None = None
    mean_G_to_R_m4: float | None = None


def _safe_mean(xs: list[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


def measure_run(script: EditingScript, regime: str, seed: int, prompt: str,
                elapsed_s: float) -> RunStats:
    segs = list(script.segments)
    if not segs:
        return RunStats(
            regime=regime, seed=seed, user_prompt=prompt,
            n_segments=0, n_retrieval=0, n_generation=0,
            duration_s=0.0, elapsed_s=elapsed_s,
            mean_m1=0, mean_m2=0, mean_m3=0, mean_m4=0, mean_m5=0, mean_m6=0,
            mean_validator=0, accepted_rate=0.0,
        )

    n_retrieval = sum(1 for s in segs if s.source == "retrieval")
    n_generation = sum(1 for s in segs if s.source == "generation")

    def _mean(key: str) -> float:
        vals = [float(s.metric_scores.get(key, 0.0)) for s in segs]
        return float(sum(vals) / len(vals)) if vals else 0.0

    boundaries: list[SegmentBoundary] = []
    for i in range(len(segs) - 1):
        prev_src = segs[i].source
        next_src = segs[i + 1].source
        next_seg = segs[i + 1]
        transition = (
            ("R" if prev_src == "retrieval" else "G")
            + "→"
            + ("R" if next_src == "retrieval" else "G")
        )
        boundaries.append(SegmentBoundary(
            idx=i,
            prev_source=prev_src, next_source=next_src,
            transition_type=transition,
            next_segment_m1=float(next_seg.metric_scores.get("m1", 0.0)),
            next_segment_m2=float(next_seg.metric_scores.get("m2", 0.0)),
            next_segment_m3=float(next_seg.metric_scores.get("m3", 0.0)),
            next_segment_m4=float(next_seg.metric_scores.get("m4", 0.0)),
        ))

    by_type_m3: dict[str, list[float]] = {"R→R": [], "R→G": [], "G→R": [], "G→G": []}
    by_type_m4: dict[str, list[float]] = {"R→R": [], "R→G": [], "G→R": [], "G→G": []}
    for b in boundaries:
        by_type_m3[b.transition_type].append(b.next_segment_m3)
        by_type_m4[b.transition_type].append(b.next_segment_m4)

    return RunStats(
        regime=regime, seed=seed, user_prompt=prompt,
        n_segments=len(segs),
        n_retrieval=n_retrieval, n_generation=n_generation,
        duration_s=float(script.total_duration),
        elapsed_s=elapsed_s,
        mean_m1=_mean("m1"), mean_m2=_mean("m2"),
        mean_m3=_mean("m3"), mean_m4=_mean("m4"),
        mean_m5=_mean("m5"), mean_m6=_mean("m6"),
        mean_validator=_mean("validator"),
        accepted_rate=sum(1 for s in segs if s.accepted_by_validator) / len(segs),
        boundaries=boundaries,
        mean_R_to_R_m3=_safe_mean(by_type_m3["R→R"]),
        mean_R_to_G_m3=_safe_mean(by_type_m3["R→G"]),
        mean_R_to_G_m4=_safe_mean(by_type_m4["R→G"]),
        mean_G_to_R_m3=_safe_mean(by_type_m3["G→R"]),
        mean_G_to_R_m4=_safe_mean(by_type_m4["G→R"]),
    )


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────


def aggregate(runs: list[RunStats]) -> dict[str, dict[str, Any]]:
    by_regime: dict[str, list[RunStats]] = {}
    for r in runs:
        by_regime.setdefault(r.regime, []).append(r)

    report: dict[str, dict[str, Any]] = {}
    for regime, group in by_regime.items():
        report[regime] = {
            "n_runs": len(group),
            "mean_segments": float(statistics.mean(r.n_segments for r in group)),
            "mean_pct_generation": float(statistics.mean(
                (r.n_generation / max(1, r.n_segments)) for r in group
            )),
            "mean_accepted_rate": float(statistics.mean(r.accepted_rate for r in group)),
            "mean_m1": float(statistics.mean(r.mean_m1 for r in group)),
            "mean_m2": float(statistics.mean(r.mean_m2 for r in group)),
            "mean_m3": float(statistics.mean(r.mean_m3 for r in group)),
            "mean_m4": float(statistics.mean(r.mean_m4 for r in group)),
            "mean_m5": float(statistics.mean(r.mean_m5 for r in group)),
            "mean_m6": float(statistics.mean(r.mean_m6 for r in group)),
            "mean_validator": float(statistics.mean(r.mean_validator for r in group)),
            "mean_duration_s": float(statistics.mean(r.duration_s for r in group)),
            "mean_elapsed_s": float(statistics.mean(r.elapsed_s for r in group)),
        }

        # Transition-type analysis only makes sense if the regime actually
        # produces mixed transitions (Hybrid). Still surface the means.
        def _avg_optional(key: str) -> float | None:
            vs = [getattr(r, key) for r in group if getattr(r, key) is not None]
            return float(statistics.mean(vs)) if vs else None

        report[regime].update({
            "mean_R_to_R_m3": _avg_optional("mean_R_to_R_m3"),
            "mean_R_to_G_m3": _avg_optional("mean_R_to_G_m3"),
            "mean_R_to_G_m4": _avg_optional("mean_R_to_G_m4"),
            "mean_G_to_R_m3": _avg_optional("mean_G_to_R_m3"),
            "mean_G_to_R_m4": _avg_optional("mean_G_to_R_m4"),
        })

    return report


def write_markdown_report(
    aggregated: dict[str, dict[str, Any]],
    raw_runs: list[RunStats],
    output_path: Path,
) -> None:
    """Emit a markdown report — *with* the hybrid claim verdict spelled out."""
    if not aggregated:
        output_path.write_text("# Baseline report (empty)\n\nNo runs collected.\n")
        return

    R = aggregated.get("R_only", {})
    G = aggregated.get("G_only", {})
    H = aggregated.get("Hybrid", {})

    def _f(d: dict, k: str) -> str:
        v = d.get(k)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    def _pct(d: dict, k: str) -> str:
        v = d.get(k)
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"

    lines: list[str] = [
        "# v0.2 Baseline Measurement",
        "",
        "> Auto-generated by `scripts/measure_baseline.py`. Compares the v0.2",
        "> mock pipeline under three forced routing regimes — **R-only**,",
        "> **G-only**, **Hybrid** — to test the design doc's central claim",
        "> (§0): hybrid retrieval+generation should *beat* either alone.",
        "",
        f"- regimes measured: **{list(aggregated)}**",
        f"- total runs in this experiment: **{sum(d['n_runs'] for d in aggregated.values())}**",
        "",
        "## 1. Per-regime summary",
        "",
        "| Metric | R-only | G-only | Hybrid |",
        "|---|---|---|---|",
        f"| # runs | {R.get('n_runs','—')} | {G.get('n_runs','—')} | {H.get('n_runs','—')} |",
        f"| mean #segments | {_f(R,'mean_segments')} | {_f(G,'mean_segments')} | {_f(H,'mean_segments')} |",
        f"| %generation | {_pct(R,'mean_pct_generation')} | {_pct(G,'mean_pct_generation')} | {_pct(H,'mean_pct_generation')} |",
        f"| accepted rate | {_pct(R,'mean_accepted_rate')} | {_pct(G,'mean_accepted_rate')} | {_pct(H,'mean_accepted_rate')} |",
        f"| m1 prompt-relevance | {_f(R,'mean_m1')} | {_f(G,'mean_m1')} | {_f(H,'mean_m1')} |",
        f"| m2 seg-consistency  | {_f(R,'mean_m2')} | {_f(G,'mean_m2')} | {_f(H,'mean_m2')} |",
        f"| m3 motion-continuity| {_f(R,'mean_m3')} | {_f(G,'mean_m3')} | {_f(H,'mean_m3')} |",
        f"| m4 framing          | {_f(R,'mean_m4')} | {_f(G,'mean_m4')} | {_f(H,'mean_m4')} |",
        f"| m5 beat-sync        | {_f(R,'mean_m5')} | {_f(G,'mean_m5')} | {_f(H,'mean_m5')} |",
        f"| m6 energy           | {_f(R,'mean_m6')} | {_f(G,'mean_m6')} | {_f(H,'mean_m6')} |",
        f"| **validator (1-10)**| {_f(R,'mean_validator')} | {_f(G,'mean_validator')} | {_f(H,'mean_validator')} |",
        "",
        "## 2. Cross-segment boundary quality (Hybrid claim probe)",
        "",
        "Design doc §5.3 / §7.2 promise: generation is conditioned on neighbour",
        "anchor frames + flow + character refs, so **retrieve→generate** and",
        "**generate→retrieve** transitions should not be markedly worse than",
        "**retrieve→retrieve** transitions on m3 (motion) and m4 (framing).",
        "If they are, the conditioning isn't actually buying us anything in mock mode.",
        "",
        "| Transition type | m3 (Hybrid) | m4 (Hybrid) |",
        "|---|---|---|",
        f"| R → R | {_f(H,'mean_R_to_R_m3')} | (n/a in this collection) |",
        f"| R → G | {_f(H,'mean_R_to_G_m3')} | {_f(H,'mean_R_to_G_m4')} |",
        f"| G → R | {_f(H,'mean_G_to_R_m3')} | {_f(H,'mean_G_to_R_m4')} |",
        "",
        "## 3. The hybrid verdict",
        "",
    ]

    # Verdict logic
    if R and G and H:
        delta_val_vs_R = H.get("mean_validator", 0.0) - R.get("mean_validator", 0.0)
        delta_val_vs_G = H.get("mean_validator", 0.0) - G.get("mean_validator", 0.0)
        verdict_lines = [
            f"Hybrid validator − R-only = **{delta_val_vs_R:+.3f}**",
            f"Hybrid validator − G-only = **{delta_val_vs_G:+.3f}**",
            "",
        ]
        if delta_val_vs_R > 0.1 and delta_val_vs_G > 0.1:
            verdict_lines.append(
                "✅ **Hybrid wins.** The current v0.2 routing (even with mocked LLM)"
                " produces edits that score better than either pure-retrieval or"
                " pure-generation under the existing validator. The architectural"
                " bet pays off in mock mode — next step is to repeat under real"
                " LLM + real video-gen backends."
            )
        elif abs(delta_val_vs_R) <= 0.1 and abs(delta_val_vs_G) <= 0.1:
            verdict_lines.append(
                "⚠️  **Hybrid is statistical noise w.r.t. either alone.** Under the"
                " mocked v0.2 stack, the routing decision doesn't move validator"
                " scores. This is either (a) the mock pipeline is too flat to"
                " expose routing differences, or (b) the EditorAgent's routing"
                " rule is too coarse. Next experiment: real LLM + force a wider"
                " feasibility distribution."
            )
        else:
            verdict_lines.append(
                "❌ **Hybrid is *worse* than one of the pure baselines.** This is"
                " a falsification signal — investigate which boundary type drags"
                " the average down. The cross-segment table above usually"
                " explains why: if R→G or G→R m3/m4 is much lower than R→R,"
                " the generation anchor is failing."
            )
        lines.extend(verdict_lines)
    else:
        lines.append("(Not all three regimes were measured — verdict skipped.)")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        prog="measure_baseline",
        description="Sweep R-only / G-only / Hybrid on the v0.2 mock pipeline.",
    )
    p.add_argument("--source", required=True, type=Path, nargs="+",
                   help="Source video(s); the mock pipeline only needs one.")
    p.add_argument("--prompts", required=True, type=str, nargs="+",
                   help="One or more user prompts to sweep across.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Seeds for ensuring multiple independent runs.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    p.add_argument("--regimes", type=str, nargs="+", default=list(REGIMES),
                   choices=list(REGIMES))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "runs.jsonl"
    report_json = args.output_dir / "report.json"
    report_md = args.output_dir / "report.md"
    runs_path.write_text("")    # truncate

    all_runs: list[RunStats] = []
    for regime in args.regimes:
        overrides = REGIMES[regime]
        for prompt in args.prompts:
            for seed in args.seeds:
                # The seed enters two places: as the trajectory file suffix
                # (so runs don't clobber each other) AND as ``random_seed``
                # for the mock perception.
                tmp = tempfile.mkdtemp(prefix=f"lva_baseline_{regime}_")
                tmp = Path(tmp)
                cache = tmp / "cache"
                output = tmp / "out.mp4"
                traj = tmp / "trajectory.jsonl"
                # Combine the regime overrides with the seed override.
                this_overrides = json.loads(json.dumps(overrides))   # deep copy
                this_overrides.setdefault("random_seed", seed)

                t0 = time.perf_counter()
                try:
                    script = run_pipeline(
                        source_videos=args.source,
                        user_prompt=prompt,
                        output_path=output,
                        cache_dir=cache,
                        trajectory_log_path=traj,
                        overrides=this_overrides,
                        log_level="WARNING",            # keep stdout clean
                        run_critic=False,                # avoid noise + writes
                    )
                except Exception as e:                  # pragma: no cover
                    print(f"[ERROR] {regime}/{prompt[:30]!r}/seed={seed}: {e!r}")
                    shutil.rmtree(tmp, ignore_errors=True)
                    continue
                elapsed = time.perf_counter() - t0

                stats = measure_run(script, regime, seed, prompt, elapsed)
                all_runs.append(stats)
                with runs_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(stats), default=str) + "\n")
                shutil.rmtree(tmp, ignore_errors=True)
                print(f"[OK] {regime} / seed={seed} / prompt='{prompt[:30]}…' "
                      f"⇒ validator={stats.mean_validator:.2f} "
                      f"%gen={(stats.n_generation/max(1,stats.n_segments))*100:.0f}% "
                      f"({elapsed:.1f}s)")

    if not all_runs:
        print("No runs completed — see errors above.")
        return 1

    aggregated = aggregate(all_runs)
    report_json.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    write_markdown_report(aggregated, all_runs, report_md)
    print(f"\nReport: {report_md}\nJSON:   {report_json}\nRaw:    {runs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
