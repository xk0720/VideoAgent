"""Shared deterministic stub trainer.

Real TRL trainers do gradient updates. The stub:
    • iterates over the dataset for ``n_epochs``
    • tracks a fake "loss" that decreases as the dataset is consumed
    • writes a JSON metrics file to ``output_dir/metrics.json``
    • writes a JSON "ckpt manifest" to ``output_dir/ckpt.json``

This is enough to drive end-to-end tests, prove the data plumbing works,
and validate the configs — and the real backend slots in as a one-line swap.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class StubMetrics:
    stage: str
    backend: str
    n_records: int
    n_epochs: int
    final_loss: float
    elapsed_s: float
    extras: dict[str, Any] = field(default_factory=dict)


def stub_train(
    stage: str,
    dataset: Iterable[dict[str, Any]],
    n_epochs: int = 1,
    output_dir: Path | None = None,
    extras: dict[str, Any] | None = None,
) -> StubMetrics:
    records = list(dataset)
    n = len(records)
    t0 = time.perf_counter()
    # Fake decreasing loss: starts at log(n+e), halves each epoch.
    base = math.log(n + math.e)
    losses = [base / (2 ** i) for i in range(n_epochs)]
    elapsed = time.perf_counter() - t0
    metrics = StubMetrics(
        stage=stage, backend="stub", n_records=n, n_epochs=n_epochs,
        final_loss=(losses[-1] if losses else 0.0), elapsed_s=elapsed,
        extras=dict(extras or {}),
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(asdict(metrics) | {"losses_per_epoch": losses}, indent=2),
            encoding="utf-8",
        )
        (output_dir / "ckpt.json").write_text(
            json.dumps({"stage": stage, "backend": "stub", "n_records": n,
                        "n_epochs": n_epochs, "note": "stub checkpoint — "
                        "swap backend='trl' to produce real weights."}),
            encoding="utf-8",
        )
    return metrics
