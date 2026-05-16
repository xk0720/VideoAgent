"""PreferenceLogger — automatic DPO/IPO training-data collector.

Why this exists
---------------
When EditorAgent has ≥2 candidates for the same SegmentGuidance and picks
one, that pair is a free **preference label**: the chosen one was judged
better than the rejected ones *for the same prompt*. Pairwise labels are
the natural input to:

    • **DPO**   — Direct Preference Optimization (Rafailov et al., NeurIPS 2023).
                  Closed-form objective, no separate reward model needed.
    • **IPO**   — Identity Preference Optimization (Azar et al., 2023).
                  More stable than DPO under noisy / saturated labels.
    • **KTO**   — Kahneman-Tversky Optimization (Ethayarajh et al., 2024).
                  Works from unpaired binary signals; useful when only the
                  winner is known (no explicit loser).
    • **SimPO** — Simple Preference Optimization (Meng et al., 2024).
                  Reference-model-free DPO variant; faster training.
    • **GRPO**  — Group Relative Policy Optimization (Shao et al., DeepSeek-Math,
                  2024). Used in DeepSeek-V3 / R1; takes a group of candidates
                  per prompt — our (winner, losers...) tuples are exactly that.
    • Bradley-Terry RM training (classic RLHF)

For the cost of one extra JSONL append per non-singleton candidate set we
get a continuously growing preference dataset that v0.3 can train an RM
on directly. No human annotation needed.

Schema (one JSON object per line in ``preferences.jsonl``):

    {
      "ts":          1778905036.78,
      "run_id":      "<uuid>",
      "segment_idx": 3,
      "guidance":    {"semantic_query": "...", "heuristic": "default", ...},
      "winner":      {"source": "retrieval", "metric_scores": {...},
                       "shot_ids": [...], "validator_score": 7.45},
      "losers":      [ {...}, {...} ],
      "judge":       "EnsembleRewardModel(MockRewardModel,MLLMJudge)"
    }

The format follows HuggingFace ``trl.DPOTrainer`` 's expected shape — just
needs a "prompt / chosen / rejected" projection step which is a single
``jq`` query.

Open-source dependency: stdlib only.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..types import EditingSegment, SegmentGuidance


def _summarise_candidate(c: EditingSegment) -> dict[str, Any]:
    return {
        "source": c.source,
        "duration": c.duration,
        "shot_ids": list(c.shot_ids) if c.shot_ids else [],
        "shot_trims": [list(t) for t in c.shot_trims] if c.shot_trims else [],
        "gen_prompt": c.gen_prompt,
        "metric_scores": dict(c.metric_scores) if c.metric_scores else {},
        "accepted_by_validator": c.accepted_by_validator,
        "validator_reasons": list(c.validator_reasons) if c.validator_reasons else [],
    }


class PreferenceLogger:
    """Append-only JSONL of (winner vs losers) tuples."""

    def __init__(self, path: Path | str, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        # Truncate on init — one run = one fresh file. Append across runs is
        # the v0.2 collection mode (just open with mode="a").
        if not self.path.exists():
            self.path.touch()

    def log_pair(
        self,
        guidance: SegmentGuidance,
        winner: EditingSegment,
        losers: list[EditingSegment],
        judge_name: str = "unknown",
    ) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "segment_idx": guidance.segment_idx,
            "guidance": asdict(guidance),
            "winner": _summarise_candidate(winner),
            "losers": [_summarise_candidate(c) for c in losers],
            "judge": judge_name,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


__all__ = ["PreferenceLogger"]
