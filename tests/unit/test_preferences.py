"""PreferenceLogger tests — DPO/IPO-ready (winner, losers) JSONL."""
from __future__ import annotations

from pathlib import Path

from longvideoagent.types import EditingSegment, SegmentGuidance
from longvideoagent.utils.preferences import PreferenceLogger


def _g():
    return SegmentGuidance(segment_idx=3, parent_section_idx=0,
                           semantic_query="action chase",
                           editing_heuristic="motion_continuity")


def _seg(idx: int, source="retrieval"):
    return EditingSegment(segment_idx=idx, source=source, duration=1.0,
                          shot_ids=[f"s{idx}"], shot_trims=[(0.0, 1.0)],
                          metric_scores={"validator": 8.0 if idx == 0 else 5.0},
                          accepted_by_validator=(idx == 0))


def test_log_pair_writes_jsonl(tmp_path: Path):
    p = tmp_path / "prefs.jsonl"
    pl = PreferenceLogger(p, run_id="abc")
    pl.log_pair(_g(), winner=_seg(0), losers=[_seg(1), _seg(2)], judge_name="EnsembleRM")
    records = pl.read_all()
    assert len(records) == 1
    rec = records[0]
    assert rec["segment_idx"] == 3
    assert rec["run_id"] == "abc"
    assert rec["winner"]["shot_ids"] == ["s0"]
    assert len(rec["losers"]) == 2
    assert rec["judge"] == "EnsembleRM"


def test_append_across_segments(tmp_path: Path):
    p = tmp_path / "prefs.jsonl"
    pl = PreferenceLogger(p)
    pl.log_pair(_g(), winner=_seg(0), losers=[_seg(1)])
    pl.log_pair(_g(), winner=_seg(0), losers=[_seg(2)])
    assert len(pl.read_all()) == 2
