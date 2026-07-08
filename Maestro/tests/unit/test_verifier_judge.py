"""VerifierAgent blind pairwise confirmation (NEWTON borrow).

The judge only VETOES a MARGINAL metric win; it never rescues a loss and is
never consulted on a decisive win. Mock-tie keeps the metric verdict."""
from pathlib import Path

from maestro.agents.verifier import VerifierAgent
from maestro.types import CandidateClip, ShotSpec


def _clip(total):
    c = CandidateClip(shot_idx=0, video_path=Path("x.mp4"))
    c.metric_scores = {"weighted_total": total}
    return c


def _spec():
    return ShotSpec(shot_idx=0, duration=1.0, prompt="a ball falls")


class _SeqJudge:
    """compare() replies from a fixed sequence: [fwd(cand,best), rev(best,cand)]."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def compare(self, a, b, spec):
        r = self.replies[self.calls]
        self.calls += 1
        return r


def test_marginal_win_vetoed_when_judge_prefers_old_best():
    judge = _SeqJudge([-1, 1])            # fwd: best wins; rev: best wins -> net -2
    v = VerifierAgent(judge=judge, margin=0.02)
    assert v.is_better(_clip(0.505), _clip(0.50), spec=_spec()) is False
    assert judge.calls == 2               # bidirectional (order-debiased)


def test_marginal_win_kept_on_judge_tie():
    judge = _SeqJudge([1, 1])             # fwd: cand; rev: cand? -> net 0 (tie)
    v = VerifierAgent(judge=judge, margin=0.02)
    assert v.is_better(_clip(0.505), _clip(0.50), spec=_spec()) is True


def test_decisive_win_skips_the_judge():
    judge = _SeqJudge([-1, 1])            # would veto if consulted
    v = VerifierAgent(judge=judge, margin=0.02)
    assert v.is_better(_clip(0.60), _clip(0.50), spec=_spec()) is True
    assert judge.calls == 0               # outside the noise band -> no VLM spend


def test_metric_loss_never_rescued_by_judge():
    judge = _SeqJudge([1, -1])            # judge LOVES the candidate
    v = VerifierAgent(judge=judge, margin=0.02)
    assert v.is_better(_clip(0.49), _clip(0.50), spec=_spec()) is False
    assert judge.calls == 0               # monotonicity is a hard rule


def test_no_spec_or_no_judge_keeps_pure_metric_gate():
    v = VerifierAgent()                   # no judge wired (default)
    assert v.is_better(_clip(0.505), _clip(0.50)) is True
    v2 = VerifierAgent(judge=_SeqJudge([-1, 1]))
    assert v2.is_better(_clip(0.505), _clip(0.50)) is True  # spec=None -> skip
