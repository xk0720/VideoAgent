"""Internal-component audits.

These verify the *semantics* of building blocks the higher loop relies on —
properties that are easy to silently regress because they don't show up in
end-to-end pass/fail:

  1. **Embeddings** — hash-based bag-of-tokens, but does it actually rank
     semantically related text closer than unrelated text? Determinism, L2
     normalization, empty-input safety.
  2. **Tournament** — VISTA-style bidirectional pairwise judge: does it
     actually pick the strongest candidate AND survive a position-biased mock
     judge?
  3. **C6 verification signal path** — annotation → track extractor →
     reliability gate → law checks → measured verdict. The audit pins the
     property the self-improve loop depends on: violations are detected
     from OBSERVED tracks (and clear after refinement), never from
     generation metadata.
  4. **PlanValidator CCV** — Critique → Correct → Verify converges on
     ungroundable refs (i.e., the second iter actually passes).
"""
from __future__ import annotations

from pathlib import Path

from maestro.agents.director import DirectorAgent
from maestro.agents.generator import GeneratorAgent
from maestro.agents.physics_planner import PhysicsPlannerAgent
from maestro.agents.plan_validator import PlanValidatorAgent
from maestro.agents.screenwriter import ScreenwriterAgent
from maestro.critics.tournament import Tournament
from maestro.embeddings import cosine, embed_text
from maestro.physics.annotate import annotate_physics
from maestro.pipeline.plan import plan_shots
from maestro.types import AssetMemory, CandidateClip, Identity, ShotSpec


# ─────────────────────────────────────────────────────────────────────────────
# 1) Embeddings
# ─────────────────────────────────────────────────────────────────────────────
def test_embedding_is_deterministic_and_normalized():
    a = embed_text("a ball is thrown and bounces")
    b = embed_text("a ball is thrown and bounces")
    # determinism
    assert (a == b).all()
    # L2 norm == 1 for non-empty input
    assert abs(float((a * a).sum()) - 1.0) < 1e-5


def test_embedding_empty_input_safe():
    z = embed_text("")
    assert float(z.sum()) == 0.0
    # cosine guards against zero-norm inputs (returns 0, not NaN).
    assert cosine(z, embed_text("hello")) == 0.0


def test_embedding_ranks_semantic_overlap_higher():
    """Same-domain text must outrank unrelated text. This is what makes
    LessonLibrary.retrieve actually pick relevant lessons rather than noise.
    """
    q = embed_text("a ball is bouncing on the floor")
    near = embed_text("a ball is thrown and bounces off a wall")
    far = embed_text("a cat eats sushi at midnight on a roof")
    s_near = cosine(q, near)
    s_far = cosine(q, far)
    assert s_near > s_far, (s_near, s_far)
    # And the gap should be meaningful (not within float noise).
    assert s_near - s_far > 0.1, (s_near, s_far)


def test_embedding_cjk_per_character_tokenization():
    """A Chinese prompt must NOT collapse to one giant token (that would make
    every CJK-vs-CJK cosine = 0 unless strings match literally). With per-char
    BoW, two prompts sharing 水 / 倒 should rank above an unrelated prompt.
    """
    q = embed_text("水从瓶子里倒出来")
    near = embed_text("一杯水倒在地上")
    far = embed_text("一只猫在屋顶上吃寿司")
    assert cosine(q, near) > cosine(q, far) + 0.05, \
        (cosine(q, near), cosine(q, far))


def test_embedding_mixes_latin_and_cjk():
    """A bilingual prompt must produce a non-trivial vector that responds to
    BOTH the Latin words and the CJK chars (Maestro's user prompts are
    routinely bilingual)."""
    v_mixed = embed_text("a ball 水 bounces 倒")
    v_only_lat = embed_text("a ball bounces")
    v_only_cjk = embed_text("水 倒")
    # The mixed embedding should be more similar to each half than the halves
    # are to each other.
    sim_latin = cosine(v_mixed, v_only_lat)
    sim_cjk = cosine(v_mixed, v_only_cjk)
    halves = cosine(v_only_lat, v_only_cjk)
    assert sim_latin > halves
    assert sim_cjk > halves


def test_cosine_in_zero_to_one_for_bow():
    """Bag-of-tokens cosine is bounded in [0,1] (non-negative buckets)."""
    a = embed_text("the quick brown fox")
    b = embed_text("a lazy old dog")
    c = cosine(a, b)
    assert 0.0 <= c <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2) Tournament — bidirectional debias
# ─────────────────────────────────────────────────────────────────────────────
def _clip(idx, total):
    c = CandidateClip(shot_idx=idx, video_path=Path("x.mp4"))
    c.metric_scores = {"weighted_total": total}
    return c


class _PositionBiasedJudge:
    """Mock judge that always says "the FIRST arg wins" regardless of quality.
    A non-debiased pick would crown the first candidate every time; the
    bidirectional Tournament must counteract that bias and reach a tie/
    correct winner.
    """

    name = "biased"

    def compare(self, a, b, spec):
        return 1   # first arg always wins → +1 fwd, +1 rev → net 0 (tie)


def test_tournament_neutralizes_a_position_biased_judge():
    """If the underlying judge is fully position-biased, bidirectional swap
    must yield ties — not crown the candidate that was simply listed first."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="p")
    cands = [_clip(0, 0.1), _clip(0, 0.9)]    # second is objectively stronger
    biased = _PositionBiasedJudge()

    # _bidirectional should net to 0 (tie) under a fully-biased judge —
    # i.e., the bias was cancelled, even if the result then can't pick by
    # quality (the judge is information-free anyway).
    tournament = Tournament(judge=biased)
    net = tournament._bidirectional(cands[0], cands[1], spec)
    assert net == 0, f"bidirectional did not cancel position bias: {net}"


def test_tournament_picks_strongest_under_honest_judge():
    """The DEFAULT mock judge ranks by weighted_total. Verify the highest-
    scoring candidate wins from arbitrary positions in the list."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="p")
    cands = [_clip(0, 0.3), _clip(0, 0.9), _clip(0, 0.5), _clip(0, 0.7)]
    best = Tournament().select(cands, spec)
    assert best.metric_scores["weighted_total"] == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 3) C6 verification signal path — observed tracks, not metadata
# ─────────────────────────────────────────────────────────────────────────────
def test_verification_signal_comes_from_observed_tracks(tmp_path: Path):
    """C6's lifeline: the measured verdict must derive from the extracted
    track (an unrepaired clip contains a mid-air reversal; a clip whose body
    records the APPLIED physics fix is law-consistent) and NOT from the
    revision counter or any quality claim the generator wrote about itself.
    A regression that re-couples verdicts to revision metadata would turn
    the loop into a clock and break it silently.
    """
    from maestro.critics.physics_consistency import PhysicsConsistencyCritic

    spec = ShotSpec(shot_idx=0, duration=2.0, prompt="a ball is thrown")
    spec.physics_annotation = annotate_physics(spec)
    critic = PhysicsConsistencyCritic()

    c0 = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    critic.review(c0, spec, fps=8)
    measured0 = [v for v in c0.physics_verdicts if v.source == "law_verifier"]
    assert measured0, "unrepaired violation must be detected from the track"

    c1 = GeneratorAgent().run(spec, tmp_path, revision=1, seed=0, fps=8,
                              extra_prompt="one continuous passive trajectory")
    critic.review(c1, spec, fps=8)
    measured1 = [v for v in c1.physics_verdicts if v.source == "law_verifier"]
    assert not measured1, "repaired clip must clear the measured check"

    # Negative control (signal honesty): a regeneration that never applies
    # the fix must stay flagged — a clock-driven mock would 'clear' here.
    c2 = GeneratorAgent().run(spec, tmp_path, revision=2, seed=5, fps=8)
    critic.review(c2, spec, fps=8)
    assert [v for v in c2.physics_verdicts if v.source == "law_verifier"], \
        "fix-free regeneration must NOT clear the measured check"

    # The generator's own metadata must contain no physics claims at all.
    body = c0.video_path.read_text(encoding="utf-8", errors="ignore")
    assert "control_signal" not in body


def test_verifier_reports_explicit_coverage(tmp_path: Path):
    """S3 transparency: entities outside the measurement tier are explicit
    deferrals in the coverage report — partial verification must never read
    as full verification."""
    from maestro.physics.verifier import PhysicsFromPixelsVerifier

    spec = ShotSpec(shot_idx=0, duration=2.0,
                    prompt="a ball falls while a person runs")
    spec.physics_annotation = annotate_physics(spec)
    clip = GeneratorAgent().run(spec, tmp_path, revision=1, seed=0, fps=8)
    result = PhysicsFromPixelsVerifier().verify(clip, spec, fps=8)
    assert "world_model" in result.coverage          # person: deferred
    assert "person" in result.coverage["world_model"]
    measured = {e.entity for e in result.entities if e.measured}
    assert "person" not in measured


# ─────────────────────────────────────────────────────────────────────────────
# 3b) Signal honesty — the loop's improvement signal is CONTENT-derived
# ─────────────────────────────────────────────────────────────────────────────
def test_loop_signal_is_content_derived(tmp_path: Path):
    """Signal-honesty regression guard (parent repo docs/CRITICAL_REVIEW.md
    §meta-error-1): the loop's convergence must come from repair instructions
    ACTUALLY APPLIED to the artifact, never from the revision counter. With a
    stub generator that silently IGNORES all repair conditioning, a
    clock-driven mock would still "converge"; the honest mock must never
    accept a candidate (every round falls through to the escape hatch) and
    the defects must persist.
    """
    from maestro.agents.refiner import RefinerAgent
    from maestro.agents.verifier import VerifierAgent
    from maestro.critics.board import ReviewBoard
    from maestro.critics.consistency import ConsistencyCritic
    from maestro.critics.physics import PhysicsCritic
    from maestro.critics.physics_consistency import PhysicsConsistencyCritic
    from maestro.critics.rhythm import RhythmCritic
    from maestro.critics.semantic import SemanticCritic
    from maestro.pipeline.generate_loop import generate_shot

    class _FixIgnoringGenerator(GeneratorAgent):
        """Simulates a generator that does not respond to repair
        conditioning: extra_prompt and the first-frame anchor are dropped, so
        no fix ever lands in the artifact."""

        def run(self, spec, cache_dir, revision=0, seed=0, extra_prompt="",
                first_frame=None, **kwargs):
            return super().run(spec, cache_dir, revision=revision, seed=seed,
                               extra_prompt="", first_frame=None, **kwargs)

    def _run(generator, cache_dir):
        spec = ShotSpec(shot_idx=0, duration=2.0,
                        prompt="a ball is thrown and bounces off a wall")
        spec.physics_annotation = annotate_physics(spec)
        board = ReviewBoard([
            SemanticCritic(), PhysicsCritic(), PhysicsConsistencyCritic(),
            ConsistencyCritic(), RhythmCritic(),
        ])
        return generate_shot(
            spec, board, generator, RefinerAgent(), VerifierAgent(),
            cache_dir, n_candidates=1, max_revisions=2, k_retries=1,
        )

    # Control: the real loop threads fix text into regenerations → converges.
    honest = _run(GeneratorAgent(), tmp_path / "honest")
    assert honest.converged and not honest.escape_hatched
    assert honest.clip.physics_verdicts == []
    assert not honest.clip.checklist.failed_items

    # Fix-ignoring world: regeneration without the fix must NOT "improve".
    broken = _run(_FixIgnoringGenerator(), tmp_path / "broken")
    assert not broken.converged, "clock-driven convergence — signal leaked"
    # No candidate was ever accepted: every revision fell through to the
    # escape hatch (tier 3) because identical content ⇒ identical verdicts.
    assert broken.tier_used and all(t == 3 for t in broken.tier_used), \
        broken.tier_used
    assert broken.escape_hatched
    # The defects persist instead of decaying with the revision counter.
    assert broken.clip.physics_verdicts or broken.clip.checklist.failed_items


# ─────────────────────────────────────────────────────────────────────────────
# 4) PlanValidator CCV convergence
# ─────────────────────────────────────────────────────────────────────────────
def test_plan_validator_ccv_loop_converges(tmp_path: Path):
    """Build a plan whose identity_refs include a known anchor and a bogus
    one. The plan_shots() Validate→Correct→Verify loop must drop the bogus
    one (Director.revise) and the second validator pass must succeed.
    """
    mem = AssetMemory(
        identity_anchors={"id_real": Identity("id_real", source="/data/r.png")},
    )

    # Build a screenwriter / director that produces a spec with both refs.
    sw = ScreenwriterAgent(config={"n_shots": 1, "max_shots": 1})

    class _GhostDirector(DirectorAgent):
        """Director that initially attaches both real+bogus identity refs so
        the first PlanValidator pass flags 'id_ghost' as ungroundable.
        """
        def run(self, outline, asset_memory, lesson_library=None):
            specs = super().run(outline, asset_memory, lesson_library)
            for s in specs:
                s.identity_refs = ["id_real", "id_ghost"]
            return specs

    director = _GhostDirector(config={"n_shots": 1, "max_shots": 1})
    validator = PlanValidatorAgent()

    specs = plan_shots(
        user_prompt="hero runs",
        asset_memory=mem,
        screenwriter=sw,
        director=director,
        physics_planner=PhysicsPlannerAgent(),
        cache_dir=tmp_path,
        plan_validator=validator,
        max_plan_iters=3,
    )
    # After CCV convergence, the bogus ref must be gone.
    for s in specs:
        assert "id_ghost" not in s.identity_refs
        assert "id_real" in s.identity_refs
    # And a fresh validator pass on the corrected specs must succeed.
    passed, feedback = validator.run(specs, mem)
    assert passed, feedback
