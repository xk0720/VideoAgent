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
  3. **C1 control-signal plumbing** — sketch → physics_sketch.control_signal
     → GeneratorAgent → video metadata. The C6 PhysicsConsistencyCritic
     reads the metadata line `control_signal=<path>` to decide divergence,
     so a missing record is a silent C6 break.
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
from maestro.physics.sketch import build_physics_sketch
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
# 3) C1 control-signal plumbing — sketch → generator metadata
# ─────────────────────────────────────────────────────────────────────────────
def test_control_signal_flows_from_sketch_to_generator_metadata(tmp_path: Path):
    """Mock generator writes `control_signal=<path>` into the output file.
    PhysicsConsistencyCritic parses that to score p2_sketch_consistency, so
    this plumbing is C6's lifeline. If it ever breaks, C6 silently always
    sees `control_signal=None` and emits CONSERVATION verdicts spuriously.
    """
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    spec.physics_sketch = build_physics_sketch(spec, tmp_path, fps=8)
    assert spec.physics_sketch.control_signal is not None
    assert spec.physics_sketch.control_signal.exists()

    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    body = clip.video_path.read_text(encoding="utf-8", errors="ignore")
    sketch_path = str(spec.physics_sketch.control_signal)
    assert f"control_signal={sketch_path}" in body, body[:200]


def test_no_control_signal_when_sketch_absent(tmp_path: Path):
    """A spec with no physics_sketch must produce a clip whose metadata
    explicitly records control_signal=None — that's the C6 critic's signal
    to STAY silent (nothing to be consistent with)."""
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a quiet still life")
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    body = clip.video_path.read_text(encoding="utf-8", errors="ignore")
    assert "control_signal=None" in body


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
