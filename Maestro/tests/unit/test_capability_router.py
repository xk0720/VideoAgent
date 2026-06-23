"""Phase-2 capability routing tests — routing is a SKILL, not config.

Covers:
  • skill-reuse path (matched_skill.gen_capability in available → source=skill)
  • cold-start heuristic for each branch (edit / flf2v / i2v / t2v)
  • NEVER returns a capability outside `available` (mock {t2v,i2v} downgrades,
    and the downgrade is reported)
  • GeneratorAgent dispatches flf2v/edit to the right WaveSpeed method (stub
    client recording the call; NO network) and falls back on a mock backend
  • a distilled skill's gen_capability round-trips through persistence and is
    reused on a second plan
"""
from __future__ import annotations

from pathlib import Path

from maestro.agents.capability_router import CapabilityRouter
from maestro.agents.generator import GeneratorAgent
from maestro.memory.skill_library import SkillLibrary
from maestro.models.video_gen import BaseVideoGenClient, MockVideoGenClient
from maestro.physics.annotate import annotate_physics
from maestro.types import (
    AssetMemory,
    CinematographyTags,
    Identity,
    Skill,
    ShotSpec,
)

WAVE_CAPS = {"t2v", "i2v", "flf2v", "edit"}
MOCK_CAPS = {"t2v", "i2v"}


def _spec(idx: int = 0, prompt: str = "a ball falls", **kw) -> ShotSpec:
    return ShotSpec(shot_idx=idx, duration=1.0, prompt=prompt, **kw)


# ── (a) skill reuse wins ────────────────────────────────────────────────
def test_router_skill_reuse_wins():
    skill = Skill(skill_id="Sedit01", name="edit_recipe",
                  gen_capability="edit",
                  gen_params={"source_video": "clip.mp4", "backend": "vace"})
    spec = _spec()
    spec.matched_skill = skill
    d = CapabilityRouter().route(spec, AssetMemory(), WAVE_CAPS)
    assert d.capability == "edit"
    assert d.source == "skill"
    assert d.params["source_video"] == "clip.mp4"
    assert d.params["backend"] == "vace"
    assert d.downgraded_from == ""


def test_router_skill_capability_not_offered_falls_through():
    """A skill recorded a capability the backend lacks → fall to heuristic,
    and the intent is preserved in the reason."""
    skill = Skill(skill_id="Sflf", name="flf", gen_capability="flf2v")
    spec = _spec()
    spec.matched_skill = skill
    d = CapabilityRouter().route(spec, AssetMemory(), MOCK_CAPS)
    assert d.capability in MOCK_CAPS
    assert d.source == "heuristic"
    assert "flf2v" in d.reason


# ── (b) cold-start heuristic per branch ─────────────────────────────────
def test_heuristic_edit_when_source_video_marked():
    spec = _spec(gen_params={"source_video": "src.mp4", "task": "pose"})
    d = CapabilityRouter().route(spec, AssetMemory(), WAVE_CAPS)
    assert d.capability == "edit"
    assert d.source == "heuristic"
    assert d.params["source_video"] == "src.mp4"
    assert d.params["task"] == "pose"


def test_heuristic_flf2v_when_both_keyframes():
    spec = _spec(gen_params={"first_frame": "a.png", "last_frame": "b.png"})
    d = CapabilityRouter().route(spec, AssetMemory(), WAVE_CAPS)
    assert d.capability == "flf2v"
    assert d.params == {"first_frame": "a.png", "last_frame": "b.png"}


def test_heuristic_i2v_when_identity_anchor():
    am = AssetMemory(identity_anchors={"hero": Identity(identity_id="hero",
                                                        source="hero.png")})
    spec = _spec(identity_refs=["hero"])
    d = CapabilityRouter().route(spec, am, WAVE_CAPS)
    assert d.capability == "i2v"


def test_heuristic_t2v_default():
    d = CapabilityRouter().route(_spec(), AssetMemory(), WAVE_CAPS)
    assert d.capability == "t2v"
    assert d.source == "heuristic"


# ── (c) NEVER returns a capability outside `available` ──────────────────
def test_router_never_returns_unavailable_edit_downgrades_to_t2v():
    spec = _spec(gen_params={"source_video": "src.mp4"})
    d = CapabilityRouter().route(spec, AssetMemory(), MOCK_CAPS)
    assert d.capability == "t2v"               # edit not offered → downgrade
    assert d.capability in MOCK_CAPS
    assert d.downgraded_from == "edit"          # the downgrade is reported
    assert "edit" in d.reason


def test_router_downgrades_edit_to_i2v_when_anchor_present():
    am = AssetMemory(identity_anchors={"hero": Identity(identity_id="hero",
                                                        source="hero.png")})
    spec = _spec(identity_refs=["hero"], gen_params={"source_video": "src.mp4"})
    d = CapabilityRouter().route(spec, am, MOCK_CAPS)
    assert d.capability == "i2v"                # anchor exists → i2v over t2v
    assert d.downgraded_from == "edit"


# ── GeneratorAgent dispatch (no network) ────────────────────────────────
class _StubWaveClient(BaseVideoGenClient):
    """Records which capability method was called; writes a tiny file."""

    def __init__(self):
        self.calls: list[str] = []

    def _write(self, out_path: Path, tag: str) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"STUB {tag}\n", encoding="utf-8")
        return out_path

    def generate(self, prompt, duration, out_path, fps=8, first_frame=None,
                 reference_images=None, seed=0):
        self.calls.append("generate")
        return self._write(out_path, "generate")

    def frame_to_frame(self, prompt, first_frame, last_frame, out_path,
                       duration=5, seed=0):
        self.calls.append("frame_to_frame")
        return self._write(out_path, "flf2v")

    def edit_video(self, prompt, video_path, out_path, backend="runway",
                   task="depth", seed=0):
        self.calls.append(f"edit_video:{backend}:{task}")
        return self._write(out_path, "edit")

    def supported_conditions(self):
        return {"first_frame"}

    def capabilities(self):
        return WAVE_CAPS


def test_generator_dispatches_flf2v(tmp_path: Path):
    stub = _StubWaveClient()
    gen = GeneratorAgent(video_gen=stub)
    spec = _spec()
    spec.gen_capability = "flf2v"
    spec.gen_params = {"first_frame": str(tmp_path / "a.png"),
                       "last_frame": str(tmp_path / "b.png")}
    clip = gen.run(spec, tmp_path)
    assert stub.calls == ["frame_to_frame"]
    assert "capability=flf2v" in clip.keyframes[0].read_text()


def test_generator_dispatches_edit(tmp_path: Path):
    stub = _StubWaveClient()
    gen = GeneratorAgent(video_gen=stub)
    spec = _spec()
    spec.gen_capability = "edit"
    spec.gen_params = {"source_video": str(tmp_path / "src.mp4"),
                       "backend": "vace", "task": "depth"}
    gen.run(spec, tmp_path)
    assert stub.calls == ["edit_video:vace:depth"]


def test_generator_falls_back_on_mock_backend(tmp_path: Path):
    """Mock backend lacks flf2v/edit → fall back to generate(), record the
    downgraded capability, no exception."""
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    spec = _spec()
    spec.gen_capability = "edit"
    spec.gen_params = {"source_video": "src.mp4"}
    clip = gen.run(spec, tmp_path)
    body = clip.keyframes[0].read_text()
    assert "capability=t2v" in body              # downgraded, generate() ran
    assert clip.video_path.exists()


def test_generator_default_t2v_unchanged(tmp_path: Path):
    """Default path (no capability set) still goes through generate()."""
    gen = GeneratorAgent(video_gen=MockVideoGenClient())
    clip = gen.run(_spec(), tmp_path)
    assert "capability=t2v" in clip.keyframes[0].read_text()


# ── end-to-end: capability round-trips through persistence + is reused ──
def test_distilled_capability_round_trips_and_is_reused(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0,
                    prompt="a ball is thrown and bounces off a wall")
    ann = annotate_physics(spec)
    lib = SkillLibrary(tmp_path / "skills.jsonl")
    skill = lib.distill(
        name="bounce_i2v", spec_prompt=spec.prompt, annotation=ann,
        cinematography=CinematographyTags(), thresholds={}, weighted_total=0.9,
        gen_capability="i2v", gen_params={"anchor": "hero.png"},
    )
    assert skill.gen_capability == "i2v"

    # Reload from disk — the recorded capability survives persistence.
    lib2 = SkillLibrary(tmp_path / "skills.jsonl")
    s2 = lib2.skills[0]
    assert s2.gen_capability == "i2v"
    assert s2.gen_params == {"anchor": "hero.png"}

    # A second similar shot retrieves the skill → router reuses the capability.
    spec2 = ShotSpec(shot_idx=1, duration=1.0, prompt=spec.prompt)
    hits = lib2.retrieve(spec2.prompt, list(ann.expected_modes), top_k=1)
    assert hits
    spec2.matched_skill = hits[0]
    d = CapabilityRouter().route(spec2, AssetMemory(), WAVE_CAPS)
    assert d.capability == "i2v"
    assert d.source == "skill"                    # the skill decides the model
