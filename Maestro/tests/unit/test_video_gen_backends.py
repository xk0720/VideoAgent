from pathlib import Path

import pytest

from maestro.models.video_gen import MockVideoGenClient, build_video_gen
from maestro.physics.control_render import load_control_spec
from maestro.physics.sketch import build_physics_sketch
from maestro.types import ShotSpec


def test_factory_returns_mock_by_default():
    assert isinstance(build_video_gen("mock-video-gen"), MockVideoGenClient)
    assert isinstance(build_video_gen(None), MockVideoGenClient)


def test_factory_dispatches_real_backend():
    client = build_video_gen({"name": "omniweaving"})
    assert client.__class__.__name__ == "OmniWeavingClient"
    # conditioning contract preserved (C1 needs control_signal support)
    assert "control_signal" in client.supported_conditions()


def test_real_backend_guards_when_unwired(tmp_path: Path):
    client = build_video_gen({"name": "omniweaving"})
    with pytest.raises((RuntimeError, NotImplementedError)):
        client.generate("a ball falls", 1.0, tmp_path / "o.mp4")


def test_control_spec_from_physics_sketch(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown and hits a wall")
    sketch = build_physics_sketch(spec, tmp_path, fps=8)
    cspec = load_control_spec(sketch.control_signal)
    assert cspec is not None
    assert cspec.n_frames == 8
    # screen-space tracks are normalized to [0,1]
    for track in cspec.tracks_2d.values():
        for x, y in track:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    # collision interaction surfaced as a hint the prompt can stress
    assert any("collision" in h for h in cspec.interaction_hints)


def test_load_control_spec_none():
    assert load_control_spec(None) is None
