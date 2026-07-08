"""GenesisSimClient — physics sim as repair conditioning (NEWTON borrow).

CPU-only, NO genesis/torch/network: tests cover the loud guard, scene_spec
validation, the ground-truth describer, and the factory. The real Genesis run
needs a GPU and is exercised on the server via --with-sim."""
import sys

import pytest

from maestro.physics.sim_backends import (
    GenesisSimClient,
    build_sim_client,
    describe_scene_spec,
    validate_scene_spec,
)


def test_loud_without_genesis(tmp_path, monkeypatch):
    """No genesis package → RuntimeError with the install hint, never a mock."""
    monkeypatch.setitem(sys.modules, "genesis", None)
    client = GenesisSimClient(config={"output_dir": str(tmp_path)})
    spec = {"objects": [{"id": "ball", "type": "sphere", "pos": [0, 0, 1.0],
                         "radius": 0.1}]}
    with pytest.raises(RuntimeError, match="genesis-world"):
        client.run(spec)


def test_validate_scene_spec_rejects_bad_specs():
    with pytest.raises(ValueError, match="objects"):
        validate_scene_spec({})
    with pytest.raises(ValueError, match="objects"):
        validate_scene_spec({"objects": []})
    with pytest.raises(ValueError, match="unknown object type"):
        validate_scene_spec({"objects": [{"type": "teapot"}]})
    # v1 is RIGID-ONLY: particle materials degrade honestly, never fake-sim.
    with pytest.raises(ValueError, match="RIGID-ONLY"):
        validate_scene_spec({"objects": [{"type": "sphere", "material": "liquid"}]})


def test_validate_scene_spec_accepts_rigid_scene():
    spec = {"objects": [
        {"id": "ball", "type": "sphere", "pos": [0, 0, 1.2], "radius": 0.1,
         "init_velocity": [0.4, 0, 0]},
        {"id": "table", "type": "box", "pos": [0, 0, 0.4],
         "size": [1.2, 0.8, 0.05], "fixed": True},
    ]}
    assert len(validate_scene_spec(spec)) == 2


def test_describe_scene_spec_ground_truth_counts():
    """The describer reads counts from the SPEC (ground truth), skipping fixed
    supports and the floor — NEWTON's ref_video_desc pattern."""
    spec = {"objects": [
        {"id": "b1", "type": "sphere"},
        {"id": "b2", "type": "sphere"},
        {"id": "table", "type": "box", "fixed": True},
        {"id": "floor", "type": "plane"},
    ]}
    desc = describe_scene_spec(spec)
    assert "exactly 2 spheres" in desc
    assert "box" not in desc and "plane" not in desc


def test_build_sim_client_factory():
    assert build_sim_client(None) is None
    assert build_sim_client("") is None
    assert isinstance(build_sim_client("genesis"), GenesisSimClient)
    assert isinstance(build_sim_client({"name": "genesis", "backend": "cpu"}),
                      GenesisSimClient)
    with pytest.raises(ValueError):
        build_sim_client("mujoco")
