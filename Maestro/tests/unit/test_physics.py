from pathlib import Path

from maestro.physics.failure_modes import detect_expected_modes, suggest_intervention
from maestro.physics.sketch import build_physics_sketch
from maestro.physics.sim_wrapper import MockSimulator
from maestro.types import PhysFailureMode, ShotSpec


def test_detect_expected_modes_keywords():
    modes = detect_expected_modes("a ball is thrown and falls to the ground")
    assert PhysFailureMode.GRAVITY_INERTIA in modes
    assert PhysFailureMode.PENETRATION in modes  # 'ground'


def test_detect_modes_chinese():
    modes = detect_expected_modes("水从杯子里倒出来")
    assert PhysFailureMode.FLUID in modes


def test_every_mode_has_intervention():
    for mode in PhysFailureMode:
        assert suggest_intervention(mode)


def test_sketch_builds_control_signal(tmp_path: Path):
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball is thrown")
    sketch = build_physics_sketch(spec, tmp_path, fps=8)
    assert sketch.entities
    assert sketch.control_signal and sketch.control_signal.exists()
    assert PhysFailureMode.GRAVITY_INERTIA in sketch.expected_modes


def test_simulator_gravity_pulls_down(tmp_path: Path):
    from maestro.types import PhysEntity

    sim = MockSimulator()
    out = sim.simulate(
        [PhysEntity(name="ball", init_velocity=(0, 0, 0), forces=["gravity"])],
        [], duration=1.0, fps=8, out_path=tmp_path / "s.json",
    )
    import json

    data = json.loads(out.read_text())
    ys = [p[1] for p in data["tracks"]["ball"]]
    assert ys[-1] < ys[0]  # fell under gravity
