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
    """An object dropped from a height falls under gravity (toward the floor)."""
    import json

    from maestro.types import PhysEntity

    sim = MockSimulator()
    out = sim.simulate(
        [PhysEntity(name="ball", init_velocity=(0, 0, 0),
                    init_position=(0, 3.0, 0), forces=["gravity"])],
        [], duration=1.0, fps=8, out_path=tmp_path / "s.json",
    )
    data = json.loads(out.read_text())
    ys = [p[1] for p in data["tracks"]["ball"]]
    assert ys[-1] < ys[0]               # fell under gravity
    assert min(ys) >= 0.0               # never penetrates the ground plane


def test_simulator_ground_plane_no_fallthrough(tmp_path: Path):
    """A ball resting on the ground with no velocity stays put — the old
    integrator let it fall through to negative y (unphysical)."""
    import json

    from maestro.types import PhysEntity

    sim = MockSimulator()
    out = sim.simulate(
        [PhysEntity(name="ball", init_velocity=(0, 0, 0),
                    init_position=(0, 0, 0), forces=["gravity"])],
        [], duration=1.0, fps=8, out_path=tmp_path / "s.json",
    )
    ys = [p[1] for p in json.loads(out.read_text())["tracks"]["ball"]]
    assert all(y >= 0.0 for y in ys)
    assert max(ys) < 0.5                # essentially rests on the floor


def test_simulator_ball_bounces_off_ground(tmp_path: Path):
    """A ball dropped from a height hits the floor fast and clearly rebounds —
    y must go back up after the first contact, recorded as a ground_bounce."""
    import json

    from maestro.types import PhysEntity

    sim = MockSimulator()
    out = sim.simulate(
        [PhysEntity(name="ball", init_velocity=(0, 0, 0),
                    init_position=(0, 3.0, 0), forces=["gravity"],
                    restitution=0.85)],
        [], duration=3.0, fps=8, out_path=tmp_path / "s.json",
    )
    data = json.loads(out.read_text())
    ys = [p[1] for p in data["tracks"]["ball"]]
    assert min(ys) >= 0.0                       # never penetrates the floor
    bounces = [e for e in data["events"] if e["kind"] == "ground_bounce"]
    assert bounces, data["events"]              # at least one real bounce
    # after the first bounce frame the ball goes back UP
    bf = bounces[0]["frame"]
    assert max(ys[bf:]) > ys[bf] - 1e-9


def test_simulator_wall_bounce_records_event(tmp_path: Path):
    """A horizontally-moving collider hits a wall ahead and rebounds in x."""
    import json

    from maestro.types import PhysEntity, PhysInteraction

    sim = MockSimulator()
    out = sim.simulate(
        [PhysEntity(name="ball", init_velocity=(2.0, 0, 0),
                    init_position=(0, 1.0, 0), forces=[])],   # no gravity, pure x
        [PhysInteraction(kind="collision", entities=["ball"])],
        duration=2.0, fps=8, out_path=tmp_path / "s.json",
    )
    data = json.loads(out.read_text())
    xs = [p[0] for p in data["tracks"]["ball"]]
    assert max(xs) <= MockSimulator.WALL_X + 1e-6       # never passes the wall
    walls = [e for e in data["events"] if e["kind"] == "wall_bounce"]
    assert walls, data["events"]


def test_simulator_support_keeps_object_resting(tmp_path: Path):
    """A supported object does not fall — it rests on its support surface."""
    import json

    from maestro.types import PhysEntity, PhysInteraction

    sim = MockSimulator()
    out = sim.simulate(
        [PhysEntity(name="box", init_velocity=(0, 0, 0),
                    init_position=(0, 0, 0), forces=["gravity"])],
        [PhysInteraction(kind="support", entities=["box"])],
        duration=1.0, fps=8, out_path=tmp_path / "s.json",
    )
    ys = [p[1] for p in json.loads(out.read_text())["tracks"]["box"]]
    assert all(abs(y) < 1e-6 for y in ys)               # stays exactly at rest
