from maestro.agents.director import DirectorAgent
from maestro.agents.plan_validator import PlanValidatorAgent
from maestro.planning.event_graph import (
    build_event_graph,
    fill_relations,
    validate_event_graph,
)
from maestro.types import (
    AssetMemory,
    EventEdge,
    EventGraph,
    EventNode,
    Identity,
    ShotSpec,
)


def test_event_graph_build_and_valid():
    spec = ShotSpec(shot_idx=0, duration=2.0, prompt="a ball is thrown and hits a wall")
    g = build_event_graph(spec)
    ok, issues = validate_event_graph(g)
    assert ok, issues
    assert len(g.nodes) >= 1
    # 'hit' is an impact action -> a 'causes' edge should appear
    assert any(e.relation == "causes" for e in g.edges)


def test_event_graph_validation_catches_bad_temporal():
    g = EventGraph(
        nodes=[EventNode("e0", "run", actors=["x"], start=0, end=1),
               EventNode("e1", "jump", actors=["x"], start=1, end=2)],
        edges=[EventEdge(src="e1", dst="e0", relation="before")],  # reversed!
    )
    ok, issues = validate_event_graph(g)
    assert not ok
    assert any("temporal order" in m for m in issues)


def test_validation_catches_missing_actor_and_dangling_edge():
    g = EventGraph(
        nodes=[EventNode("e0", "appear", actors=[])],
        edges=[EventEdge(src="e0", dst="eX")],
    )
    ok, issues = validate_event_graph(g)
    assert not ok
    assert any("no actors" in m for m in issues)
    assert any("dangling" in m for m in issues)


def test_plan_validator_flags_missing_identity_ref():
    spec = ShotSpec(shot_idx=0, duration=2.0, prompt="hero runs",
                    identity_refs=["id_ghost"])
    spec.event_graph = build_event_graph(spec)
    mem = AssetMemory()  # empty -> id_ghost ungroundable
    passed, feedback = PlanValidatorAgent().run([spec], mem)
    assert not passed
    assert 0 in feedback


def test_director_revise_drops_ungroundable_ref():
    spec = ShotSpec(shot_idx=0, duration=2.0, prompt="hero runs",
                    identity_refs=["id_ghost", "id_real"])
    mem = AssetMemory(identity_anchors={"id_real": Identity("id_real")})
    DirectorAgent().revise(spec, mem, ["bad ref"])
    assert spec.identity_refs == ["id_real"]
    ok, _ = validate_event_graph(spec.event_graph)
    assert ok
