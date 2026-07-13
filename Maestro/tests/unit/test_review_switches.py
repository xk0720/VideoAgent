"""Reviewer switches — the physics branches are OPTIONAL (config-controlled)."""
from maestro.config import Config
from maestro.pipeline.run import build_components


def _critic_names(cfg):
    comp = build_components(Config(data=cfg))
    return [c.__class__.__name__ for c in comp.board.critics]


def test_default_keeps_both_physics_critics(tmp_path):
    names = _critic_names({"memory": {"path": str(tmp_path / "m")}})
    assert "PhysicsCritic" in names
    assert "PhysicsConsistencyCritic" in names
    assert "SemanticCritic" in names


def test_physics_measure_switch_off(tmp_path):
    names = _critic_names({"memory": {"path": str(tmp_path / "m")},
                           "review": {"physics_measure": False}})
    assert "PhysicsConsistencyCritic" not in names
    assert "PhysicsCritic" in names          # VLM 观点仍在


def test_pure_semantic_review(tmp_path):
    """两个物理分支都关 → 纯语义/一致性/节奏评审(用户:可能不用纯 physics)。"""
    names = _critic_names({"memory": {"path": str(tmp_path / "m")},
                           "review": {"physics_measure": False,
                                      "physics_vlm": False}})
    assert "PhysicsCritic" not in names
    assert "PhysicsConsistencyCritic" not in names
    assert names[0] == "SemanticCritic"
