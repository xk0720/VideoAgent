"""Config loader smoke tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from longvideoagent.config import (
    Config,
    load_config,
    load_prompt,
    load_yaml,
    configs_dir,
    prompts_dir,
)


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.project_name == "longvideoagent"
    assert cfg.compose.retrieval.beam_width == 3
    assert cfg.mocks.llm is True
    assert isinstance(cfg.cache_root, Path)


def test_overrides_merge():
    cfg = load_config(overrides={"plan": {"max_iterations": 9}})
    assert cfg.plan.max_iterations == 9


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        load_config(overrides={"this_key_does_not_exist": 42})


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LVA_CACHE_ROOT", str(tmp_path / "cache"))
    cfg = load_config()
    assert cfg.cache_root == tmp_path / "cache"


def test_load_prompt_bare_name():
    txt = load_prompt("screenwriter")
    assert "ScreenwriterAgent" in txt


def test_configs_and_prompts_dirs_exist():
    assert configs_dir().is_dir()
    assert prompts_dir().is_dir()


def test_load_yaml_heuristics():
    presets = load_yaml(configs_dir() / "heuristics" / "presets.yaml")["presets"]
    assert "default" in presets
    assert sum(presets["default"]["weights"].values()) == pytest.approx(1.0, rel=1e-2)
