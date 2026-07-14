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


def test_gemini_vlm_dispatch_and_loud_without_key(tmp_path, monkeypatch):
    """Gemini VLM:注册表分发;无 key 时真评审(有帧)必须大声报错;
    mock 桩视频(无帧)仍走诚实沉默分支(不需要 key)。"""
    import numpy as np
    import pytest

    from maestro.models.mllm import build_mllm
    from maestro.models.mllm_backends import GeminiVLM
    from maestro.types import CandidateClip, ShotSpec

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    vlm = build_mllm({"name": "gemini"})
    assert isinstance(vlm, GeminiVLM)
    assert vlm.model == "gemini-3.5-flash"
    assert vlm.base_url.startswith("https://generativelanguage.googleapis.com")
    # env 覆盖模型名
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-pro")
    assert build_mllm({"name": "gemini"}).model == "gemini-3.5-pro"

    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "x.mp4")
    spec = ShotSpec(shot_idx=0, duration=1.0, prompt="a ball falls")
    # 文本桩(魔数嗅探不过)→ 诚实沉默,不需要 key,不碰网络
    (tmp_path / "x.mp4").write_text("MOCK VIDEO")
    assert vlm.assess_semantic(clip, spec) == []
    # 真视频魔数 + 无 key → loud(原生视频路径命中 key 检查)
    real = tmp_path / "real.mp4"
    real.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    clip2 = CandidateClip(shot_idx=0, video_path=real)
    with pytest.raises(RuntimeError, match="API key"):
        vlm.assess_semantic(clip2, spec)
    # caption:非图片/缺文件 → ""(诚实),不碰网络
    assert vlm.caption_image(tmp_path / "nope.png") == ""


def test_load_dotenv_semantics(tmp_path, monkeypatch):
    """.env 加载:注释/空行/export 前缀/引号;已导出的环境变量优先;
    空值占位行不写入;缺文件返回 0 不报错。"""
    from maestro.config import load_dotenv

    envf = tmp_path / ".env"
    envf.write_text(
        "# comment\n"
        "OPENAI_API_KEY=sk-from-file\n"
        "export GEMINI_API_KEY='g-key'\n"
        'GEMINI_MODEL="gemini-3.5-flash"\n'
        "EMPTY_PLACEHOLDER=\n"
        "not a kv line\n",
        encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("EMPTY_PLACEHOLDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-export")  # 真环境优先

    import os
    n = load_dotenv(envf)
    assert n == 2                                    # gemini key + model
    assert os.environ["OPENAI_API_KEY"] == "sk-real-export"
    assert os.environ["GEMINI_API_KEY"] == "g-key"   # 引号剥掉
    assert os.environ["GEMINI_MODEL"] == "gemini-3.5-flash"
    assert "EMPTY_PLACEHOLDER" not in os.environ     # 空占位不写
    assert load_dotenv(tmp_path / "missing.env") == 0
