"""VLM-backend factory + the honesty branch (v0.4).

CPU-only, NO network. The load-bearing test: OpenAICompatVLM.assess_* on a
non-decodable mock clip returns [] (no judgment from no pixels) WITHOUT needing
an api_key — the decode-None check runs BEFORE the key check. We never hit a
real endpoint.
"""
from __future__ import annotations

import pytest

from maestro.agents.generator import GeneratorAgent
from maestro.models.mllm import MockMLLMClient, build_mllm
from maestro.physics.annotate import annotate_physics
from maestro.types import CandidateClip, ShotSpec


def _spec(prompt="a ball falls") -> ShotSpec:
    spec = ShotSpec(shot_idx=0, duration=2.0, prompt=prompt)
    spec.physics_annotation = annotate_physics(spec)
    return spec


# ── factory dispatch ──
def test_factory_returns_mock_by_default():
    assert isinstance(build_mllm(None), MockMLLMClient)
    assert isinstance(build_mllm("mock-mllm"), MockMLLMClient)
    assert isinstance(build_mllm({"name": "mock-mllm"}), MockMLLMClient)


def test_factory_dispatches_real_backends_without_io():
    for name in ["gpt-4o", "openai-vlm", "openai", "qwen-vl", "qwen"]:
        client = build_mllm({"name": name})
        assert client.__class__.__name__ == "OpenAICompatVLM"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_mllm({"name": "definitely-not-a-vlm"})


def test_base_url_and_model_defaults(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    gpt = build_mllm({"name": "gpt-4o"})
    assert gpt.base_url == "https://api.openai.com/v1"
    assert gpt.model == "gpt-4o"
    qwen = build_mllm({"name": "qwen-vl"})
    assert qwen.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert qwen.model == "qwen-vl-max"


# ── HONESTY BRANCH: no pixels → no verdict, no key required ──
def test_assess_semantic_empty_on_mock_text_clip(tmp_path, monkeypatch):
    """A mock pipeline writes a TEXT placeholder with a .mp4 name. The VLM has
    no pixels to judge → assess_semantic returns [] (NOT a fake passed=True),
    and must do so WITHOUT an api_key (decode-None before key check)."""
    for var in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    vlm = build_mllm({"name": "gpt-4o"})           # no key set
    assert vlm.assess_semantic(clip, spec) == []   # no judgment, no crash


def test_assess_physics_empty_on_mock_text_clip(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    spec = _spec()
    clip = GeneratorAgent().run(spec, tmp_path, revision=0, seed=0, fps=8)
    vlm = build_mllm({"name": "gpt-4o"})
    assert vlm.assess_physics(clip, spec, fps=8) == []


def test_assess_empty_on_missing_file(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    spec = _spec()
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "nope.mp4")
    vlm = build_mllm({"name": "qwen-vl"})
    assert vlm.assess_semantic(clip, spec) == []
    assert vlm.assess_physics(clip, spec, fps=8) == []


def test_compare_falls_back_to_metric_on_non_video(tmp_path, monkeypatch):
    """No decodable pixels on a side → compare uses the base metric ranking."""
    for var in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    spec = _spec()
    a = CandidateClip(shot_idx=0, video_path=tmp_path / "a.mp4")
    b = CandidateClip(shot_idx=0, video_path=tmp_path / "b.mp4")
    a.metric_scores["weighted_total"] = 0.8
    b.metric_scores["weighted_total"] = 0.4
    vlm = build_mllm({"name": "gpt-4o"})
    assert vlm.compare(a, b, spec) == 1   # a wins on metric, no key/network needed
    assert vlm.compare(b, a, spec) == -1


# ── loud on missing key ONLY when there is real evidence (decodable frames) ──
def test_loud_without_key_when_frames_present(tmp_path, monkeypatch):
    """With real frames (evidence) but no key, the VLM must raise loudly — it
    cannot silently skip a judgment it was configured to make."""
    pytest.importorskip("numpy")
    import numpy as np

    import maestro.models.mllm_backends as be

    for var in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        be, "_decode_frames", lambda p: np.zeros((4, 8, 8, 3), dtype="uint8")
    )
    spec = _spec()
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "real.mp4")
    vlm = build_mllm({"name": "gpt-4o"})
    with pytest.raises(RuntimeError, match="API key"):
        vlm.assess_semantic(clip, spec)


def test_inference_failure_degrades_safely(tmp_path, monkeypatch):
    """Frames + key present, but the chat call yields nothing (HTTP failure /
    unparseable reply) → [] (non-fatal), mirroring CoTrackerExtractor: no crash,
    no fabricated verdict. _chat itself logs the WARN on real transport errors."""
    pytest.importorskip("numpy")
    import numpy as np

    import maestro.models.mllm_backends as be

    monkeypatch.setattr(
        be, "_decode_frames", lambda p: np.zeros((4, 8, 8, 3), dtype="uint8")
    )
    # simulate a degraded call: _chat returns None (what it does on HTTP error)
    monkeypatch.setattr(be.OpenAICompatVLM, "_chat",
                        lambda self, frames, text: None)
    spec = _spec()
    clip = CandidateClip(shot_idx=0, video_path=tmp_path / "real.mp4")
    vlm = be.OpenAICompatVLM(name="gpt-4o", config={"api_key": "k"})
    assert vlm.assess_semantic(clip, spec) == []      # non-fatal
    assert vlm.assess_physics(clip, spec, fps=8) == []
