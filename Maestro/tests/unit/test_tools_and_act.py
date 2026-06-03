"""Tests for v0.2.2 UniVA-borrowed additions:

  • ToolRegistry self-description + 4-category taxonomy
  • Representative tools (probe / extract / concat / image / caption / detect /
    audio) — all CPU-only, all gracefully degrade when ffmpeg / PIL absent.
  • ActAgent — generic tool-call executor (Plan-Act dual-agent's Act side),
    including a synthetic Plan→Act handoff exercised end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro.agents.act import ActAgent, ToolCall, ToolResult
from maestro.tools.base import ToolCategory, ToolRegistry, default_registry
from maestro.tools.captioning import CaptioningTool
from maestro.tools.detection import DetectionTool
from maestro.tools.audio_gen import AudioGenTool
from maestro.tools.frame_extract import FrameExtractTool
from maestro.tools.image_ops import ImageOpsTool
from maestro.tools.video_concat import VideoConcatTool
from maestro.tools.video_probe import VideoProbeTool


# ─────────────────────────────────────────────────────────────────────────────
# ToolRegistry
# ─────────────────────────────────────────────────────────────────────────────
def test_default_registry_populates_all_categories():
    reg = default_registry()
    # Maestro's category surface (UniVA's 4 + our physics/metric/retrieval).
    cats = {s.category for s in reg.list_specs()}
    for required in ("analysis", "generation", "editing", "tracking",
                     "metric", "retrieval"):
        # Note: "retrieval" RetrievalTool needs AssetMemory at construction time
        # so it's not in the default global registry. We assert the other 5
        # categories are populated by default tools.
        if required == "retrieval":
            continue
        assert required in cats, f"missing tools for category {required}: {cats}"


def test_tool_specs_are_machine_readable():
    """Every registered tool must self-describe with name/category/description."""
    reg = default_registry()
    for spec in reg.list_specs():
        assert spec.name
        assert spec.category in {"analysis", "generation", "editing", "tracking",
                                 "physics", "metric", "retrieval"}
        assert isinstance(spec.description, str)


def test_registry_get_raises_for_unknown_tool():
    reg = ToolRegistry()
    reg.register(CaptioningTool())
    with pytest.raises(KeyError):
        reg.get("nope")
    assert reg.get("caption").name == "caption"


def test_registry_last_write_wins_for_swap():
    """A real-backend tool should be hot-swappable for its mock with same name."""
    reg = ToolRegistry()
    a = CaptioningTool()
    reg.register(a)
    b = CaptioningTool()  # same name "caption"
    reg.register(b)
    assert reg.get("caption") is b


# ─────────────────────────────────────────────────────────────────────────────
# Individual tools (mock fallback paths — always CPU)
# ─────────────────────────────────────────────────────────────────────────────
def test_video_probe_falls_back_on_missing_ffprobe(tmp_path: Path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"\x00" * 8192)
    info = VideoProbeTool().run(v)
    assert info["exists"]
    assert info["duration"] >= 0.0
    # source is one of ffprobe/heuristic depending on the box.
    assert info["source"] in {"ffprobe", "heuristic"}


def test_video_probe_handles_missing_file(tmp_path: Path):
    info = VideoProbeTool().run(tmp_path / "nope.mp4")
    assert info["exists"] is False
    assert info["duration"] == 0.0


def test_frame_extract_writes_one_artifact_per_timestamp(tmp_path: Path):
    src = tmp_path / "src.mp4"
    src.write_text("mock", encoding="utf-8")
    out = FrameExtractTool().run(src, [0.0, 0.5, 1.0], tmp_path / "frames")
    assert len(out) == 3
    for p in out:
        assert p.exists()


def test_video_concat_mock_writes_manifest(tmp_path: Path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_text("mock a"); b.write_text("mock b")
    out = VideoConcatTool().run([a, b], tmp_path / "out.mp4")
    assert out.exists()
    text = out.read_text(encoding="utf-8", errors="ignore")
    # The mock fallback always writes a manifest (real ffmpeg path produces an mp4).
    assert "MOCK CONCAT" in text or out.stat().st_size > 0


def test_image_ops_resize_runs_on_any_box(tmp_path: Path):
    src = tmp_path / "x.png"
    src.write_text("mock image bytes")
    out = ImageOpsTool().resize(src, tmp_path / "y.png", (64, 64))
    assert out.exists()


def test_captioning_is_deterministic():
    cap = CaptioningTool()
    a = cap.run("/data/hero-shot.png")
    b = cap.run("/data/hero-shot.png")
    assert a == b
    assert "hero shot" in a


def test_detection_emits_normalized_bboxes(tmp_path: Path):
    det = DetectionTool().run(tmp_path / "img.png", query="person ball car")
    assert len(det) == 3
    for d in det:
        x0, y0, x1, y1 = d["bbox"]
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0


def test_audio_gen_writes_placeholder(tmp_path: Path):
    out = AudioGenTool().run("a calm piano motif", tmp_path / "m.mp3", duration=4.0)
    assert out.exists()
    assert "MOCK AUDIO" in out.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# ActAgent — Plan→Act handoff
# ─────────────────────────────────────────────────────────────────────────────
def test_act_agent_executes_a_plan_via_registry(tmp_path: Path):
    """A Plan emits ToolCalls; ActAgent runs them in order against the registry."""
    act = ActAgent(registry=default_registry())
    plan = [
        ToolCall(name="caption", args=["/data/hero-shot.png"], note="describe hero"),
        ToolCall(name="detect_objects",
                 kwargs={"media": tmp_path / "scene.png", "query": "person ball"}),
        ToolCall(name="audio_gen",
                 kwargs={"prompt": "tense strings",
                         "out_path": tmp_path / "score.mp3", "duration": 2.0}),
    ]
    results = act.run(plan)
    assert all(r.ok for r in results), [r.error for r in results]
    assert isinstance(results[0].value, str) and "hero" in results[0].value
    assert isinstance(results[1].value, list) and len(results[1].value) == 2
    assert (tmp_path / "score.mp3").exists()


def test_act_agent_returns_error_result_for_missing_tool():
    act = ActAgent()
    [res] = act.run([ToolCall(name="does_not_exist")])
    assert res.ok is False
    assert "unknown tool" in res.error


def test_act_agent_keeps_going_after_a_failed_call(tmp_path: Path):
    """A failure on step k must not halt step k+1 (UniVA executor semantics)."""
    act = ActAgent()
    plan = [
        ToolCall(name="does_not_exist"),
        ToolCall(name="caption", args=["/data/ok.png"]),
    ]
    results = act.run(plan)
    assert [r.ok for r in results] == [False, True]


def test_sandbox_refuses_side_effecting_tools(tmp_path: Path, monkeypatch):
    """MAESTRO_SANDBOX=1 must reject tools whose spec.side_effects=True
    (documented in `.env.example`). Read-only tools still go through.
    """
    import os
    monkeypatch.setenv("MAESTRO_SANDBOX", "1")
    act = ActAgent()
    # Side-effecting: audio_gen writes a file → refused.
    res = act.call(ToolCall(name="audio_gen",
                            kwargs={"prompt": "x", "out_path": tmp_path / "x.mp3"}))
    assert not res.ok
    assert "sandbox" in res.error.lower()
    # Read-only: caption has no side_effects → allowed.
    res = act.call(ToolCall(name="caption", args=["/data/x.png"]))
    assert res.ok


def test_sandbox_off_by_default(tmp_path: Path, monkeypatch):
    """Default behavior (no env var set) must allow side-effecting calls so
    production pipelines aren't crippled."""
    monkeypatch.delenv("MAESTRO_SANDBOX", raising=False)
    act = ActAgent()
    res = act.call(ToolCall(name="audio_gen",
                            kwargs={"prompt": "x", "out_path": tmp_path / "x.mp3"}))
    assert res.ok
