"""Detection backend factory + GroundingDINO graceful degradation (C6 v0.4).

These close the physics-from-pixels break: detection grounds each entity so
CoTracker seeds on the ACTUAL named entity instead of a fixed pixel. CPU-only,
NO network, NO torch — the real backend's loud-on-missing-deps guard is probed
without ever importing torch/transformers (this env has neither)."""
from pathlib import Path

import pytest

from maestro.models.detection_backends import (
    BaseDetector,
    GroundingDINODetector,
    MockDetector,
    build_detector,
)
from maestro.tools.detection import DetectionTool


# ── factory dispatch ──
def test_factory_returns_mock_by_default():
    assert isinstance(build_detector(None), MockDetector)
    assert isinstance(build_detector("mock"), MockDetector)
    assert isinstance(build_detector({"name": "mock-detect"}), MockDetector)


def test_factory_dispatches_groundingdino_without_loading_torch():
    """Constructing the real backend must NOT import torch/transformers (lazy);
    it only loads on the first detect() call."""
    for name in ("groundingdino", "dino", "grounding-dino"):
        det = build_detector({"name": name})
        assert isinstance(det, GroundingDINODetector)
        assert isinstance(det, BaseDetector)
    assert isinstance(build_detector("groundingdino"), GroundingDINODetector)


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_detector({"name": "definitely-not-a-detector"})


# ── GroundingDINO loud-on-missing-deps (NEVER imports real torch / network) ──
def test_groundingdino_detect_loud_without_deps(monkeypatch):
    """The real backend must fail LOUDLY (RuntimeError) when its deps are
    unavailable — and the TEST must never import real torch/transformers (slow,
    and would hit the network on a real model id). We make `import torch` raise
    by poisoning sys.modules, forcing _ensure_loaded's import guard
    deterministically and fast, regardless of what's installed."""
    import sys

    np = pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` → ImportError
    det = build_detector({"name": "groundingdino", "device": "cpu"})
    frame = np.zeros((8, 8, 3), dtype="uint8")
    with pytest.raises(RuntimeError, match="torch"):
        det.detect(frame, "ball")


# ── MockDetector contract + centroid helper ──
def test_mock_detector_returns_normalized_bboxes():
    np = pytest.importorskip("numpy")
    det = MockDetector()
    out = det.detect(np.zeros((4, 4, 3), "uint8"), "person ball car", max_results=3)
    assert len(out) == 3
    assert [d["label"] for d in out] == ["person", "ball", "car"]
    for d in out:
        x0, y0, x1, y1 = d["bbox"]
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0


def test_centroid_helper():
    assert BaseDetector.centroid([0.1, 0.3, 0.3, 0.7]) == (0.2, 0.5)


# ── DetectionTool mock path unchanged (back-compat) ──
def test_detection_tool_default_is_mock():
    assert isinstance(DetectionTool().client, MockDetector)


def test_detection_tool_mock_path_byte_shape(tmp_path: Path):
    """Same return structure as v0.2.2 (label/bbox/score/source), incl. source."""
    out = DetectionTool().run(tmp_path / "img.png", query="person ball car")
    assert len(out) == 3
    for d in out:
        assert set(d) == {"label", "bbox", "score", "source"}
        x0, y0, x1, y1 = d["bbox"]
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0
        assert d["source"] == str(tmp_path / "img.png")
