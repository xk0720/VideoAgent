"""Image-edit backends — mock default + REAL WaveSpeed seedream-v4 route."""
from pathlib import Path

import pytest

from maestro.models.image_edit import (
    MockImageEditClient,
    WaveSpeedImageEditClient,
    build_image_edit,
)


def test_factory_mock_default_and_real_dispatch():
    assert isinstance(build_image_edit(None), MockImageEditClient)
    assert isinstance(build_image_edit("mock-image-edit"), MockImageEditClient)
    assert isinstance(build_image_edit({"name": "wavespeed"}),
                      WaveSpeedImageEditClient)
    assert isinstance(build_image_edit("seedream"), WaveSpeedImageEditClient)
    with pytest.raises(ValueError):
        build_image_edit("definitely-not-a-backend")


def test_wavespeed_edit_loud_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    client = build_image_edit({"name": "wavespeed"})
    kf = tmp_path / "kf.png"; kf.write_bytes(b"\x89PNG\r\n")
    with pytest.raises(RuntimeError, match="API key"):
        client.edit(kf, "add a red cup", tmp_path / "out.png")


def test_wavespeed_edit_rejects_non_image_keyframe(tmp_path: Path, monkeypatch):
    """mock 的 .txt 关键帧桩绝不能流进真实编辑端点(诚实分界)。"""
    monkeypatch.setenv("WAVESPEED_API_KEY", "dummy-key")
    client = build_image_edit({"name": "wavespeed"})
    stub = tmp_path / "kf0.txt"; stub.write_text("keyframe stub")
    with pytest.raises(ValueError, match="image"):
        client.edit(stub, "x", tmp_path / "out.png")
    with pytest.raises(FileNotFoundError):
        client.edit(tmp_path / "missing.png", "x", tmp_path / "out.png")
