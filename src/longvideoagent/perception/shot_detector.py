"""Shot segmentation.

Open-source library wrapped here:
    • **PySceneDetect**  (pip: ``scenedetect``)
      https://github.com/Breakthrough/PySceneDetect

v0.1 mode:
    If ``scenedetect`` is not importable or ``mock=True``, returns an
    evenly-spaced split of the video into ``min(N, video_duration/2s)`` shots
    so downstream stages can still run.
"""
from __future__ import annotations

from pathlib import Path

from ..config import PreprocessCfg
from ..logging import logger
from ..utils.video_io import probe_duration


class ShotDetector:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = mock

    def detect(self, video_path: Path | str) -> list[tuple[float, float]]:
        """Return a list of (start_s, end_s) shot intervals."""
        if self.mock:
            return self._mock_split(video_path)
        try:
            from scenedetect import open_video, SceneManager  # type: ignore
            from scenedetect.detectors import ContentDetector  # type: ignore
        except ImportError:
            logger.warning("scenedetect not installed — falling back to mock shot detector.")
            return self._mock_split(video_path)

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=self.cfg.shot_detector.threshold,
                                             min_scene_len=self.cfg.shot_detector.min_scene_len))
        manager.detect_scenes(video=video, show_progress=False)
        scenes = manager.get_scene_list()
        return [(float(a.get_seconds()), float(b.get_seconds())) for a, b in scenes]

    def _mock_split(self, video_path: Path | str) -> list[tuple[float, float]]:
        try:
            duration = probe_duration(video_path)
        except Exception:                                       # pragma: no cover
            duration = 5.0
        shot_len = 2.0
        n = max(1, int(duration // shot_len))
        return [(i * shot_len, min((i + 1) * shot_len, duration)) for i in range(n)]


__all__ = ["ShotDetector"]
