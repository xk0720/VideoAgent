"""Music structure analysis.

Open-source library wrapped here:
    • **All-In-One**  (pip: ``allin1``)
      https://github.com/mir-aidj/all-in-one
      One-shot tempo + beat + downbeat + functional-section analysis.
    Fallback: ``librosa`` (tempo + beat tracking only).

v0.1 mock returns a deterministic, structurally plausible MusicProfile
(intro / verse / chorus / outro) for any audio file, so Stage 2 can run
without a real audio dep.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import PreprocessCfg
from ..logging import logger
from ..types import MusicProfile, MusicSection


class MusicAnalyzer:
    def __init__(self, cfg: PreprocessCfg, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = mock

    def analyze(self, audio_path: Optional[Path | str]) -> MusicProfile:
        if audio_path is None or self.mock:
            return self._mock(audio_path)
        try:
            import allin1  # type: ignore
        except ImportError:
            logger.warning("allin1 not installed — using mock music profile.")
            return self._mock(audio_path)
        # v0.2: real allin1 path.
        result = allin1.analyze(str(audio_path))                # pragma: no cover
        sections = [                                            # pragma: no cover
            MusicSection(name=s.label, start_time=s.start, end_time=s.end,
                         energy_db=getattr(s, "energy_db", -20.0),
                         num_beats=int(getattr(s, "num_beats", 0)))
            for s in result.segments
        ]
        return MusicProfile(                                    # pragma: no cover
            audio_path=Path(audio_path), duration=result.duration, bpm=result.bpm,
            beats=list(result.beats), downbeats=list(result.downbeats),
            sections=sections,
        )

    def _mock(self, audio_path: Optional[Path | str]) -> MusicProfile:
        bpm = 120.0
        duration = 60.0
        beat_interval = 60.0 / bpm
        beats = [i * beat_interval for i in range(int(duration / beat_interval))]
        downbeats = beats[::4]
        sections = [
            MusicSection(name="intro",   start_time=0.0,  end_time=8.0,  energy_db=-25.0, num_beats=16),
            MusicSection(name="verse",   start_time=8.0,  end_time=24.0, energy_db=-18.0, num_beats=32),
            MusicSection(name="chorus",  start_time=24.0, end_time=44.0, energy_db=-12.0, num_beats=40),
            MusicSection(name="outro",   start_time=44.0, end_time=60.0, energy_db=-22.0, num_beats=32),
        ]
        return MusicProfile(
            audio_path=Path(audio_path) if audio_path else None,
            duration=duration, bpm=bpm, beats=beats, downbeats=downbeats,
            sections=sections,
        )


__all__ = ["MusicAnalyzer"]
