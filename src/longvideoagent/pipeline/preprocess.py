"""Stage 1 — offline understanding.

Pipeline:
    sources/music
        ├── ShotDetector  (PySceneDetect)
        ├── FeatureExtractor (CLIP)        ┐
        ├── FlowExtractor (RAFT)           │
        ├── SaliencyExtractor (U²-Net)     ├─ per-shot features
        ├── CharacterIdentifier (InsightFace)
        ├── CinematographyTagger (ShotVL)  │
        ├── DialogueMatcher (EasyOCR + WeSpeaker)
        ├── ShotCaptioner (Qwen-VL, rolling buffer)
        └── MusicAnalyzer (All-In-One)
    → MemoryStore on disk (SQLite + npz + optional FAISS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Config
from ..logging import logger
from ..memory.builder import build_memory_from_shots
from ..memory.store import MemoryStore
from ..perception import (
    CharacterIdentifier, CinematographyTagger, DialogueMatcher,
    FeatureExtractor, FlowExtractor, MusicAnalyzer, SaliencyExtractor,
    ShotCaptioner, ShotDetector,
)
from ..types import NarrativeMemory, Shot, ShotFeatures


def preprocess(
    source_videos: list[Path],
    music: Optional[Path],
    cache_dir: Path,
    config: Config,
) -> NarrativeMemory:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(cache_dir)

    pcfg = config.preprocess
    mock = config.mocks.perception
    detector = ShotDetector(pcfg, mock=mock)
    feats_x = FeatureExtractor(pcfg, mock=mock)
    flow_x = FlowExtractor(pcfg, mock=mock)
    sal_x = SaliencyExtractor(pcfg, mock=mock)
    captioner = ShotCaptioner(pcfg, mock=mock)
    char_id = CharacterIdentifier(pcfg, mock=mock)
    dialogue = DialogueMatcher(pcfg, mock=mock)
    cine = CinematographyTagger(pcfg, mock=mock)
    music_analyzer = MusicAnalyzer(pcfg, mock=mock)

    all_shots: list[Shot] = []
    for src in source_videos:
        src = Path(src)
        logger.info(f"[preprocess] segmenting {src.name}")
        intervals = detector.detect(src)
        for i, (a, b) in enumerate(intervals):
            shot_id = f"{src.stem}__s{i:05d}"
            clip_emb = feats_x.embed_shot(src, a, b, shot_id=shot_id)
            sflow, eflow, mag = flow_x.extract_boundary_flows(src, a, b)
            ssal, esal = sal_x.extract_boundary_saliency(src, a, b)
            cap = captioner.caption(src, a, b)
            tags = cine.tag(src, a, b)
            cid_list = char_id.identify_shot(src, a, b)
            dlg = dialogue.extract(src, a, b)
            shot = Shot(
                shot_id=shot_id, source_video=str(src), start_time=a, end_time=b,
                caption=cap, cinematography=tags,
                features=ShotFeatures(
                    clip_embedding=clip_emb, start_flow=sflow, end_flow=eflow,
                    start_saliency=ssal, end_saliency=esal, avg_flow_magnitude=mag,
                ),
                character_ids=cid_list, dialogue=dlg,
            )
            all_shots.append(shot)
        logger.info(f"[preprocess] {src.name}: {len(intervals)} shots")

    # Build hierarchical memory (shots → events → stories).
    memory = build_memory_from_shots(all_shots, store)

    # Character bank (post-process).
    for ch in char_id.build_character_bank():
        store.add_character(ch)

    # Music profile.
    if music is not None:
        mp = music_analyzer.analyze(music)
        store.set_music_profile(mp)
        memory.music_profile = mp
    else:
        mp = music_analyzer.analyze(None)   # mock fallback profile
        store.set_music_profile(mp)
        memory.music_profile = mp

    logger.info(f"[preprocess] memory built: {len(memory.shots)} shots, "
                f"{len(memory.events)} events, {len(memory.characters)} characters")
    store.close()
    return memory


__all__ = ["preprocess"]
