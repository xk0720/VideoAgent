"""Perception modules (mock mode)."""
from __future__ import annotations

from longvideoagent.config import load_config
from longvideoagent.perception import (
    CharacterIdentifier, CinematographyTagger, DialogueMatcher,
    FeatureExtractor, FlowExtractor, MusicAnalyzer, SaliencyExtractor,
    ShotCaptioner, ShotDetector,
)


def test_shot_detector_mock(tiny_clip_path):
    cfg = load_config().preprocess
    det = ShotDetector(cfg, mock=True)
    intervals = det.detect(tiny_clip_path)
    assert len(intervals) >= 1
    assert all(b > a for a, b in intervals)


def test_feature_extractor_mock(tiny_clip_path):
    cfg = load_config().preprocess
    fx = FeatureExtractor(cfg, mock=True)
    emb = fx.embed_shot(tiny_clip_path, 0.0, 2.0)
    assert emb.shape == (cfg.feature_extractor.embed_dim,)


def test_flow_and_saliency_shapes(tiny_clip_path):
    cfg = load_config().preprocess
    flow_x = FlowExtractor(cfg, mock=True, height=16, width=16)
    sf, ef, mag = flow_x.extract_boundary_flows(tiny_clip_path, 0.0, 1.0)
    assert sf.shape == (16, 16, 2)
    assert isinstance(mag, float)
    sx = SaliencyExtractor(cfg, mock=True, height=16, width=16)
    ss, es = sx.extract_boundary_saliency(tiny_clip_path, 0.0, 1.0)
    assert ss.shape == (16, 16)
    assert es.shape == (16, 16)


def test_captioner_buffer(tiny_clip_path):
    cfg = load_config().preprocess
    cap = ShotCaptioner(cfg, mock=True)
    a = cap.caption(tiny_clip_path, 0.0, 1.0)
    b = cap.caption(tiny_clip_path, 1.0, 2.0)
    assert "context=0" in a
    assert "context=1" in b


def test_music_analyzer_mock():
    cfg = load_config().preprocess
    mp = MusicAnalyzer(cfg, mock=True).analyze(None)
    assert mp.bpm > 0
    assert len(mp.sections) == 4
    assert len(mp.beats) > 0


def test_character_and_dialogue_stubs(tiny_clip_path):
    cfg = load_config().preprocess
    cids = CharacterIdentifier(cfg, mock=True).identify_shot(tiny_clip_path, 0.0, 1.0)
    assert cids == ["char_0"]
    assert DialogueMatcher(cfg, mock=True).extract(tiny_clip_path, 0.0, 1.0) is None
    tags = CinematographyTagger(cfg, mock=True).tag(tiny_clip_path, 0.0, 1.0)
    assert tags.shot_scale in {"close_up", "medium", "long", "extreme_long", "extreme_close_up"}
