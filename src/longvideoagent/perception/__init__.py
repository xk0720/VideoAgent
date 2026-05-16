"""Stage-1 perception wrappers.

Each module in this package exposes a stable public API and a v0.1 mock
implementation; the docstring at the top of each file names the upstream
open-source library it will use once mocks are turned off.
"""
from .shot_detector import ShotDetector
from .feature_extractor import FeatureExtractor
from .flow_extractor import FlowExtractor
from .saliency import SaliencyExtractor
from .captioner import ShotCaptioner
from .character_id import CharacterIdentifier
from .dialogue_matcher import DialogueMatcher
from .cinematography import CinematographyTagger
from .music_analyzer import MusicAnalyzer

__all__ = [
    "ShotDetector",
    "FeatureExtractor",
    "FlowExtractor",
    "SaliencyExtractor",
    "ShotCaptioner",
    "CharacterIdentifier",
    "DialogueMatcher",
    "CinematographyTagger",
    "MusicAnalyzer",
]
