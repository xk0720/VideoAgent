"""Typed configuration loader.

Open-source dependencies used here:
    • PyYAML        — https://pyyaml.org           (YAML parsing)
    • python-dotenv — https://github.com/theskumar/python-dotenv (.env loading)

Originally drafted with pydantic v2; refactored to plain dataclasses for
v0.1 so the base install is small and the tests don't depend on
``pydantic_core``'s binary wheel. The migration to ``pydantic.BaseModel``
is mechanical (each field already has a type) — see ``docs/decisions.md``
D-012 if you want to flip it back.

The config tree mirrors configs/default.yaml; runtime overrides (CLI flags,
env vars) are merged on top by ``load_config(...)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:                                            # pragma: no cover
    pass


# ─────────────────────────────────────────────────────────────────────
# Sub-models — one dataclass per config block in default.yaml
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ShotDetectorCfg:
    name: str = "pyscenedetect"
    threshold: float = 27.0
    min_scene_len: int = 15


@dataclass
class FeatureExtractorCfg:
    name: str = "clip-vit-base-patch32"
    stride: int = 4
    embed_dim: int = 512


@dataclass
class FlowExtractorCfg:
    name: str = "raft-large"
    spatial_pool: int = 8


@dataclass
class SaliencyCfg:
    name: str = "u2net"


@dataclass
class CaptionerCfg:
    name: str = "qwen-vl-8b-instruct"
    buffer_size: int = 10
    max_new_tokens: int = 128


@dataclass
class CinematographyCfg:
    name: str = "shotvl-3b"


@dataclass
class MusicAnalyzerCfg:
    name: str = "allin1"


@dataclass
class CharacterIdCfg:
    detector: str = "insightface"
    fallback: str = "solider"
    cluster_threshold: float = 0.6


@dataclass
class DialogueCfg:
    ocr_engine: str = "easyocr"
    voiceprint: str = "wespeaker"


@dataclass
class ParallelCfg:
    num_workers: int = 4


@dataclass
class PreprocessCfg:
    shot_detector: ShotDetectorCfg = field(default_factory=ShotDetectorCfg)
    feature_extractor: FeatureExtractorCfg = field(default_factory=FeatureExtractorCfg)
    flow_extractor: FlowExtractorCfg = field(default_factory=FlowExtractorCfg)
    saliency: SaliencyCfg = field(default_factory=SaliencyCfg)
    captioner: CaptionerCfg = field(default_factory=CaptionerCfg)
    cinematography: CinematographyCfg = field(default_factory=CinematographyCfg)
    music_analyzer: MusicAnalyzerCfg = field(default_factory=MusicAnalyzerCfg)
    character_id: CharacterIdCfg = field(default_factory=CharacterIdCfg)
    dialogue: DialogueCfg = field(default_factory=DialogueCfg)
    parallel: ParallelCfg = field(default_factory=ParallelCfg)


@dataclass
class PlanModelsCfg:
    screenwriter: str = "deepseek-v3"
    director: str = "deepseek-v3"
    orchestrator: str = "claude-sonnet-4-5"


@dataclass
class PlanCfg:
    max_iterations: int = 5
    models: PlanModelsCfg = field(default_factory=PlanModelsCfg)
    use_langgraph: bool = False


@dataclass
class RetrievalCfg:
    beam_width: int = 3
    sliding_stride: int = 4
    top_k_pool: int = 200


@dataclass
class GenerationCfg:
    enabled: bool = True
    backend: str = "omniweaving"
    fallback_threshold: float = 0.4
    duration_default: float = 4.0


@dataclass
class MetricWeightsCfg:
    m1_prompt: float = 0.20
    m2_seg_consistency: float = 0.15
    m3_motion_continuity: float = 0.20
    m4_framing: float = 0.15
    m5_beat_sync: float = 0.15
    m6_energy: float = 0.15


@dataclass
class ComposeCfg:
    editor_model: str = "claude-sonnet-4-5"
    validator_model: str = "qwen-vl-8b-instruct"
    max_editor_steps: int = 10
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    metric_weights: MetricWeightsCfg = field(default_factory=MetricWeightsCfg)
    validator_threshold: float = 6.0


@dataclass
class AssemblyCfg:
    output_codec: str = "libx264"
    output_pix_fmt: str = "yuv420p"
    output_fps: int = 30
    crossfade_duration: float = 0.0
    music_volume: float = 0.85
    loglevel: str = "error"


@dataclass
class TrajectoryCfg:
    enabled: bool = True
    format: str = "jsonl"
    redact_large_tensors: bool = True


@dataclass
class MocksCfg:
    perception: bool = True
    llm: bool = True
    video_gen: bool = True
    reward: bool = True


@dataclass
class Config:
    project_name: str = "longvideoagent"
    cache_root: Path = field(default_factory=lambda: Path("./.cache"))
    output_root: Path = field(default_factory=lambda: Path("./outputs"))
    random_seed: int = 42

    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    plan: PlanCfg = field(default_factory=PlanCfg)
    compose: ComposeCfg = field(default_factory=ComposeCfg)
    assembly: AssemblyCfg = field(default_factory=AssemblyCfg)
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)
    mocks: MocksCfg = field(default_factory=MocksCfg)


# ─────────────────────────────────────────────────────────────────────
# Validation: build dataclass tree from raw dict, coercing types
# ─────────────────────────────────────────────────────────────────────


def _from_dict(cls, data: Any):
    """Convert a nested dict into a dataclass instance, applying type coercion
    for the simple cases (Path, basic scalars). Unknown keys raise to surface typos."""
    if data is None:
        return cls()
    if not is_dataclass(cls) or not isinstance(data, dict):
        # No coercion target; let dataclass __init__ handle it.
        return data

    from typing import get_type_hints
    hints = get_type_hints(cls)        # resolves string annotations to real types
    kwargs: dict[str, Any] = {}
    field_names = {f.name for f in fields(cls)}
    for key, raw in data.items():
        if key not in field_names:
            raise ValueError(f"{cls.__name__}: unknown config key '{key}'")
        anno = hints.get(key)
        kwargs[key] = _coerce(raw, anno)
    return cls(**kwargs)


def _coerce(value: Any, anno: Any) -> Any:
    """Best-effort type coercion for the small set of types we use in YAML."""
    if value is None:
        return None
    origin = get_origin(anno)
    args = get_args(anno)
    # Path
    if anno is Path or anno == "Path":
        return Path(os.path.expanduser(os.path.expandvars(str(value))))
    # Optional[X] / Union — try each member, return first that works.
    if origin is not None and args:
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(value, a)
            except Exception:
                continue
        return value
    # Nested dataclass
    if isinstance(anno, type) and is_dataclass(anno):
        return _from_dict(anno, value)
    # Primitives — leave to the dataclass __init__ (no strict cast to avoid
    # surprises like int("0.4") for float fields).
    return value


_DEFAULT_CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base``; values in overrides win."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    config_path: Optional[Path | str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> Config:
    """Load configs/default.yaml (or a custom path), merge overrides, validate."""
    path = Path(config_path) if config_path else _DEFAULT_CFG_PATH
    if path.exists():
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        raw = {}

    if os.getenv("LVA_CACHE_ROOT"):
        raw["cache_root"] = os.environ["LVA_CACHE_ROOT"]
    if os.getenv("LVA_OUTPUT_ROOT"):
        raw["output_root"] = os.environ["LVA_OUTPUT_ROOT"]

    if overrides:
        raw = _deep_merge(raw, overrides)

    return _from_dict(Config, raw)


# ─────────────────────────────────────────────────────────────────────
# Model alias / prompt loaders (kept small; used by agent ABCs)
# ─────────────────────────────────────────────────────────────────────

_PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"
_CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Tiny helper for loading any of the side-car YAMLs (agents/, models/, heuristics/)."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_prompt(name_or_path: str) -> str:
    """Load a prompt template from src/longvideoagent/prompts/<name>.txt by
    either bare name ('screenwriter') or a relative/absolute path."""
    p = Path(name_or_path)
    if not p.is_absolute() and not p.exists():
        candidate = _PROMPTS_ROOT / (p.name if p.suffix == ".txt" else f"{name_or_path}.txt")
        if candidate.exists():
            p = candidate
    return p.read_text(encoding="utf-8")


def configs_dir() -> Path:
    return _CONFIGS_ROOT


def prompts_dir() -> Path:
    return _PROMPTS_ROOT
