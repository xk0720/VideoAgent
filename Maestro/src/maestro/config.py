"""Config loading. Config-driven: no hardcoded agent/model/metric params."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_dotenv(path: Path, override: bool = False) -> int:
    """Load KEY=VALUE lines from a .env file into os.environ; returns how many
    were set. Standard dotenv semantics, zero dependencies:
      • blank lines and `#` comments skipped; optional `export ` prefix ok;
      • values may be 'single' or "double" quoted (quotes stripped);
      • EXISTING environment variables WIN unless override=True — a real
        exported key always beats the file, so CI/server env stays in charge.
    Missing file → 0 (no error): the file is a convenience, not a requirement."""
    import os

    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\'\"":
            value = value[1:-1]
        if not key or (key in os.environ and not override):
            continue
        if value == "":
            continue                      # 空值占位行不写入(模板留空的 key)
        os.environ[key] = value
        n += 1
    return n


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not installed; `pip install pyyaml`")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Config:
    """Thin attribute/dict accessor over a merged YAML tree."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def data(self) -> dict:
        return self._data


def load_config(
    path: Path | str | None = None, overrides: dict | None = None
) -> Config:
    """Load default.yaml, optionally merge a user config and dict overrides."""
    base_path = _DEFAULT_CONFIG_DIR / "default.yaml"
    data = load_yaml(base_path) if base_path.exists() else {}
    if path is not None:
        data = _deep_merge(data, load_yaml(Path(path)))
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)
