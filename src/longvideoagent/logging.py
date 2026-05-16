"""Structured logging adapter.

Open-source dependency: **loguru** (https://github.com/Delgan/loguru), with a
stdlib ``logging`` fallback so v0.1 boots even when the optional ``[dev]``
extra is not installed. The interface re-exports a ``logger`` object that
supports ``.info()`` / ``.warning()`` / ``.error()`` / ``.debug()`` on both
backends.

The trajectory log is a *separate* JSONL stream — see ``utils.trajectory``.
"""
from __future__ import annotations

import logging as _stdlogging
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from loguru import logger as _logger
    _HAS_LOGURU = True
except ImportError:                                            # pragma: no cover
    _HAS_LOGURU = False
    _logger = _stdlogging.getLogger("longvideoagent")
    _logger.setLevel(_stdlogging.INFO)
    if not _logger.handlers:
        _h = _stdlogging.StreamHandler(sys.stderr)
        _h.setFormatter(_stdlogging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%H:%M:%S",
        ))
        _logger.addHandler(_h)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_sink: Optional[Path] = None,
) -> None:
    """Reset sinks and install pretty stderr + optional file sinks."""
    if _HAS_LOGURU:
        _logger.remove()
        fmt = ("<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>")
        _logger.add(sys.stderr, level=level, format=fmt, enqueue=False)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            _logger.add(log_file, level=level, format=fmt, rotation="50 MB", retention=10)
        if json_sink is not None:
            json_sink.parent.mkdir(parents=True, exist_ok=True)
            _logger.add(json_sink, level=level, serialize=True, rotation="50 MB", retention=10)
    else:                                                      # pragma: no cover
        _logger.setLevel(getattr(_stdlogging, level.upper(), _stdlogging.INFO))
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = _stdlogging.FileHandler(log_file)
            fh.setLevel(_logger.level)
            _logger.addHandler(fh)


# Re-export under a stable name so callers don't care which backend is live.
logger: Any = _logger
__all__ = ["logger", "configure_logging"]
