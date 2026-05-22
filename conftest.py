"""Project-root conftest.

pytest auto-imports any ``conftest.py`` it finds while walking up from a
test file to ``rootdir``. We use this hook to make BOTH ``src/`` and the
project root importable, so tests can ``import longvideoagent.*`` AND
``import training.*`` without environment fiddling. This is the simplest
fix that survives both ``pytest tests/`` and ``pytest tests/training/...``
invocations across the various pytest / pyproject combinations we've hit.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
