# This file's existence makes ``tests/training`` a Python package, which
# would normally shadow our top-level ``training`` package for absolute
# imports inside the test modules. To prevent that, we push the project
# root to ``sys.path`` *before* any sibling test module is imported.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
