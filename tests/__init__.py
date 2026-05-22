# Marker file. Its existence changes pytest's import-mode resolution so
# that test modules are imported as ``tests.<sub>.<file>`` and pytest puts
# the project root (not ``tests/<sub>``) on ``sys.path``. That keeps our
# absolute imports of ``training.*`` from being shadowed by the
# ``tests/training/`` package.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
