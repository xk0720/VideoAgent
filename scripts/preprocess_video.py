#!/usr/bin/env python
"""scripts/preprocess_video.py — thin wrapper around longvideoagent.scripts_impl.

Use either ``python scripts/preprocess_video.py ...`` or, after install,
``lva-preprocess ...``.
"""
from __future__ import annotations

import sys

from longvideoagent.scripts_impl import preprocess_main

if __name__ == "__main__":
    sys.exit(preprocess_main())
