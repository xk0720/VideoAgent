#!/usr/bin/env python
"""scripts/build_memory.py — print/derive narrative memory stats from a cache dir."""
from __future__ import annotations

import sys

from longvideoagent.scripts_impl import build_memory_main

if __name__ == "__main__":
    sys.exit(build_memory_main())
