#!/usr/bin/env python
"""scripts/visualize_trajectory.py — Rich-table view of an agent trajectory.jsonl."""
from __future__ import annotations

import sys

from longvideoagent.scripts_impl import viz_trajectory_main

if __name__ == "__main__":
    sys.exit(viz_trajectory_main())
