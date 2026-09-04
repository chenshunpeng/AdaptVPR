#!/usr/bin/env python3
"""Command-line wrapper for the AdaptVPR manifest builder."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.manifest import main


if __name__ == "__main__":
    main()
