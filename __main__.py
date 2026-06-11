"""Entry point for `python -m phoneanalyzer` (run from the parent directory)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli import main

sys.exit(main())
