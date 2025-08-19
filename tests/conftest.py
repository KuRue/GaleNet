"""Pytest configuration to activate import hooks."""

import sys
from pathlib import Path

# Add repository root to sys.path so `sitecustomize` can be imported, which
# enables the PyTorch docstring patch before any tests run.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sitecustomize  # noqa: E402,F401
