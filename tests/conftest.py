"""Shared test setup: make the repository root importable.

`python -m pytest tests` already puts the working directory on sys.path.
This keeps a bare `pytest` call, or a call from another directory,
working the same way.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
