"""Lets the package be run as a program: python -m qopt_builder"""

import sys

from .cli import main

sys.exit(main())
