"""Quantum Optimisation Circuit Builder.

Describe an optimisation problem; the builder extracts its features,
recommends an algorithm and parameters with the rationale in view, and
builds the Qiskit circuit.
"""

__version__ = "0.1.0"

from .build import build_qaoa
from .package import package_files, write_folder, zip_bytes
from .presets import PRESETS, Preset, get_preset
from .problems import QUBO, maxcut_qubo
from .recommend import Recommendation, recommend
from .simulate import ScanResult, expectation, scan_angles
from .textio import parse_edge_list, parse_matrix, parse_offset

__all__ = [
    "QUBO", "maxcut_qubo", "recommend", "Recommendation", "build_qaoa",
    "expectation", "scan_angles", "ScanResult",
    "PRESETS", "Preset", "get_preset",
    "parse_edge_list", "parse_matrix", "parse_offset",
    "package_files", "write_folder", "zip_bytes",
]
