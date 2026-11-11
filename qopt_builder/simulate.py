"""Statevector scoring of a QAOA circuit, and the coarse angle scan.

A QAOA circuit is scored by its expected cost: simulate the statevector
with Qiskit, turn it into measurement probabilities, and average the
problem's own cost over them. The scan draws angles at random and keeps
the best expected cost it sees. It is a first look at what the circuit
can reach, not an optimiser, and it is always reported next to the
certified optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .build import build_qaoa
from .problems import QUBO

SCAN_SAMPLES = 200
SCAN_SEED = 20261110

# Largest n the demo simulates: 2^14 amplitudes per sample. The browser
# demo stops at the same size. Larger problems still get the
# recommendation, the certified optimum and the circuit.
SIMULATION_LIMIT = 14


@dataclass
class ScanResult:
    value: float  # lowest expected cost over the samples
    gammas: np.ndarray  # the angles that gave it
    betas: np.ndarray
    trace: np.ndarray  # expected cost of every sample, in the order drawn


def expectation(qc: QuantumCircuit, costs: np.ndarray, values: dict) -> float:
    """Expected cost of measuring `qc` with its angles set to `values`.

    `costs` is the cost of every assignment, from QUBO.all_costs(); it
    shares the bit order of the statevector. `values` maps the circuit's
    Parameter objects to numbers.
    """
    # strict=False: a cost function with no terms has no gate for gamma to act on
    bound = qc.assign_parameters(values, strict=False)
    probabilities = Statevector.from_instruction(bound).probabilities()
    return float(probabilities @ costs)


def scan_angles(problem: QUBO, p: int, samples: int = SCAN_SAMPLES, seed: int = SCAN_SEED) -> ScanResult:
    """Score `samples` random angle settings of the depth-p QAOA circuit.

    Each sample draws p gammas and then p betas, uniformly in [0, pi),
    from a generator seeded with `seed`, so a run is reproducible.
    """
    qc, meta = build_qaoa(problem, p)
    costs = problem.all_costs()
    rng = np.random.default_rng(seed)
    trace = np.empty(samples)
    best = (np.inf, None, None)
    for s in range(samples):
        gammas = rng.uniform(0, np.pi, size=p)
        betas = rng.uniform(0, np.pi, size=p)
        values = {meta["gammas"][k]: gammas[k] for k in range(p)}
        values.update({meta["betas"][k]: betas[k] for k in range(p)})
        trace[s] = expectation(qc, costs, values)
        if trace[s] < best[0]:
            best = (float(trace[s]), gammas, betas)
    return ScanResult(best[0], best[1], best[2], trace)
