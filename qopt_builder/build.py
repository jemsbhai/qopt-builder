"""Circuit factory: build the circuit the recommender chose.

Currently builds QAOA for a QUBO: p layers of the phase separator
exp(-i gamma H_C) (ZZ rotations for couplings, Z rotations for linear
terms) and the mixer exp(-i beta H_B) (X rotations), with angles as
free parameters the caller optimises. H_C is the Ising form of the
QUBO under x = (1 - z) / 2.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from .problems import QUBO


def _ising_terms(problem: QUBO) -> tuple[np.ndarray, np.ndarray, float]:
    """QUBO -> Ising: cost = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const."""
    q = problem.matrix.copy()
    # fold symmetric part: cost uses x @ Q @ x with full matrix
    n = problem.n
    h = np.zeros(n)
    J = np.zeros((n, n))
    const = problem.offset
    for i in range(n):
        const += q[i, i] / 2.0
        h[i] -= q[i, i] / 2.0
        for j in range(i + 1, n):
            w = q[i, j] + q[j, i]
            const += w / 4.0
            h[i] -= w / 4.0
            h[j] -= w / 4.0
            J[i, j] = w / 4.0
    return h, J, const


def build_qaoa(problem: QUBO, p: int) -> tuple[QuantumCircuit, dict]:
    """Return (parameterised QAOA circuit, ising metadata incl. offset)."""
    n = problem.n
    h, J, const = _ising_terms(problem)
    betas = [Parameter(f"b{k}") for k in range(p)]
    gammas = [Parameter(f"g{k}") for k in range(p)]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        g, b = gammas[layer], betas[layer]
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * g * h[i], i)
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2.0 * g * J[i, j], i, j)
        qc.rx(2.0 * b, range(n))
    return qc, {"h": h, "J": J, "offset": const, "p": p, "betas": betas, "gammas": gammas}
