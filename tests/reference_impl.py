"""Independent reference implementations used by the tests.

Nothing in this file imports qopt_builder or Qiskit. Each function
recomputes a quantity straight from its definition, so a test that
compares the package against it is comparing two separately written
computations.

Convention used throughout: assignment number k has x_i = bit i of k,
with x_0 the least significant bit. This is also the order of Qiskit's
statevector, where bit i of the basis index belongs to qubit i.
"""

from __future__ import annotations

import math

import numpy as np


def bits(index: int, n: int) -> list[int]:
    """Assignment number `index` as a list [x_0, ..., x_{n-1}]."""
    return [(index >> i) & 1 for i in range(n)]


def qubo_cost(matrix, x, offset: float = 0.0) -> float:
    """cost(x) = sum_i sum_j Q[i][j] x_i x_j + offset, with exact summation."""
    n = len(matrix)
    terms = [float(matrix[i][j]) for i in range(n) if x[i] for j in range(n) if x[j]]
    return math.fsum(terms + [float(offset)])


def all_costs(matrix, offset: float = 0.0) -> list[float]:
    """Cost of every assignment, indexed by assignment number."""
    n = len(matrix)
    return [qubo_cost(matrix, bits(k, n), offset) for k in range(1 << n)]


def reference_minimum(matrix, offset: float = 0.0, tol: float = 1e-9):
    """Minimum over all 2^n assignments.

    Returns (value, x, n_optimal): the minimum cost, the optimal
    assignment with the lowest number, and how many assignments are
    optimal to within `tol`.
    """
    n = len(matrix)
    costs = all_costs(matrix, offset)
    value = min(costs)
    optimal = [k for k, c in enumerate(costs) if c <= value + tol]
    return value, bits(optimal[0], n), len(optimal)


def cut_weight(edges, x) -> float:
    """Total weight of the edges whose endpoints fall on different sides."""
    return math.fsum(w for i, j, w in edges if x[i] != x[j])


def qaoa_probabilities(costs, n: int, gammas, betas) -> np.ndarray:
    """Measurement distribution of textbook QAOA for a cost vector.

    Starts from the uniform superposition and applies, per layer, the
    phase separator exp(-i gamma C) followed by the mixer
    exp(-i beta X) on every qubit. `costs` is indexed by assignment
    number. A constant added to the costs is a global phase and does not
    change the result.
    """
    size = 1 << n
    state = np.full(size, 1.0 / math.sqrt(size), dtype=complex)
    costs = np.asarray(costs, dtype=float)
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * costs)
        psi = state.reshape([2] * n)
        c, s = math.cos(beta), math.sin(beta)
        for axis in range(n):
            # exp(-i beta X) = cos(beta) I - i sin(beta) X on one qubit
            psi = c * psi - 1j * s * np.flip(psi, axis=axis)
        state = psi.reshape(size)
    return np.abs(state) ** 2
