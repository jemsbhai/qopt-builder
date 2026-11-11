"""Problem intake: optimisation problems as QUBO / Ising models.

A problem is a square matrix Q plus a constant offset, minimised over
binary variables x in {0,1}^n:

    cost(x) = x^T Q x + offset

Q may be stored symmetric or upper-triangular. Either way the coupling
between x_i and x_j is Q[i,j] + Q[j,i], and the diagonal carries the
linear terms (x_i^2 = x_i). MaxCut on a weighted graph converts to this
form exactly; other problems enter by supplying Q directly.
"""

from __future__ import annotations

import numpy as np


class QUBO:
    def __init__(self, matrix: np.ndarray, offset: float = 0.0, name: str = "qubo"):
        q = np.asarray(matrix, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("QUBO matrix must be square")
        self.matrix = q
        self.offset = float(offset)
        self.name = name

    @property
    def n(self) -> int:
        return self.matrix.shape[0]

    def cost(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.matrix @ x) + self.offset

    def density(self) -> float:
        """Fraction of variable pairs {i, j} that are coupled.

        A pair is coupled when Q[i,j] + Q[j,i] is nonzero, so a symmetric
        layout and an upper-triangular layout of the same cost function
        report the same value. For a graph with m edges this is the usual
        edge density m / (n (n - 1) / 2): 0 with no edges, 1 for a
        complete graph.
        """
        n = self.n
        if n < 2:
            return 0.0
        folded = self.matrix + self.matrix.T
        upper = folded[np.triu_indices(n, k=1)]
        coupled = int(np.count_nonzero(np.abs(upper) > 1e-12))
        return coupled / (n * (n - 1) / 2)

    def brute_force(self) -> tuple[float, np.ndarray]:
        """Exact optimum by enumeration. Feasible only for small n."""
        best, best_x = np.inf, None
        for mask in range(1 << self.n):
            x = np.array([(mask >> i) & 1 for i in range(self.n)], dtype=float)
            c = self.cost(x)
            if c < best:
                best, best_x = c, x
        return best, best_x


def maxcut_qubo(edges: list[tuple[int, int, float]], n: int) -> QUBO:
    """MaxCut as a minimisation QUBO.

    cut(x) = sum_{(i,j,w)} w (x_i + x_j - 2 x_i x_j); minimise -cut.
    """
    q = np.zeros((n, n))
    for i, j, w in edges:
        q[i, i] -= w
        q[j, j] -= w
        q[i, j] += 2.0 * w
    return QUBO(q, name="maxcut")
