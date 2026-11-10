"""Problem intake: optimisation problems as QUBO / Ising models.

A problem is a symmetric matrix Q (upper-triangular convention) plus a
constant offset, minimised over binary variables x in {0,1}^n:

    cost(x) = sum_{i<=j} Q[i,j] x_i x_j + offset

MaxCut on a weighted graph converts to this form exactly; other
problems enter by supplying Q directly.
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
        """Fraction of nonzero off-diagonal couplings."""
        n = self.n
        if n < 2:
            return 0.0
        off = self.matrix - np.diag(np.diag(self.matrix))
        return float(np.count_nonzero(np.abs(off) > 1e-12) / (n * (n - 1)))

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
