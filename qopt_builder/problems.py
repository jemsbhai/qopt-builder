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

# Assignments whose cost is within this distance of the minimum count as
# tied; brute_force() then reports the lowest-numbered one.
TIE_TOLERANCE = 1e-9

# brute_force() evaluates 2**_BLOCK_BITS assignments per vectorised block.
_BLOCK_BITS = 20


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
        """Exact optimum by enumerating all 2^n assignments.

        Returns (cost, x). Assignment number k has x_i = bit i of k, with
        x_0 the least significant bit. When several assignments are
        optimal (their costs agree to within TIE_TOLERANCE) the one with
        the lowest number is returned; the web port applies the same
        rule, so both report the same x. The returned cost is cost(x)
        evaluated directly from the matrix.

        Every assignment is visited, in blocks rather than one at a time.
        A block holds all settings of the low 20 variables and is filled
        by doubling: the costs with x_k = 1 are the costs with x_k = 0
        plus one array of increments. The remaining variables are stepped
        through one setting per block. Time grows as 2^n; memory stays at
        a few arrays of 2^20 numbers whatever n is.
        """
        n = self.n
        diag = np.diag(self.matrix).copy()
        coupling = self.matrix + self.matrix.T  # coupling[j, k]: coefficient of x_j x_k, j != k
        n_low = min(n, _BLOCK_BITS)
        size = 1 << n_low
        scratch = np.empty(size)

        # low[m]: cost of assignment m when every variable above the block is 0
        low = np.empty(size)
        low[0] = self.offset
        for k in range(n_low):
            _fill_linear(scratch, diag[k], coupling[:k, k])
            half = 1 << k
            np.add(low[:half], scratch[:half], out=low[half:2 * half])

        best, best_index = np.inf, 0
        block = np.empty(size)
        for high in range(1 << (n - n_low)):
            if high == 0:
                costs = low
            else:
                # switching on the high variables in `high` adds a constant
                # and a linear term in the low variables
                on = [n_low + t for t in range(n - n_low) if (high >> t) & 1]
                constant = diag[on].sum() + sum(
                    coupling[a, b] for i, a in enumerate(on) for b in on[i + 1:]
                )
                _fill_linear(scratch, constant, coupling[:n_low, on].sum(axis=1))
                costs = np.add(low, scratch, out=block)
            lowest = float(costs.min())
            if lowest < best - TIE_TOLERANCE:
                # a later block can only tie, and ties keep the lower number
                best = lowest
                best_index = (high << n_low) | int(np.argmax(costs <= lowest + TIE_TOLERANCE))

        x = np.array([(best_index >> i) & 1 for i in range(n)], dtype=float)
        return self.cost(x), x


def _fill_linear(out: np.ndarray, constant: float, weights) -> None:
    """Write constant + sum_j weights[j] * x_j(m) into out[m], for every
    assignment m of len(weights) binary variables (x_j = bit j of m)."""
    out[0] = constant
    for j, w in enumerate(weights):
        half = 1 << j
        np.add(out[:half], w, out=out[half:2 * half])


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
