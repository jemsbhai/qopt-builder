"""Problem intake: optimisation problems as QUBO / Ising models.

A problem is a square matrix Q plus a constant offset, minimised over
binary variables x in {0,1}^n:

    cost(x) = x^T Q x + offset

Q may be stored symmetric, or triangular with each coupling written
once on one side of the diagonal. Either way the coupling between x_i
and x_j is Q[i,j] + Q[j,i], and the diagonal carries the linear terms
(x_i^2 = x_i). MaxCut on a weighted graph converts to this form exactly;
other problems enter by supplying Q directly.

Input is checked on the way in. A problem that fails a check is rejected
with a ValueError that says what is wrong and where: a matrix that is
empty, is not square, holds something that is not a finite number, or
is neither symmetric nor triangular; an edge that names a node outside
the graph, joins a node to itself, or carries a weight that is not a
finite number. The web port (web/qubo.js) runs the same checks in the
same order and words its errors the same way.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping

import numpy as np

# Assignments whose cost is within this distance of the minimum count as
# tied; brute_force() then reports the lowest-numbered one.
TIE_TOLERANCE = 1e-9

# brute_force() evaluates 2**_BLOCK_BITS assignments per vectorised block.
_BLOCK_BITS = 20


# Q[i,j] and Q[j,i] count as equal when they differ by no more than this,
# relative to the larger of the two (or to 1, for entries smaller than 1).
SYMMETRY_TOLERANCE = 1e-9

# An entry no larger than this counts as zero when deciding whether a
# matrix is triangular. density() uses the same size for "not coupled".
ZERO_TOLERANCE = 1e-12


def is_number(value) -> bool:
    """A real number, Python or numpy. Booleans, strings and None are not."""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def is_finite_number(value) -> bool:
    return is_number(value) and math.isfinite(float(value))


def whole_number(value) -> int | None:
    """`value` as an int when it is a whole number (5, 5.0, numpy 5),
    otherwise None."""
    if not is_number(value):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    as_float = float(value)
    if not math.isfinite(as_float) or as_float != math.floor(as_float):
        return None
    return int(as_float)


_NOT_A_TABLE = "QUBO matrix must be a square table of numbers, given as a list of rows."
_EMPTY = "QUBO matrix is empty; a problem needs at least one variable."


def _not_square(rows: int, row: int, length: int) -> str:
    return f"QUBO matrix must be square: number of rows is {rows}, but row {row} has length {length}."


def _checked_matrix(matrix) -> np.ndarray:
    """`matrix` as a new float array, or a ValueError saying what is wrong.

    The checks run in a fixed order: it is a list of rows; it is not
    empty; every row is as long as the number of rows; every entry is a
    number, and finite; the layout is symmetric or triangular.
    makeQUBO() in web/qubo.js follows the same order, so both ports
    report the same fault for the same input.
    """
    if isinstance(matrix, np.ndarray) and matrix.ndim == 2 and matrix.dtype.kind in "iuf":
        # a numeric array: the same checks, without visiting entries one by one
        rows, columns = matrix.shape
        if matrix.size == 0:
            raise ValueError(_EMPTY)
        if columns != rows:
            raise ValueError(_not_square(rows, 0, columns))
        q = matrix.astype(float)  # always a copy
        bad = np.argwhere(~np.isfinite(q))
        if len(bad):
            raise ValueError(f"QUBO matrix entry [{bad[0][0]}][{bad[0][1]}] is not finite.")
    else:
        if not isinstance(matrix, (list, tuple, np.ndarray)) or getattr(matrix, "ndim", 1) == 0:
            raise ValueError(_NOT_A_TABLE)
        table = list(matrix)
        if any(not isinstance(row, (list, tuple, np.ndarray)) or getattr(row, "ndim", 1) == 0 for row in table):
            raise ValueError(_NOT_A_TABLE)
        if not any(len(row) for row in table):
            raise ValueError(_EMPTY)
        for i, row in enumerate(table):
            if len(row) != len(table):
                raise ValueError(_not_square(len(table), i, len(row)))
        for i, row in enumerate(table):
            for j, value in enumerate(row):
                if not is_number(value):
                    raise ValueError(f"QUBO matrix entry [{i}][{j}] is not a number.")
                if not math.isfinite(float(value)):
                    raise ValueError(f"QUBO matrix entry [{i}][{j}] is not finite.")
        q = np.array([[float(value) for value in row] for row in table])

    n = q.shape[0]
    above = np.triu_indices(n, k=1)  # the pairs i < j, row by row
    upper, lower = q[above], q.T[above]
    if np.any(np.abs(upper) > ZERO_TOLERANCE) and np.any(np.abs(lower) > ZERO_TOLERANCE):
        scale = np.maximum(1.0, np.maximum(np.abs(upper), np.abs(lower)))
        differ = np.abs(upper - lower) > SYMMETRY_TOLERANCE * scale
        if differ.any():
            k = int(np.argmax(differ))
            i, j = int(above[0][k]), int(above[1][k])
            raise ValueError(
                f"QUBO matrix must be symmetric or triangular. Entries [{i}][{j}] and [{j}][{i}] "
                "differ, and there are nonzero entries on both sides of the diagonal."
            )
    return q


class QUBO:
    def __init__(self, matrix, offset: float = 0.0, name: str = "qubo"):
        """A problem from its matrix and constant offset.

        `matrix` is a square table of finite numbers (nested lists or a
        numpy array), symmetric or triangular; it is copied. Anything
        else raises ValueError, as does an offset that is not a finite
        number.
        """
        self.matrix = _checked_matrix(matrix)
        if not is_finite_number(offset):
            raise ValueError("QUBO offset must be a finite number.")
        self.offset = float(offset)
        self.name = name

    @property
    def n(self) -> int:
        return self.matrix.shape[0]

    def cost(self, x) -> float:
        """Cost of the assignment x: one entry per variable, each 0 or 1
        (True and False count as 1 and 0)."""
        if isinstance(x, (str, bytes)) or not isinstance(x, (list, tuple, np.ndarray)) or getattr(x, "ndim", 1) != 1:
            raise ValueError("Assignment must be a list of 0s and 1s, one per variable.")
        if len(x) != self.n:
            raise ValueError(
                f"Assignment must have one entry per variable: this problem has {self.n}, but x has {len(x)}."
            )
        for i, value in enumerate(x):
            if isinstance(value, (bool, np.bool_)):
                continue
            if not is_number(value) or float(value) not in (0.0, 1.0):
                raise ValueError(f"Assignment entries must be 0 or 1; entry {i} is not.")
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
        coupled = int(np.count_nonzero(np.abs(upper) > ZERO_TOLERANCE))
        return coupled / (n * (n - 1) / 2)

    def all_costs(self) -> np.ndarray:
        """Cost of every assignment, as an array of length 2^n.

        Entry k is the cost of assignment number k, which has x_i = bit i
        of k. This is also the order of a Qiskit statevector, so the
        expected cost of a measured state is probabilities @ all_costs().
        The array is held in memory, so this is for n up to 20; the exact
        solver, brute_force(), has no such bound.
        """
        if self.n > _BLOCK_BITS:
            raise ValueError(
                f"all_costs() holds 2^n numbers in memory and stops at n={_BLOCK_BITS}; "
                f"this problem has n={self.n}. brute_force() works block by block and "
                "does not have this bound."
            )
        diag = np.diag(self.matrix).copy()
        return _block_costs(diag, self.matrix + self.matrix.T, self.n, self.offset)

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

        # low[m]: cost of assignment m when every variable above the block is 0
        low = _block_costs(diag, coupling, n_low, self.offset)

        best, best_index = np.inf, 0
        scratch, block = np.empty(size), np.empty(size)
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


def _block_costs(diag: np.ndarray, coupling: np.ndarray, n_low: int, offset: float) -> np.ndarray:
    """Costs of all 2^n_low assignments of the first n_low variables, the
    rest held at 0. Built by doubling: the half of the array with
    x_k = 1 is the half with x_k = 0 plus an array of increments."""
    size = 1 << n_low
    costs = np.empty(size)
    costs[0] = offset
    scratch = np.empty(max(size >> 1, 1))
    for k in range(n_low):
        _fill_linear(scratch, diag[k], coupling[:k, k])
        half = 1 << k
        np.add(costs[:half], scratch[:half], out=costs[half:2 * half])
    return costs


def _fill_linear(out: np.ndarray, constant: float, weights) -> None:
    """Write constant + sum_j weights[j] * x_j(m) into out[m], for every
    assignment m of len(weights) binary variables (x_j = bit j of m)."""
    out[0] = constant
    for j, w in enumerate(weights):
        half = 1 << j
        np.add(out[:half], w, out=out[half:2 * half])


def maxcut_qubo(edges, n: int) -> QUBO:
    """MaxCut as a minimisation QUBO.

    cut(x) = sum_{(i,j,w)} w (x_i + x_j - 2 x_i x_j); minimise -cut.

    `edges` is a list of (i, j, weight), or (i, j) for weight 1, on
    nodes numbered 0 to n - 1. An edge may be written in either
    direction: its coupling is stored above the diagonal both ways.
    Edges given more than once add up. An edge that names a node
    outside the graph, joins a node to itself, or has a weight that is
    not a finite number raises ValueError.
    """
    count = whole_number(n)
    if count is None or count < 1:
        raise ValueError("Number of nodes must be a positive whole number.")
    not_a_list = "Edges must be a list of (i, j) or (i, j, weight) entries."
    if isinstance(edges, (str, bytes, Mapping)):
        raise ValueError(not_a_list)
    try:
        edges = list(edges)
    except TypeError:
        raise ValueError(not_a_list) from None
    q = np.zeros((count, count))
    for k, edge in enumerate(edges):
        if not isinstance(edge, (list, tuple, np.ndarray)) or len(edge) not in (2, 3):
            raise ValueError(f"edges[{k}] must be (i, j) or (i, j, weight).")
        i, j = whole_number(edge[0]), whole_number(edge[1])
        if i is None or j is None:
            raise ValueError(f"edges[{k}]: node numbers must be whole numbers.")
        for node in (i, j):
            if not 0 <= node < count:
                raise ValueError(f"edges[{k}]: node {node} is outside the range 0 to {count - 1}.")
        if i == j:
            raise ValueError(f"edges[{k}] joins node {i} to itself.")
        w = edge[2] if len(edge) == 3 else 1.0
        if not is_finite_number(w):
            raise ValueError(f"edges[{k}]: weight must be a finite number.")
        w = float(w)
        q[i, i] -= w
        q[j, j] -= w
        q[min(i, j), max(i, j)] += 2.0 * w
    return QUBO(q, name="maxcut")
