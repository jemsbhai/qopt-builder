"""Exact solver: QUBO.brute_force() and QUBO.cost() against a reference.

The exact solve is what certifies every quantum result in this project,
so it is checked on many small random instances. The reference
(tests/reference_impl.py) shares no code with the package: it recomputes
each cost from the definition with plain loops and exact summation.
"""

import numpy as np
import pytest

from qopt_builder import QUBO, maxcut_qubo
from reference_impl import all_costs, bits, cut_weight, qubo_cost, reference_minimum
from run_demo import EDGES, N


def random_qubo(seed: int, n: int, layout: str, integer: bool):
    """A seeded random instance: (matrix, offset).

    layout "upper" stores each coupling once above the diagonal;
    "symmetric" stores it on both sides. Integer weights make ties
    between assignments exact; decimal weights exercise rounding.
    """
    rng = np.random.default_rng(seed)
    if integer:
        a = rng.integers(-4, 5, size=(n, n)).astype(float)
        offset = float(rng.integers(-3, 4))
    else:
        a = np.round(rng.normal(size=(n, n)), 3)
        offset = float(np.round(rng.normal(), 3))
    matrix = np.triu(a)
    if layout == "symmetric":
        matrix = matrix + np.triu(a, 1).T
    return matrix, offset


def random_graph(seed: int, n: int):
    """A seeded random weighted graph as an edge list."""
    rng = np.random.default_rng(seed)
    return [
        (i, j, float(rng.integers(1, 6)))
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < 0.5
    ]


CASES = [
    pytest.param(n, layout, integer, seed, id=f"n{n}-{layout}-{'int' if integer else 'dec'}-s{seed}")
    for n in range(1, 11)
    for layout in ("upper", "symmetric")
    for integer in (True, False)
    for seed in (0, 1, 2)
]


@pytest.mark.parametrize("n, layout, integer, seed", CASES)
def test_brute_force_matches_reference(n, layout, integer, seed):
    matrix, offset = random_qubo(1000 * n + seed, n, layout, integer)
    value, x = QUBO(matrix, offset=offset).brute_force()
    x = [int(v) for v in x]
    ref_value, ref_x, _ = reference_minimum(matrix.tolist(), offset)

    assert value == pytest.approx(ref_value, abs=1e-9)
    # the reported assignment attains the reported value
    assert qubo_cost(matrix.tolist(), x, offset) == pytest.approx(value, abs=1e-9)
    if integer:
        # ties are exact here, and the lowest-numbered optimum is reported
        assert x == ref_x


@pytest.mark.parametrize("n, layout, integer, seed", CASES[::7])
def test_cost_matches_reference_on_every_assignment(n, layout, integer, seed):
    matrix, offset = random_qubo(1000 * n + seed, n, layout, integer)
    problem = QUBO(matrix, offset=offset)
    reference = all_costs(matrix.tolist(), offset)
    for index, expected in enumerate(reference):
        assert problem.cost(bits(index, n)) == pytest.approx(expected, abs=1e-9)


def test_result_types():
    value, x = QUBO(np.triu(np.ones((4, 4))) - 2 * np.eye(4)).brute_force()
    assert isinstance(value, float)
    assert isinstance(x, np.ndarray) and x.shape == (4,)
    assert set(np.unique(x)) <= {0.0, 1.0}


def test_ties_report_the_lowest_numbered_optimum():
    """The demo graph has six optimal assignments: three cuts, each with
    its complement. The one reported has the lowest assignment number,
    reading x_0 as the least significant bit."""
    problem = maxcut_qubo(EDGES, N)
    value, x = problem.brute_force()
    ref_value, ref_x, n_optimal = reference_minimum(problem.matrix.tolist())
    assert n_optimal == 6
    assert value == ref_value == -8.0
    assert [int(v) for v in x] == ref_x == [0, 1, 0, 0, 1, 0]


def test_all_zero_problem_reports_the_all_zero_assignment():
    value, x = QUBO(np.zeros((5, 5)), offset=1.5).brute_force()
    assert value == 1.5
    assert not x.any()


@pytest.mark.parametrize(
    "q, offset, value, x",
    [(-1.0, 0.0, -1.0, 1), (2.0, 0.0, 0.0, 0), (2.0, -3.0, -3.0, 0), (-1.0, 0.5, -0.5, 1)],
)
def test_single_variable(q, offset, value, x):
    got_value, got_x = QUBO([[q]], offset=offset).brute_force()
    assert got_value == value
    assert [int(v) for v in got_x] == [x]


@pytest.mark.parametrize("offset", [-7.25, 0.0, 3.5])
def test_offset_shifts_the_value_and_keeps_the_assignment(offset):
    matrix, _ = random_qubo(42, 7, "upper", True)
    base_value, base_x = QUBO(matrix).brute_force()
    value, x = QUBO(matrix, offset=offset).brute_force()
    assert value == pytest.approx(base_value + offset, abs=1e-12)
    assert np.array_equal(x, base_x)


# ------------------------------------------------------------------- MaxCut

@pytest.mark.parametrize("n, seed", [(n, seed) for n in range(2, 10) for seed in (0, 1)])
def test_maxcut_cost_is_minus_the_cut_weight(n, seed):
    """The MaxCut conversion is exact: cost(x) = -cut(x) for every x."""
    edges = random_graph(100 * n + seed, n)
    problem = maxcut_qubo(edges, n)
    for index in range(1 << n):
        x = bits(index, n)
        assert problem.cost(x) == -cut_weight(edges, x)


@pytest.mark.parametrize("n, seed", [(n, seed) for n in range(2, 10) for seed in (0, 1)])
def test_maxcut_optimum_is_the_maximum_cut(n, seed):
    edges = random_graph(100 * n + seed, n)
    value, x = maxcut_qubo(edges, n).brute_force()
    best_cut = max(cut_weight(edges, bits(index, n)) for index in range(1 << n))
    assert -value == best_cut
    assert cut_weight(edges, [int(v) for v in x]) == best_cut


def test_demo_cut_by_hand():
    """x = [0 1 0 0 1 0] puts nodes 1 and 4 on one side. Six of the eight
    edges cross (weights 1 + 1 + 2 + 1 + 1 + 2 = 8); edges (0,2) and (3,5)
    do not."""
    x = [0, 1, 0, 0, 1, 0]
    crossing = [(i, j, w) for i, j, w in EDGES if x[i] != x[j]]
    assert sorted((i, j) for i, j, _ in crossing) == [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    assert cut_weight(EDGES, x) == 8.0
    assert maxcut_qubo(EDGES, N).cost(x) == -8.0
