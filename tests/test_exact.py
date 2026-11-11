"""Exact solver: QUBO.brute_force() and QUBO.cost() against a reference.

The exact solve is what certifies every quantum result in this project,
so it is checked on many small random instances. The reference
(tests/reference_impl.py) shares no code with the package: it recomputes
each cost from the definition with plain loops and exact summation.
"""

import numpy as np
import pytest

from qopt_builder import QUBO, maxcut_qubo, problems
from qopt_builder.problems import TIE_TOLERANCE
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
    # and it is the lowest-numbered optimum
    assert x == ref_x


@pytest.mark.parametrize("block_bits", [0, 1, 3])
@pytest.mark.parametrize("n, layout, integer, seed", CASES[1::5])
def test_brute_force_across_block_boundaries(monkeypatch, block_bits, n, layout, integer, seed):
    """brute_force() works through the assignments in blocks of
    2**_BLOCK_BITS. With tiny blocks, small problems take the same
    many-block route that n > 20 takes at the real block size."""
    monkeypatch.setattr(problems, "_BLOCK_BITS", block_bits)
    matrix, offset = random_qubo(1000 * n + seed, n, layout, integer)
    value, x = QUBO(matrix, offset=offset).brute_force()
    ref_value, ref_x, _ = reference_minimum(matrix.tolist(), offset)
    assert value == pytest.approx(ref_value, abs=1e-9)
    assert [int(v) for v in x] == ref_x


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


def test_rounding_noise_does_not_decide_a_tie():
    """Choosing x_0 alone costs -0.3. Choosing x_1 and x_2 costs
    -0.1 - 0.2, the same on paper and -0.30000000000000004 in floating
    point. The two are tied, and the lower-numbered one is reported; the
    strict floating-point minimum would have picked the other."""
    matrix = np.diag([-0.3, -0.1, -0.2])
    matrix[0, 1] = matrix[0, 2] = 1.0  # x_0 together with either of the others costs more
    problem = QUBO(matrix)
    gap = problem.cost([1, 0, 0]) - problem.cost([0, 1, 1])
    assert 0.0 < gap < TIE_TOLERANCE
    value, x = problem.brute_force()
    assert [int(v) for v in x] == [1, 0, 0]
    assert value == -0.3


@pytest.mark.parametrize("block_bits", [0, 2, 20])
def test_tie_in_a_later_block_keeps_the_lower_number(monkeypatch, block_bits):
    """Two optimal assignments, x = [1 0 0 0 0] (number 1) and
    x = [0 0 0 0 1] (number 16), which fall in different blocks when the
    blocks are small."""
    monkeypatch.setattr(problems, "_BLOCK_BITS", block_bits)
    matrix = np.diag([-1.0, 0.5, 0.5, 0.5, -1.0])
    matrix[0, 4] = 1.0  # setting both costs -1, the same as setting either
    value, x = QUBO(matrix).brute_force()
    assert value == -1.0
    assert [int(v) for v in x] == [1, 0, 0, 0, 0]


@pytest.mark.parametrize("block_bits", [0, 2, 20])
def test_optimum_only_in_the_last_block(monkeypatch, block_bits):
    monkeypatch.setattr(problems, "_BLOCK_BITS", block_bits)
    value, x = QUBO(np.diag([1.0, 1.0, 1.0, 1.0, -2.0])).brute_force()
    assert value == -2.0
    assert [int(v) for v in x] == [0, 0, 0, 0, 1]


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


# ----------------------------------------- larger n, answer known in advance

def planted(n: int, seed: int):
    """A QUBO whose unique optimum is a chosen assignment t, with cost 0.

    cost(x) = sum over edges of w * ((x_i - t_i) - (x_j - t_j))^2, plus
    (x_0 - t_0)^2, expanded into matrix form. Every term is a square, so
    the cost is zero only at x = t (the graph is connected).
    """
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, size=n)
    edges = [(i, i + 1, float(rng.integers(1, 4))) for i in range(n - 1)]
    edges += [(int(a), int(b), float(rng.integers(1, 4)))
              for a, b in rng.integers(0, n, size=(2 * n, 2)) if a < b]
    q = np.zeros((n, n))
    offset = 0.0

    def add_linear(i, w):  # w * (x_i - t_i)^2 = w * ((1 - 2 t_i) x_i + t_i)
        nonlocal offset
        q[i, i] += w * (1 - 2 * t[i])
        offset += w * t[i]

    add_linear(0, 1.0)
    for i, j, w in edges:
        # w * (a - b)^2 with a = x_i - t_i and b = x_j - t_j
        add_linear(i, w)
        add_linear(j, w)
        q[i, j] += -2 * w            # -2 w x_i x_j
        q[i, i] += 2 * w * t[j]      # +2 w t_j x_i
        q[j, j] += 2 * w * t[i]      # +2 w t_i x_j
        offset += -2 * w * t[i] * t[j]
    return QUBO(q, offset=offset), [int(v) for v in t]


@pytest.mark.parametrize("n", [5, 8])
def test_planted_construction_is_what_it_claims(n):
    problem, t = planted(n, seed=n)
    value, x, n_optimal = reference_minimum(problem.matrix.tolist(), problem.offset)
    assert (value, x, n_optimal) == (0.0, t, 1)


@pytest.mark.parametrize("n", [12, 16, 20, 21, 22, 24])
def test_planted_optimum_is_found(n):
    """Sizes up to the n = 24 limit, on both sides of the block size."""
    problem, t = planted(n, seed=n)
    value, x = problem.brute_force()
    assert value == 0.0
    assert [int(v) for v in x] == t


@pytest.mark.parametrize("n", [20, 22, 24])
def test_even_ring_maxcut(n):
    """An even ring is two-colourable: every edge is cut, and of the two
    alternating colourings the one with x_0 = 1 has the lower number."""
    ring = [(i, (i + 1) % n, 1.0) for i in range(n)]
    value, x = maxcut_qubo(ring, n).brute_force()
    assert value == -float(n)
    assert [int(v) for v in x] == [1, 0] * (n // 2)
