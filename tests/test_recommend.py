"""Recommendation engine: density bands, the exact-solve boundary, refusal.

Rules under test (qopt_builder/recommend.py):

  * n <= EXACT_CLASSICAL_LIMIT (24): exact solve offered, QAOA built alongside.
  * n >  EXACT_CLASSICAL_LIMIT: QAOA only.
  * n >  qubit_budget: refusal (algorithm "none"); no circuit is proposed.
  * QAOA depth from the coupling density d: d > 0.5 gives p=3, d > 0.15
    gives p=2, anything else p=1. Both comparisons are strict.
"""

from itertools import combinations

import numpy as np
import pytest

from qopt_builder import QUBO, maxcut_qubo, recommend
from qopt_builder.recommend import EXACT_CLASSICAL_LIMIT, Recommendation
from run_demo import EDGES, N


def graph(n: int, m: int) -> QUBO:
    """MaxCut on n nodes over the first m node pairs, unit weights.

    Its coupling density is exactly m / (n (n - 1) / 2).
    """
    pairs = list(combinations(range(n), 2))[:m]
    assert len(pairs) == m
    return maxcut_qubo([(i, j, 1.0) for i, j in pairs], n)


def path(n: int) -> QUBO:
    """MaxCut on a path with n nodes."""
    return maxcut_qubo([(i, i + 1, 1.0) for i in range(n - 1)], n)


def as_tuple(rec: Recommendation):
    return rec.algorithm, rec.params, rec.rationale


# --------------------------------------------------------------- density bands

BANDS = [
    # nodes, edges, density, expected p, band named in the rationale
    (16, 0, 0.0, 1, "sparse"),
    (16, 15, 0.125, 1, "sparse"),
    (16, 18, 0.15, 1, "sparse"),  # exactly on the lower threshold
    (16, 19, 19 / 120, 2, "moderate"),
    (6, 5, 5 / 15, 2, "moderate"),
    (4, 3, 0.5, 2, "moderate"),  # exactly on the upper threshold
    (5, 6, 0.6, 3, "high"),
    (6, 15, 1.0, 3, "high"),
]


@pytest.mark.parametrize("n, m, density, p, band", BANDS, ids=[f"{m}-edges-on-{n}-nodes" for n, m, *_ in BANDS])
def test_density_band_sets_qaoa_depth(n, m, density, p, band):
    problem = graph(n, m)
    assert problem.density() == pytest.approx(density, abs=1e-12)
    rec = recommend(problem)
    assert rec.params["p"] == p
    assert rec.rationale[-1].startswith(f"coupling density {density:.2f} is {band}")


def test_thresholds_are_strict():
    """A density sitting exactly on a threshold stays in the lower band."""
    on_lower, on_upper = graph(16, 18), graph(4, 3)
    assert on_lower.density() == 0.15
    assert on_upper.density() == 0.5
    assert recommend(on_lower).params["p"] == 1
    assert recommend(on_upper).params["p"] == 2
    # one more edge crosses each threshold
    assert recommend(graph(16, 19)).params["p"] == 2
    assert recommend(graph(4, 4)).params["p"] == 3


@pytest.mark.parametrize("n", [2, 3, 4, 6, 10])
def test_complete_graph_is_dense(n):
    """Regression for the density fix.

    Density used to count matrix entries, and an edge list fills one
    entry per edge, so every graph reported half its real density. A
    complete graph read 0.50 "moderate" and the high band could not be
    reached from the MaxCut path at all.
    """
    problem = maxcut_qubo([(i, j, 1.0) for i, j in combinations(range(n), 2)], n)
    assert problem.density() == 1.0
    assert recommend(problem).params["p"] == 3


def test_density_does_not_depend_on_matrix_layout():
    """One cost function, stored two ways, gets one recommendation."""
    upper = maxcut_qubo(EDGES, N)  # one matrix entry per edge
    symmetric = QUBO((upper.matrix + upper.matrix.T) / 2.0)  # two entries per edge
    assert not np.array_equal(upper.matrix, symmetric.matrix)
    for index in range(1 << N):
        x = [(index >> i) & 1 for i in range(N)]
        assert upper.cost(x) == pytest.approx(symmetric.cost(x), abs=1e-12)
    assert upper.density() == symmetric.density() == 8 / 15
    assert as_tuple(recommend(upper)) == as_tuple(recommend(symmetric))


def test_density_counts_pairs_not_weights():
    light = maxcut_qubo([(0, 1, 0.001), (1, 2, 0.001)], 4)
    heavy = maxcut_qubo([(0, 1, 1000.0), (1, 2, -7.5)], 4)
    assert light.density() == heavy.density() == 2 / 6


def test_density_without_couplings_is_zero():
    assert QUBO(np.diag([1.0, -2.0, 3.0])).density() == 0.0
    assert QUBO([[-1.0]]).density() == 0.0  # a single variable has no pairs


# ------------------------------------------------- exact-solve boundary (n = 24)

def test_exact_limit_is_24():
    assert EXACT_CLASSICAL_LIMIT == 24


@pytest.mark.parametrize("n", [1, 2, 6, 23, 24])
def test_exact_solve_offered_up_to_the_limit(n):
    rec = recommend(path(n))
    assert rec.algorithm == "exact+qaoa"
    assert rec.params["also_build"] == "qaoa"
    assert rec.rationale[0].startswith(f"n={n} <= 24: an exact classical solve is cheap")


@pytest.mark.parametrize("n", [25, 26, 40, 100])
def test_exact_solve_not_offered_beyond_the_limit(n):
    rec = recommend(path(n))
    assert rec.algorithm == "qaoa"
    assert set(rec.params) == {"p"}
    assert rec.rationale[0] == f"n={n}: exact enumeration is not feasible (2^{n} states)."


@pytest.mark.parametrize("m, p", [(30, 1), (100, 2), (200, 3)])
def test_depth_rule_still_applies_beyond_the_limit(m, p):
    rec = recommend(graph(25, m))  # 300 pairs: densities 0.10, 0.33, 0.67
    assert rec.algorithm == "qaoa"
    assert rec.params == {"p": p}


# ------------------------------------------------------------ budget refusal

@pytest.mark.parametrize("n, budget", [(6, 5), (6, 1), (24, 23), (25, 24), (40, 27)])
def test_refuses_when_the_problem_exceeds_the_qubit_budget(n, budget):
    rec = recommend(path(n), qubit_budget=budget)
    assert rec.algorithm == "none"
    assert rec.params == {"n": n}  # no depth, nothing to build
    assert len(rec.rationale) == 2
    assert rec.rationale[-1].startswith(f"n={n} exceeds the qubit budget of {budget}")
    assert "refuse rather than silently truncate" in rec.rationale[-1]


@pytest.mark.parametrize("n, budget", [(6, 6), (6, 7), (24, 24), (25, 25), (25, 127)])
def test_budget_at_or_above_n_changes_nothing(n, budget):
    """The refusal test is n > budget, so a budget equal to n still fits."""
    with_budget = recommend(path(n), qubit_budget=budget)
    assert with_budget.algorithm != "none"
    assert as_tuple(with_budget) == as_tuple(recommend(path(n)))


def test_refusal_does_not_depend_on_density():
    for problem in (graph(8, 0), graph(8, 10), graph(8, 28)):
        assert recommend(problem, qubit_budget=7).algorithm == "none"


def test_refusal_beyond_the_exact_limit_gives_both_reasons():
    rec = recommend(path(30), qubit_budget=20)
    assert rec.algorithm == "none"
    assert rec.rationale[0].startswith("n=30: exact enumeration is not feasible")
    assert rec.rationale[1].startswith("n=30 exceeds the qubit budget of 20")


# ------------------------------------------------------------------ shape

@pytest.mark.parametrize("problem", [graph(6, 5), graph(25, 100)], ids=["n=6", "n=25"])
def test_every_recommendation_states_its_reasons(problem):
    rec = recommend(problem)
    assert isinstance(rec, Recommendation)
    assert rec.algorithm in {"exact+qaoa", "qaoa"}
    assert len(rec.rationale) == 2  # one reason for the algorithm, one for the depth
    assert all(isinstance(line, str) and line for line in rec.rationale)
