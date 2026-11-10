"""Smoke tests pinning the demo record: the 6-node MaxCut graph from
run_demo.py must reproduce the published numbers exactly.

Run directly:  python tests/test_demo.py
Or under pytest if available.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qopt_builder import build_qaoa, maxcut_qubo, recommend  # noqa: E402

EDGES = [
    (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 2.0),
    (2, 4, 1.0), (3, 4, 1.0), (3, 5, 1.0), (4, 5, 2.0),
]
N = 6


def test_recommendation_is_exact_plus_qaoa_p2():
    rec = recommend(maxcut_qubo(EDGES, N))
    assert rec.algorithm == "exact+qaoa", rec.algorithm
    assert rec.params["p"] == 2, rec.params
    assert len(rec.rationale) == 2


def test_brute_force_optimum():
    problem = maxcut_qubo(EDGES, N)
    opt, x = problem.brute_force()
    assert abs(opt - (-8.0)) < 1e-9, opt
    assert list(x.astype(int)) == [0, 1, 0, 0, 1, 0]


def test_circuit_shape():
    problem = maxcut_qubo(EDGES, N)
    qc, _meta = build_qaoa(problem, 2)
    assert qc.num_qubits == 6
    assert qc.depth() == 14, qc.depth()
    assert qc.num_parameters == 4


def test_qubit_budget_refusal():
    rec = recommend(maxcut_qubo(EDGES, N), qubit_budget=4)
    assert rec.algorithm == "none", rec.algorithm


if __name__ == "__main__":
    test_recommendation_is_exact_plus_qaoa_p2()
    test_brute_force_optimum()
    test_circuit_shape()
    test_qubit_budget_refusal()
    print("4 passed")
