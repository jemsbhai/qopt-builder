"""Smoke tests pinning the demo record: the 6-node MaxCut graph from
run_demo.py must reproduce the published numbers exactly.

Run directly:  python tests/test_demo.py
Or under pytest if available.
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from qopt_builder import build_qaoa, maxcut_qubo, recommend  # noqa: E402

EDGES = [
    (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 2.0),
    (2, 4, 1.0), (3, 4, 1.0), (3, 5, 1.0), (4, 5, 2.0),
]
N = 6

# What `python run_demo.py` prints, line for line.
DEMO_OUTPUT = """\
Recommendation: exact+qaoa {'p': 3, 'also_build': 'qaoa'}
 - n=6 <= 24: an exact classical solve is cheap and certifies the quantum result; recommending exact with QAOA built alongside for comparison.
 - coupling density 0.53 is high: deeper QAOA (p=3) earns its depth on dense graphs.
Brute-force optimum: -8.000 at x=[0 1 0 0 1 0]
QAOA circuit: 6 qubits, depth 19, params 6
Best scanned expectation: -7.308 (optimum -8.000)
"""


def test_recommendation_is_exact_plus_qaoa_p3():
    rec = recommend(maxcut_qubo(EDGES, N))
    assert rec.algorithm == "exact+qaoa", rec.algorithm
    assert rec.params["p"] == 3, rec.params
    assert len(rec.rationale) == 2


def test_coupling_density():
    # 8 of the 15 node pairs are joined by an edge
    assert maxcut_qubo(EDGES, N).density() == 8 / 15


def test_brute_force_optimum():
    problem = maxcut_qubo(EDGES, N)
    opt, x = problem.brute_force()
    assert abs(opt - (-8.0)) < 1e-9, opt
    assert list(x.astype(int)) == [0, 1, 0, 0, 1, 0]


def test_circuit_shape():
    problem = maxcut_qubo(EDGES, N)
    qc, _meta = build_qaoa(problem, 3)
    assert qc.num_qubits == 6
    assert qc.depth() == 19, qc.depth()
    assert qc.num_parameters == 6


def test_circuit_depth_per_layer():
    # 4 + 5p on this graph; p=2 gives the depth 14 of the earlier record
    problem = maxcut_qubo(EDGES, N)
    depths = [build_qaoa(problem, p)[0].depth() for p in (1, 2, 3, 4)]
    assert depths == [9, 14, 19, 24], depths


def test_qubit_budget_refusal():
    rec = recommend(maxcut_qubo(EDGES, N), qubit_budget=4)
    assert rec.algorithm == "none", rec.algorithm


def test_run_demo_prints_the_record():
    """End to end: the script's output, including the scanned expectation."""
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_demo.py")],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == DEMO_OUTPUT.splitlines(), proc.stdout


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
    print(f"{len(tests)} passed")
