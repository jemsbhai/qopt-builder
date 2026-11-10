"""End-to-end demo: MaxCut on a 6-node graph.

Pipeline: problem -> features -> recommendation -> QAOA circuit ->
a coarse angle scan scored by exact statevector expectation, checked
against the brute-force optimum.
"""

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector

from qopt_builder import build_qaoa, maxcut_qubo, recommend

EDGES = [
    (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 2.0),
    (2, 4, 1.0), (3, 4, 1.0), (3, 5, 1.0), (4, 5, 2.0),
]
N = 6


def expectation(qc, h, J, offset, params) -> float:
    bound = qc.assign_parameters(params)
    sv = Statevector.from_instruction(transpile(bound, optimization_level=0))
    probs = sv.probabilities()
    total = 0.0
    for idx, pr in enumerate(probs):
        if pr < 1e-12:
            continue
        z = np.array([1.0 if ((idx >> i) & 1) == 0 else -1.0 for i in range(len(h))])
        e = offset + float(h @ z) + float(z @ J @ z)
        total += pr * e
    return total


if __name__ == "__main__":
    problem = maxcut_qubo(EDGES, N)
    rec = recommend(problem)
    print("Recommendation:", rec.algorithm, rec.params)
    for r in rec.rationale:
        print(" -", r)
    opt, x = problem.brute_force()
    print(f"Brute-force optimum: {opt:.3f} at x={x.astype(int)}")

    p = rec.params.get("p", 1)
    qc, meta = build_qaoa(problem, p)
    print(f"QAOA circuit: {qc.num_qubits} qubits, depth {qc.depth()}, params {qc.num_parameters}")
    rng = np.random.default_rng(20261110)
    best = (np.inf, None)
    for _ in range(200):
        g = rng.uniform(0, np.pi, size=p)
        b = rng.uniform(0, np.pi, size=p)
        params = {**{meta["gammas"][k]: g[k] for k in range(p)},
                  **{meta["betas"][k]: b[k] for k in range(p)}}
        e = expectation(qc, meta["h"], meta["J"], meta["offset"], params)
        if e < best[0]:
            best = (e, (g, b))
    print(f"Best scanned expectation: {best[0]:.3f} (optimum {opt:.3f})")
