"""QAOA circuit factory: structure, parameters, and what the circuit computes.

The structural tests count qubits, parameters and gates. The statevector
tests then check the physics: the built circuit must produce the same
measurement distribution as textbook QAOA written directly from the
problem's own cost function (tests/reference_impl.py), and its expected
cost can never fall below the certified optimum.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qopt_builder import QUBO, build_qaoa, maxcut_qubo
from reference_impl import all_costs, qaoa_probabilities
from run_demo import EDGES, N, expectation

PROBLEMS = {
    "demo-maxcut-6": maxcut_qubo(EDGES, N),
    "path-maxcut-5": maxcut_qubo([(i, i + 1, 1.0) for i in range(4)], 5),
    "weighted-triangle": maxcut_qubo([(0, 1, 0.5), (1, 2, 1.25), (0, 2, 3.0)], 3),
    "qubo-linear-terms-4": QUBO(
        [[-3.0, 2.0, 0.0, 1.0], [0.0, 1.5, -2.0, 0.0], [0.0, 0.0, -1.0, 4.0], [0.0, 0.0, 0.0, 2.0]],
        offset=1.25,
    ),
    "qubo-symmetric-5": QUBO(
        [
            [-2.0, 1.0, 0.5, 0.0, 1.5],
            [1.0, 3.0, -1.0, 2.0, 0.0],
            [0.5, -1.0, -1.0, 0.25, 1.0],
            [0.0, 2.0, 0.25, 1.0, -3.0],
            [1.5, 0.0, 1.0, -3.0, -0.5],
        ],
        offset=-0.75,
    ),
    "single-variable": QUBO([[-1.0]]),
}
NAMES = list(PROBLEMS)
DEPTHS = [1, 2, 3, 4]


def bind(qc, meta, gammas, betas):
    values = {meta["gammas"][k]: gammas[k] for k in range(meta["p"])}
    values.update({meta["betas"][k]: betas[k] for k in range(meta["p"])})
    return qc.assign_parameters(values)


def probabilities(problem, p, gammas, betas) -> np.ndarray:
    qc, meta = build_qaoa(problem, p)
    return Statevector(bind(qc, meta, gammas, betas)).probabilities()


def angles(seed: int, p: int):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, np.pi, size=p), rng.uniform(0, np.pi, size=p)


def ising_counts(problem):
    """(linear terms, couplings) of the Ising form, from the definition:
    h_i = -Q_ii / 2 - sum_{j != i} (Q_ij + Q_ji) / 4,  J_ij = (Q_ij + Q_ji) / 4."""
    q, n = problem.matrix, problem.n
    folded = q + q.T
    h = [-q[i, i] / 2 - sum(folded[i, j] for j in range(n) if j != i) / 4 for i in range(n)]
    linear = sum(abs(v) > 1e-12 for v in h)
    couplings = sum(abs(folded[i, j] / 4) > 1e-12 for i in range(n) for j in range(i + 1, n))
    return linear, couplings


# ---------------------------------------------------------------- structure

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("p", DEPTHS)
def test_qubit_and_parameter_counts(name, p):
    problem = PROBLEMS[name]
    qc, meta = build_qaoa(problem, p)
    assert qc.num_qubits == problem.n
    assert qc.num_parameters == 2 * p  # one gamma and one beta per layer (every problem here has terms)
    expected_names = {f"g{k}" for k in range(p)} | {f"b{k}" for k in range(p)}
    assert {parameter.name for parameter in qc.parameters} == expected_names
    assert meta["p"] == p
    assert len(meta["gammas"]) == len(meta["betas"]) == p


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("p", DEPTHS)
def test_gate_counts(name, p):
    """n Hadamards, then per layer one RZ per linear term, one RZZ per
    coupling and one RX per qubit. Nothing else."""
    problem = PROBLEMS[name]
    linear, couplings = ising_counts(problem)
    qc, _ = build_qaoa(problem, p)
    expected = {"h": problem.n, "rx": p * problem.n, "rz": p * linear, "rzz": p * couplings}
    assert dict(qc.count_ops()) == {gate: count for gate, count in expected.items() if count}


@pytest.mark.parametrize("p", DEPTHS)
def test_cost_function_without_terms(p):
    """A constant cost function has nothing for gamma to act on, so the
    circuit is the Hadamards and the mixer, with the p betas as its only
    parameters. Scoring it still works and returns the constant."""
    problem = QUBO(np.zeros((3, 3)), offset=2.5)
    qc, meta = build_qaoa(problem, p)
    assert dict(qc.count_ops()) == {"h": 3, "rx": 3 * p}
    assert {parameter.name for parameter in qc.parameters} == {f"b{k}" for k in range(p)}
    gammas, betas = angles(5, p)
    values = {meta["gammas"][k]: gammas[k] for k in range(p)}
    values.update({meta["betas"][k]: betas[k] for k in range(p)})
    assert expectation(qc, meta["h"], meta["J"], meta["offset"], values) == pytest.approx(2.5, abs=1e-12)


def test_maxcut_has_no_linear_terms():
    """Under x = (1 - z) / 2 the linear parts of MaxCut cancel, so the
    phase separator is ZZ rotations only: one per edge, per layer."""
    for name in ("demo-maxcut-6", "path-maxcut-5", "weighted-triangle"):
        qc, meta = build_qaoa(PROBLEMS[name], 2)
        assert not np.any(meta["h"])
        assert "rz" not in qc.count_ops()
    qc, _ = build_qaoa(PROBLEMS["demo-maxcut-6"], 2)
    assert qc.count_ops()["rzz"] == 2 * len(EDGES)


@pytest.mark.parametrize("name", NAMES)
def test_ising_form_reproduces_the_qubo_cost(name):
    """h, J and the constant returned with the circuit give back the
    QUBO cost on every assignment, with z_i = 1 - 2 x_i."""
    problem = PROBLEMS[name]
    n = problem.n
    _, meta = build_qaoa(problem, 1)
    h, J, constant = meta["h"], meta["J"], meta["offset"]
    assert h.shape == (n,) and J.shape == (n, n)
    assert not np.any(np.tril(J))  # couplings are stored once, above the diagonal
    reference = all_costs(problem.matrix.tolist(), problem.offset)
    for index, expected in enumerate(reference):
        z = np.array([1.0 - 2.0 * ((index >> i) & 1) for i in range(n)])
        assert constant + h @ z + z @ J @ z == pytest.approx(expected, abs=1e-9)


# -------------------------------------------------------------- statevector

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("p", [1, 2, 3])
def test_circuit_is_textbook_qaoa_for_the_cost_function(name, p):
    """Same measurement distribution as QAOA written from the definition."""
    problem = PROBLEMS[name]
    costs = all_costs(problem.matrix.tolist(), problem.offset)
    for seed in range(3):
        gammas, betas = angles(10 * p + seed, p)
        got = probabilities(problem, p, gammas, betas)
        want = qaoa_probabilities(costs, problem.n, gammas, betas)
        assert got.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_expectation_never_beats_the_certified_optimum(name):
    """The point of certifying: for any angles, optimum <= <cost> <= worst."""
    problem = PROBLEMS[name]
    costs = np.array(all_costs(problem.matrix.tolist(), problem.offset))
    optimum, _ = problem.brute_force()
    assert optimum == pytest.approx(costs.min(), abs=1e-9)
    for p in (1, 2, 3):
        for seed in range(5):
            gammas, betas = angles(100 * p + seed, p)
            expected_cost = float(probabilities(problem, p, gammas, betas) @ costs)
            assert optimum - 1e-9 <= expected_cost <= costs.max() + 1e-9


@pytest.mark.parametrize("name", NAMES)
def test_zero_angles_leave_the_uniform_superposition(name):
    """With every gamma zero, or every beta zero, all assignments stay
    equally likely and the expected cost is the plain average."""
    problem = PROBLEMS[name]
    size = 1 << problem.n
    gammas, betas = angles(7, 2)
    for g, b in ((np.zeros(2), betas), (gammas, np.zeros(2))):
        assert np.allclose(probabilities(problem, 2, g, b), 1.0 / size, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_demo_scoring_agrees_with_the_cost_function(name):
    """run_demo.expectation scores a state through the Ising form; it
    must equal the same state scored with the QUBO cost directly."""
    problem = PROBLEMS[name]
    costs = np.array(all_costs(problem.matrix.tolist(), problem.offset))
    for p in (1, 2, 3):
        qc, meta = build_qaoa(problem, p)
        gammas, betas = angles(1000 + p, p)
        values = {meta["gammas"][k]: gammas[k] for k in range(p)}
        values.update({meta["betas"][k]: betas[k] for k in range(p)})
        via_ising = expectation(qc, meta["h"], meta["J"], meta["offset"], values)
        direct = float(probabilities(problem, p, gammas, betas) @ costs)
        assert via_ising == pytest.approx(direct, abs=1e-9)
