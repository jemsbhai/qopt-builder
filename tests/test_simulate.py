"""Statevector scoring and the angle scan (qopt_builder/simulate.py)."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qopt_builder import PRESETS, QUBO, ScanResult, build_qaoa, expectation, scan_angles
from qopt_builder.simulate import SCAN_SAMPLES, SCAN_SEED, SIMULATION_LIMIT
from reference_impl import all_costs

SMALL = ["demo", "ring-10", "weighted-8", "vertex-cover-6", "dense-qubo-6"]


def test_defaults():
    assert (SCAN_SAMPLES, SCAN_SEED, SIMULATION_LIMIT) == (200, 20261110, 14)


@pytest.mark.parametrize("name", SMALL)
def test_expectation_is_the_probability_weighted_cost(name):
    """Scored with the package's cost vector, and again with costs
    recomputed from the definition and probabilities taken directly."""
    problem = PRESETS[name].build()
    reference = np.array(all_costs(problem.matrix.tolist(), problem.offset))
    rng = np.random.default_rng(3)
    for p in (1, 2, 3):
        qc, meta = build_qaoa(problem, p)
        values = {parameter: float(rng.uniform(0, np.pi)) for parameter in meta["gammas"] + meta["betas"]}
        probabilities = Statevector(qc.assign_parameters(values)).probabilities()
        assert expectation(qc, problem.all_costs(), values) == pytest.approx(
            float(probabilities @ reference), abs=1e-9
        )


def test_scan_result_shape():
    result = scan_angles(PRESETS["demo"].build(), 3)
    assert isinstance(result, ScanResult)
    assert result.trace.shape == (SCAN_SAMPLES,)
    assert result.gammas.shape == result.betas.shape == (3,)
    assert result.value == result.trace.min()


def test_scan_is_reproducible_and_seeded():
    problem = PRESETS["ring-10"].build()
    first, again = scan_angles(problem, 2), scan_angles(problem, 2)
    assert np.array_equal(first.trace, again.trace)
    other = scan_angles(problem, 2, seed=SCAN_SEED + 1)
    assert not np.array_equal(first.trace, other.trace)


def test_scan_draws_gammas_then_betas_for_each_sample():
    """The draw order fixes the angle stream, and with it the record."""
    problem = PRESETS["demo"].build()
    p, samples = 3, 25
    result = scan_angles(problem, p, samples=samples)
    rng = np.random.default_rng(SCAN_SEED)
    stream = [(rng.uniform(0, np.pi, size=p), rng.uniform(0, np.pi, size=p)) for _ in range(samples)]
    gammas, betas = stream[int(np.argmin(result.trace))]
    assert np.array_equal(result.gammas, gammas)
    assert np.array_equal(result.betas, betas)

    qc, meta = build_qaoa(problem, p)
    costs = problem.all_costs()
    for index in (0, 7, samples - 1):
        g, b = stream[index]
        values = {**dict(zip(meta["gammas"], g)), **dict(zip(meta["betas"], b))}
        assert result.trace[index] == expectation(qc, costs, values)


@pytest.mark.parametrize("name", SMALL)
def test_no_scanned_value_beats_the_certified_optimum(name):
    problem = PRESETS[name].build()
    optimum, _ = problem.brute_force()
    worst = float(problem.all_costs().max())
    trace = scan_angles(problem, 2, samples=40).trace
    assert np.all(trace >= optimum - 1e-9)
    assert np.all(trace <= worst + 1e-9)


def test_scan_of_a_cost_function_without_terms():
    """Nothing to optimise: every sample scores the constant."""
    result = scan_angles(QUBO(np.zeros((3, 3)), offset=2.5), 2, samples=10)
    assert np.allclose(result.trace, 2.5, rtol=0, atol=1e-12)
