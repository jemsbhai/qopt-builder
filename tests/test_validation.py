"""Input checks in the Python API: what is turned away, and in what words.

A problem that cannot be right is rejected where it enters, with a
ValueError that says what is wrong and where. The wording is pinned
here. tests/test_web_parity.py gives the web port the same inputs
(tests/validation_cases.py) and requires the same words from it.
"""

from fractions import Fraction

import numpy as np
import pytest

import run_demo
from qopt_builder import QUBO, build_qaoa, expectation, maxcut_qubo, recommend, scan_angles
from qopt_builder.presets import DEMO_EDGES
from qopt_builder.problems import SYMMETRY_TOLERANCE, ZERO_TOLERANCE
from validation_cases import (
    ACCEPTED, DEPTH, EMPTY, NAN, INF, NODE_COUNT, NOT_A_LIST, NOT_A_TABLE, NOT_AN_ASSIGNMENT, OFFSET,
    REJECTED, SAMPLES, edge_nodes, edge_range, edge_shape, edge_weight, neither, not_a_number,
    not_binary, not_finite, not_square, run_python, wrong_length,
)

# ------------------------------------------------- the shared, JSON-able cases


@pytest.mark.parametrize("name", list(REJECTED))
def test_bad_input_is_rejected_with_its_message(name):
    spec, message = REJECTED[name]
    assert run_python(spec) == ("error", message)


@pytest.mark.parametrize("name", list(ACCEPTED))
def test_good_input_is_accepted(name):
    spec, result = ACCEPTED[name]
    assert run_python(spec) == ("ok", result)


def test_every_message_is_a_full_sentence():
    """Each one starts with a capital or with the name of the argument,
    ends with a full stop, and stays on one line."""
    for _, message in REJECTED.values():
        assert message[0].isupper() or message.startswith("edges["), message
        assert message.endswith(".") and "\n" not in message, message


# ------------------------------------------------- inputs that only Python has

ARRAYS = [
    pytest.param(np.array(3.0), NOT_A_TABLE, id="0-d-array"),
    pytest.param(np.zeros(4), NOT_A_TABLE, id="1-d-array"),
    pytest.param(np.zeros((0, 0)), EMPTY, id="0x0-array"),
    pytest.param(np.zeros((0, 3)), EMPTY, id="0x3-array"),
    pytest.param(np.zeros((3, 0)), EMPTY, id="3x0-array"),
    pytest.param(np.zeros((2, 3)), not_square(2, 0, 3), id="2x3-array"),
    pytest.param(np.zeros((3, 2), dtype=int), not_square(3, 0, 2), id="3x2-integer-array"),
    pytest.param(np.zeros((2, 2, 2)), not_a_number(0, 0), id="3-d-array"),
    pytest.param(np.eye(2, dtype=bool), not_a_number(0, 0), id="boolean-array"),
    pytest.param(np.array([[0, 1j], [1j, 0]]), not_a_number(0, 0), id="complex-array"),
    pytest.param(np.array([[0, "a"], ["a", 0]], dtype=object), not_a_number(0, 1), id="object-array"),
    pytest.param(np.array([["0", "1"], ["1", "0"]]), not_a_number(0, 0), id="text-array"),
    pytest.param(np.array([[0.0, 1.0], [NAN, 0.0]]), not_finite(1, 0), id="nan-in-an-array"),
    pytest.param(np.array([[0.0, INF], [0.0, 0.0]]), not_finite(0, 1), id="infinity-in-an-array"),
    pytest.param(np.array([[0.0, 1.0], [3.0, 0.0]]), neither(0, 1), id="conflicting-pair-in-an-array"),
    pytest.param([[0, 1j], [1j, 0]], not_a_number(0, 1), id="complex-entry"),
    pytest.param(((0, 1), (3, 0)), neither(0, 1), id="tuples"),
]


@pytest.mark.parametrize("matrix, message", ARRAYS)
def test_arrays_get_the_same_checks_as_lists(matrix, message):
    with pytest.raises(ValueError) as caught:
        QUBO(matrix)
    assert str(caught.value) == message


@pytest.mark.parametrize("matrix", [
    np.array([[1, 2], [2, -1]]),                      # integers
    np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32),
    ((1, 2), (2, -1)),                                # tuples
    [np.array([1, 2]), np.array([2, -1])],            # a list of array rows
    [[np.int64(1), np.float64(2.0)], [Fraction(4, 2), -1]],  # numpy and exact numbers as entries
], ids=["integer-array", "float32-array", "tuples", "list-of-arrays", "mixed-number-types"])
def test_every_kind_of_number_table_becomes_a_float_matrix(matrix):
    problem = QUBO(matrix)
    assert problem.matrix.dtype == np.float64
    assert problem.matrix.shape == (2, 2)
    assert problem.matrix[0, 0] == 1.0 and problem.matrix[1, 1] == -1.0


def test_the_matrix_is_copied():
    """Changing the caller's array afterwards does not change the problem."""
    given = np.array([[1.0, 2.0], [0.0, 3.0]])
    problem = QUBO(given)
    given[0, 0] = 99.0
    assert problem.matrix[0, 0] == 1.0
    assert problem.matrix is not given


@pytest.mark.parametrize("offset", [2, -2.5, np.float64(1.5), np.int64(3), Fraction(1, 2)])
def test_any_finite_number_is_an_offset(offset):
    problem = QUBO([[1.0]], offset=offset)
    assert isinstance(problem.offset, float) and problem.offset == float(offset)


@pytest.mark.parametrize("offset", [1j, np.bool_(True), np.array([1.0]), "0", NAN, -INF])
def test_anything_else_is_not_an_offset(offset):
    with pytest.raises(ValueError) as caught:
        QUBO([[1.0]], offset=offset)
    assert str(caught.value) == OFFSET


# ------------------------------------------------------- the layout rule itself

def test_the_two_tolerances_are_the_documented_ones():
    assert SYMMETRY_TOLERANCE == 1e-9
    assert ZERO_TOLERANCE == 1e-12


def test_symmetry_is_judged_relative_to_the_size_of_the_entries():
    """A difference of one part in 10^10 passes at any scale; one part in
    10^8 does not."""
    for scale in (1e-3, 1.0, 1e6, 1e12):
        QUBO([[0, scale], [scale * (1 + 1e-10), 0]])
    for scale in (1.0, 1e6, 1e12):
        with pytest.raises(ValueError, match="symmetric or triangular"):
            QUBO([[0, scale], [scale * (1 + 1e-8), 0]])


def test_small_entries_are_compared_against_one():
    """Below 1 the tolerance is absolute: 1e-10 apart is equal, 1e-8 is not."""
    QUBO([[0, 0.001], [0.001 + 1e-10, 0]])
    with pytest.raises(ValueError, match="symmetric or triangular"):
        QUBO([[0, 0.001], [0.001 + 1e-8, 0]])


def test_the_first_differing_pair_is_named_in_reading_order():
    matrix = np.zeros((4, 4))
    matrix[0, 1] = matrix[1, 0] = 2.0   # a symmetric pair
    matrix[1, 3], matrix[3, 1] = 1.0, 5.0   # differs
    matrix[2, 3], matrix[3, 2] = 4.0, 6.0   # differs too, but later
    with pytest.raises(ValueError) as caught:
        QUBO(matrix)
    assert str(caught.value) == neither(1, 3)


@pytest.mark.parametrize("seed", range(5))
def test_a_lower_triangular_matrix_is_the_same_problem_as_its_transpose(seed):
    """Lower-triangular input was always computed correctly; it is now
    also stated to be allowed. Same costs, density, recommendation,
    optimum and circuit as the upper-triangular form."""
    rng = np.random.default_rng(seed)
    n = 5
    upper = np.triu(rng.integers(-3, 4, size=(n, n)).astype(float))
    a, b = QUBO(upper, offset=1.0), QUBO(upper.T, offset=1.0)
    assert np.array_equal(a.all_costs(), b.all_costs())
    assert a.density() == b.density()
    assert recommend(a).rationale == recommend(b).rationale
    (value_a, x_a), (value_b, x_b) = a.brute_force(), b.brute_force()
    assert value_a == value_b and np.array_equal(x_a, x_b)
    qc_a, qc_b = build_qaoa(a, 2)[0], build_qaoa(b, 2)[0]
    assert (qc_a.depth(), dict(qc_a.count_ops())) == (qc_b.depth(), dict(qc_b.count_ops()))


# ------------------------------------------------------------------ assignments

def test_arrays_tuples_and_booleans_are_assignments():
    problem = maxcut_qubo(DEMO_EDGES, 6)
    best = [0, 1, 0, 0, 1, 0]
    for x in (best, tuple(best), np.array(best), np.array(best, dtype=float), [bool(v) for v in best],
              np.array(best, dtype=bool)):
        assert problem.cost(x) == -8.0


@pytest.mark.parametrize("x, message", [
    (np.zeros((6, 1)), NOT_AN_ASSIGNMENT),
    (np.array(1.0), NOT_AN_ASSIGNMENT),
    (b"010010", NOT_AN_ASSIGNMENT),
    (np.zeros(5), wrong_length(6, 5)),
    (np.array([0, 1, 0, 0, 1, 2]), not_binary(5)),
    ([0, 1, 0, 1j, 1, 0], not_binary(3)),
], ids=["2-d-array", "0-d-array", "bytes", "short-array", "array-with-a-two", "complex-entry"])
def test_arrays_that_are_not_assignments(x, message):
    with pytest.raises(ValueError) as caught:
        maxcut_qubo(DEMO_EDGES, 6).cost(x)
    assert str(caught.value) == message


# ------------------------------------------------------------------------ edges

def test_edges_may_be_tuples_arrays_or_a_generator():
    expected = maxcut_qubo([[0, 1, 2.0], [1, 2, 3.0]], 3).matrix
    for edges in (
        [(0, 1, 2.0), (1, 2, 3.0)],
        ((0, 1, 2.0), (1, 2, 3.0)),
        np.array([[0, 1, 2.0], [1, 2, 3.0]]),
        ((i, i + 1, float(i + 2)) for i in range(2)),
        [(np.int64(0), np.int32(1), np.float32(2.0)), (1, 2, Fraction(3))],
    ):
        assert np.array_equal(maxcut_qubo(edges, 3).matrix, expected)


def test_the_node_count_may_be_any_whole_number_type():
    for n in (3, 3.0, np.int64(3), Fraction(3)):
        assert maxcut_qubo([(0, 2)], n).n == 3


@pytest.mark.parametrize("n", [1j, np.bool_(True), [3], -(10 ** 400)])
def test_node_counts_only_python_can_write(n):
    with pytest.raises(ValueError) as caught:
        maxcut_qubo([], n)
    assert str(caught.value) == NODE_COUNT


@pytest.mark.parametrize("edges, message", [
    (object(), NOT_A_LIST),
    (b"01", NOT_A_LIST),
    ([{0, 1}], edge_shape(0)),
    ([(0, 1j)], edge_nodes(0)),
    ([(0, 10 ** 30)], edge_range(0, 10 ** 30, 3)),
    ([(0, 1, 1j)], edge_weight(0)),
    ([(0, 1, np.array([1.0]))], edge_weight(0)),
], ids=["an-object", "bytes", "edge-is-a-set", "complex-node", "huge-node", "complex-weight", "array-weight"])
def test_edges_only_python_can_write(edges, message):
    with pytest.raises(ValueError) as caught:
        maxcut_qubo(edges, 3)
    assert str(caught.value) == message


@pytest.mark.parametrize("seed", range(4))
def test_edge_direction_and_order_do_not_change_the_problem(seed):
    """Reversing edges or shuffling the list gives the same matrix,
    always upper-triangular, which QUBO() then accepts."""
    rng = np.random.default_rng(seed)
    n = 7
    edges = [(i, j, float(rng.integers(1, 5))) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
    expected = maxcut_qubo(edges, n).matrix
    flipped = [(j, i, w) if rng.random() < 0.5 else (i, j, w) for i, j, w in edges]
    order = rng.permutation(len(flipped))
    got = maxcut_qubo([flipped[k] for k in order], n).matrix
    assert np.array_equal(got, expected)
    assert not np.tril(got, -1).any()


def test_the_demo_edges_still_build_the_demo_matrix():
    """The checks changed nothing for valid input: the demo problem is
    entry for entry what it was."""
    q = np.zeros((6, 6))
    for i, j, w in DEMO_EDGES:
        q[i, i] -= w
        q[j, j] -= w
        q[i, j] += 2.0 * w
    assert np.array_equal(maxcut_qubo(DEMO_EDGES, 6).matrix, q)


# ------------------------------------------------------- depth, samples, costs

@pytest.mark.parametrize("p", [1, 2.0, np.int64(3), Fraction(4)])
def test_any_positive_whole_number_is_a_depth(p):
    qc, meta = build_qaoa(maxcut_qubo(DEMO_EDGES, 6), p)
    assert meta["p"] == int(p) and type(meta["p"]) is int
    assert qc.num_parameters == 2 * int(p)


@pytest.mark.parametrize("p", [1j, [2], np.bool_(True), -INF])
def test_depths_only_python_can_write(p):
    with pytest.raises(ValueError) as caught:
        build_qaoa(maxcut_qubo(DEMO_EDGES, 6), p)
    assert str(caught.value) == DEPTH


def test_scan_checks_its_depth_and_its_sample_count():
    problem = maxcut_qubo([(0, 1)], 2)
    assert len(scan_angles(problem, 1, samples=np.int64(4)).trace) == 4
    with pytest.raises(ValueError) as caught:
        scan_angles(problem, 1, samples=[10])
    assert str(caught.value) == SAMPLES
    with pytest.raises(ValueError) as caught:
        scan_angles(problem, 2.5)
    assert str(caught.value) == DEPTH


def test_expectation_needs_one_cost_per_assignment():
    problem = maxcut_qubo(DEMO_EDGES, 6)
    qc, meta = build_qaoa(problem, 1)
    values = {meta["gammas"][0]: 0.4, meta["betas"][0]: 0.3}
    costs = problem.all_costs()
    assert expectation(qc, list(costs), values) == expectation(qc, costs, values)  # a list is fine
    with pytest.raises(ValueError) as caught:
        expectation(qc, costs[:32], values)
    assert str(caught.value) == (
        "costs must hold one cost per assignment: 64 for a circuit on 6 qubits, but it has shape (32,)."
    )


# ------------------------------------------------------------- the command line

@pytest.mark.parametrize("value, reason", [
    ("-1", "a qubit budget cannot be negative"),
    ("2.5", "'2.5' is not a whole number"),
    ("many", "'many' is not a whole number"),
])
def test_budget_flag_takes_a_whole_number_of_qubits(capsys, value, reason):
    with pytest.raises(SystemExit) as exit_info:
        run_demo.main(["--budget", value])
    assert exit_info.value.code == 2
    assert f"argument --budget: {reason}" in capsys.readouterr().err


def test_budget_of_zero_refuses_everything(capsys):
    run_demo.main(["--budget", "0"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Recommendation: none {'n': 6}"
    assert lines[-1] == "No circuit built."


# ------------------------------------------- what the checks deliberately leave

def test_recommend_is_untouched_by_the_checks():
    """The recommendation engine takes its budget as before: any number
    compares against n. Its rules were not part of this change."""
    problem = maxcut_qubo(DEMO_EDGES, 6)
    assert recommend(problem, 6).algorithm == "exact+qaoa"
    assert recommend(problem, 5).algorithm == "none"
    assert recommend(problem, 5.5).algorithm == "none"
    assert recommend(problem, -1).algorithm == "none"
    assert recommend(problem, None).algorithm == "exact+qaoa"
