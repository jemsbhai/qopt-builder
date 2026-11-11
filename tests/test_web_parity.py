"""Web port parity: the JavaScript in web/ against the Python package.

The browser demo is a port of the Python pipeline. These tests run the
port's own source files (web/qubo.js, web/qaoa.js, web/circuit.js,
web/presets.js) under Node.js, through tests/web_parity.js, on the same
problems the Python package is given, and compare stage by stage: the
preset list, coupling density, the recommendation with its rationale
text, the exact solve, the Ising form, the circuit structure, and
statevector expectations at fixed angles. Bad input is part of the
comparison: both ports get the inputs in tests/validation_cases.py and
must turn each away in the same words. The page's own text parsers,
which have no Python counterpart, are pinned here as well.

The angle scan is the one stage that is not shared: the browser draws
its angles from its own seeded generator. For that stage the test
regenerates the browser's angle stream and checks that Qiskit, fed the
same angles, reproduces every scanned value.

Skipped when Node.js is not installed.
"""

import json
import math
import os
import shutil
import subprocess
from itertools import combinations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qopt_builder import PRESETS, QUBO, build_qaoa, expectation, maxcut_qubo, recommend
from qopt_builder.presets import DEMO_EDGES
from qopt_builder.problems import SYMMETRY_TOLERANCE, TIE_TOLERANCE, ZERO_TOLERANCE
from qopt_builder.recommend import EXACT_CLASSICAL_LIMIT
from qopt_builder.simulate import SCAN_SAMPLES, SCAN_SEED, SIMULATION_LIMIT
from validation_cases import ACCEPTED, REJECTED, neither, to_json

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None, reason="Node.js is not installed; this check runs web/*.js under Node"
)


# ------------------------------------------------------------------ problems

def random_qubo(seed: int, n: int, layout: str, integer: bool) -> QUBO:
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
    return QUBO(matrix, offset=offset)


def random_graph(seed: int, n: int, keep: float = 0.5) -> QUBO:
    rng = np.random.default_rng(seed)
    edges = [
        (i, j, float(rng.integers(1, 6)))
        for i, j in combinations(range(n), 2)
        if rng.random() < keep
    ]
    return maxcut_qubo(edges, n)


def first_pairs(n: int, m: int) -> QUBO:
    pairs = list(combinations(range(n), 2))[:m]
    return maxcut_qubo([(i, j, 1.0) for i, j in pairs], n)


def angle_sets(seed: int, p: int, count: int = 3):
    rng = np.random.default_rng(seed)
    return [
        (rng.uniform(0, np.pi, size=p).tolist(), rng.uniform(0, np.pi, size=p).tolist())
        for _ in range(count)
    ]


def make_case(name, problem, *, solve=True, simulate=True, depths=(1, 2, 3), budget=None, scan=False):
    """One problem plus what to compute on it. Simulation is kept to
    small n; the structural checks run at any size."""
    angles = {}
    if simulate:
        angles = {p: angle_sets(1000 * problem.n + p, p) for p in depths}
    return {
        "name": name, "problem": problem, "solve": solve, "budget": budget,
        "depths": list(depths), "angles": angles, "scan": scan,
    }


def build_cases():
    # every preset: solved exactly in both ports (the 24-node one included),
    # simulated where the page simulates, and the demo scanned as the page scans it
    cases = [
        make_case(
            f"preset-{name}", preset.build(), scan=(name == "demo"),
            simulate=preset.n <= SIMULATION_LIMIT,
            depths=(1, 2, 3) if preset.n <= SIMULATION_LIMIT else (1, 2),
        )
        for name, preset in PRESETS.items()
    ]

    for n in range(1, 9):
        for layout in ("upper", "symmetric"):
            for integer in (True, False):
                kind = "int" if integer else "dec"
                cases.append(make_case(f"qubo-n{n}-{layout}-{kind}", random_qubo(70 * n + integer, n, layout, integer)))
    for n in range(2, 11):
        cases.append(make_case(f"maxcut-n{n}", random_graph(300 + n, n)))

    # exact solve only, sizes where the two solvers take different routes:
    # Gray code in the browser, more than one block in Python above n = 20
    for n in (16, 20, 21, 22):
        cases.append(make_case(f"exact-n{n}", random_qubo(900 + n, n, "upper", True), simulate=False, depths=(1,)))

    # ties: every assignment optimal, and a tie that only rounding breaks
    # (x_0 alone costs -0.3; x_1 with x_2 costs -0.1 - 0.2)
    cases.append(make_case("tie-all-zero", QUBO(np.zeros((4, 4)), offset=1.5), depths=(1, 2)))
    rounding = np.diag([-0.3, -0.1, -0.2])
    rounding[0, 1] = rounding[0, 2] = 1.0
    cases.append(make_case("tie-rounding", QUBO(rounding)))

    # recommendation and circuit structure beyond what is simulated
    for n, m in ((16, 15), (17, 17), (16, 18), (16, 19), (16, 75), (4, 3), (4, 4), (12, 66)):
        cases.append(make_case(f"density-n{n}-m{m}", first_pairs(n, m), solve=False, simulate=False, depths=(1, 2)))
    for n in (24, 25, 40):
        cases.append(make_case(f"limit-n{n}", random_graph(500 + n, n, keep=0.2), solve=False, simulate=False, depths=(1,)))
    for n, budget in ((6, 5), (6, 6), (25, 24), (30, 20), (12, 127)):
        cases.append(make_case(f"budget-n{n}-b{budget}", random_graph(600 + n, n), solve=False, simulate=False,
                               depths=(), budget=budget))
    return cases


CASES = build_cases()
BY_NAME = {case["name"]: case for case in CASES}
assert len(BY_NAME) == len(CASES)


def names(predicate=lambda case: True):
    return [case["name"] for case in CASES if predicate(case)]


# ------------------------------------------------- text typed into the page

PAGE_LIMIT = 64  # MAX_VARIABLES in web/app.js

NO_EDGES = "No edges given. Enter one edge per line: two node numbers and an optional weight."
NO_MATRIX = "No matrix given. Enter one row per line, numbers separated by spaces or commas."


def edge_form(line: int) -> str:
    return f"Line {line}: an edge is two node numbers and an optional weight, for example 0 1 2.5."


def node_form(line: int) -> str:
    return f"Line {line}: node numbers must be whole numbers, 0 or more."


def not_a_number_text(where: str, token: str) -> str:
    return f'{where}: "{token}" is not a number.'


def too_large(where: str, token: str) -> str:
    return f'{where}: "{token}" is too large to use.'


def too_many(n: int, limit: int = PAGE_LIMIT) -> str:
    return (f"This page handles problems of up to {limit} variables, and this one has {n}. "
            "The Python package has no such limit.")


def row_length(line: int, numbers: str, rows: str) -> str:
    return (f"Line {line} has {numbers}, but the matrix has {rows}. A QUBO matrix is square: "
            "every row needs as many numbers as there are rows.")


def not_a_budget(token: str) -> str:
    return f'Qubit budget: "{token}" is not a whole number, 0 or more.'


def ok(value):
    return {"ok": value}


def error(message):
    return {"error": message}


def text(kind, typed, expected, **more):
    return {"kind": kind, "text": typed, **more}, expected


# what is typed -> what the page's parsers make of it
TEXTS = {
    # edge lists
    "edges-plain": text("edges", "0 1\n1 2 2.5\n", ok({"edges": [[0, 1, 1], [1, 2, 2.5]], "n": 3})),
    "edges-commas-blank-lines-crlf": text(
        "edges", "0, 1, 2\r\n\r\n   2,0 , -1  \r\n", ok({"edges": [[0, 1, 2], [2, 0, -1]], "n": 3})),
    "edges-tabs-and-signs": text("edges", "0\t3\t+1.5e1\n3 1 -.5", ok({"edges": [[0, 3, 15], [3, 1, -0.5]], "n": 4})),
    "edges-repeated-and-reversed": text("edges", "0 1\n1 0 2", ok({"edges": [[0, 1, 1], [1, 0, 2]], "n": 2})),
    "edges-empty": text("edges", "", error(NO_EDGES)),
    "edges-only-blank-lines": text("edges", "  \n\n\t\n", error(NO_EDGES)),
    "edges-one-number": text("edges", "0 1\n2", error(edge_form(2))),
    "edges-four-numbers": text("edges", "0 1 2 3", error(edge_form(1))),
    "edges-a-word": text("edges", "0 1\n\n1 x", error(not_a_number_text("Line 3", "x"))),
    "edges-hex": text("edges", "0 0x1", error(not_a_number_text("Line 1", "0x1"))),
    "edges-infinity": text("edges", "0 1 Infinity", error(not_a_number_text("Line 1", "Infinity"))),
    "edges-nan": text("edges", "0 1 NaN", error(not_a_number_text("Line 1", "NaN"))),
    "edges-two-points": text("edges", "0 1 1.2.3", error(not_a_number_text("Line 1", "1.2.3"))),
    "edges-overflow": text("edges", "0 1 1e999", error(too_large("Line 1", "1e999"))),
    "edges-fractional-node": text("edges", "0 1.5", error(node_form(1))),
    "edges-negative-node": text("edges", "0 1\n-1 2", error(node_form(2))),
    "edges-self-loop": text("edges", "0 1\n3 3 2", error("Line 2: an edge cannot join node 3 to itself.")),
    "edges-at-the-page-limit": text("edges", "0 63", ok({"edges": [[0, 63, 1]], "n": 64}), limit=PAGE_LIMIT),
    "edges-past-the-page-limit": text("edges", "0 64", error(too_many(65)), limit=PAGE_LIMIT),
    "edges-far-past-the-page-limit": text("edges", "0 1\n5 100000", error(too_many(100001)), limit=PAGE_LIMIT),
    "edges-no-limit-given": text("edges", "0 200", ok({"edges": [[0, 200, 1]], "n": 201})),
    # matrices
    "matrix-plain": text("matrix", "0 1\n1 0", ok([[0, 1], [1, 0]])),
    "matrix-commas-blank-lines": text("matrix", "-1, 2.5\r\n\r\n0,3\n", ok([[-1, 2.5], [0, 3]])),
    "matrix-one-entry": text("matrix", " 7 ", ok([[7]])),
    "matrix-empty": text("matrix", "", error(NO_MATRIX)),
    "matrix-only-blank-lines": text("matrix", "\n \n", error(NO_MATRIX)),
    "matrix-short-row": text("matrix", "1 2\n\n3", error(row_length(3, "1 number", "2 rows"))),
    "matrix-long-rows": text("matrix", "1 2 3\n4 5 6", error(row_length(1, "3 numbers", "2 rows"))),
    "matrix-one-long-row": text("matrix", "1 2", error(row_length(1, "2 numbers", "1 row"))),
    "matrix-a-word": text("matrix", "1 2\n3 four", error(not_a_number_text("Line 2", "four"))),
    "matrix-overflow": text("matrix", "1e400", error(too_large("Line 1", "1e400"))),
    "matrix-past-the-page-limit": text("matrix", "1 0 0\n0 1 0\n0 0 1", error(too_many(3, 2)), limit=2),
    # the offset box
    "offset-empty": text("offset", "", ok(0)),
    "offset-blank": text("offset", "   ", ok(0)),
    "offset-number": text("offset", " -2.5 ", ok(-2.5)),
    "offset-exponent": text("offset", "1e-3", ok(0.001)),
    "offset-a-word": text("offset", "abc", error(not_a_number_text("Offset", "abc"))),
    "offset-decimal-comma": text("offset", "1,5", error(not_a_number_text("Offset", "1,5"))),
    "offset-overflow": text("offset", "-1e999", error(too_large("Offset", "-1e999"))),
    # the budget box
    "budget-empty": text("budget", "", ok(None)),
    "budget-blank": text("budget", "  ", ok(None)),
    "budget-number": text("budget", " 12 ", ok(12)),
    "budget-zero": text("budget", "0", ok(0)),
    "budget-leading-zeros": text("budget", "007", ok(7)),
    "budget-negative": text("budget", "-1", error(not_a_budget("-1"))),
    "budget-fraction": text("budget", "2.5", error(not_a_budget("2.5"))),
    "budget-exponent": text("budget", "1e3", error(not_a_budget("1e3"))),
    "budget-a-word": text("budget", "many", error(not_a_budget("many"))),
    "budget-beyond-exact-integers": text("budget", "9" * 20, error(not_a_budget("9" * 20))),
    # text all the way to a problem
    "typed-matrix-neither-symmetric-nor-triangular": text("matrix-problem", "0 1\n3 0", error(neither(0, 1)), offset=""),
    "typed-matrix-bad-offset": text("matrix-problem", "0 1\n1 0", error(not_a_number_text("Offset", "x")), offset="x"),
    "typed-matrix-with-offset": text("matrix-problem", "0 1\n1 0", ok({"matrix": [[0, 1], [1, 0]], "offset": 4}), offset="4"),
    # the exact solver's own bound
    "solver-at-its-bound": ({"kind": "solver", "n": 30}, ok(2 ** 30)),
    "solver-past-its-bound": (
        {"kind": "solver", "n": 31},
        error("The exact solver in the browser stops at n = 30; this problem has n = 31."),
    ),
}

# every preset written out the way the page writes it into its text boxes
for _name, _preset in PRESETS.items():
    _built = _preset.build()
    _expected = ok({"matrix": _built.matrix.tolist(), "offset": _built.offset})
    if _preset.kind == "maxcut":
        _typed = "\n".join(", ".join(str(v) for v in edge) for edge in _preset.edges)
        TEXTS[f"preset-as-text-{_name}"] = text("edges-problem", _typed, _expected, limit=PAGE_LIMIT)
    else:
        _typed = "\n".join(" ".join(str(v) for v in row) for row in _preset.matrix)
        TEXTS[f"preset-as-text-{_name}"] = text(
            "matrix-problem", _typed, _expected, limit=PAGE_LIMIT, offset=str(_preset.offset) if _preset.offset else "")


# ------------------------------------------------------------- the web side

@pytest.fixture(scope="module")
def web():
    """Everything the web port computes for CASES, the input checks and
    the typed text, from one Node run."""
    payload = {
        "cases": [],
        "checks": [{"name": name, **to_json(spec)} for name, (spec, _) in {**REJECTED, **ACCEPTED}.items()],
        "texts": [spec for spec, _ in TEXTS.values()],
    }
    for case in CASES:
        spec = {
            "name": case["name"],
            "matrix": case["problem"].matrix.tolist(),
            "offset": case["problem"].offset,
            "solve": case["solve"],
            "depths": case["depths"],
            "angles": {str(p): sets for p, sets in case["angles"].items()},
        }
        if case["budget"] is not None:
            spec["budget"] = case["budget"]
        if case["scan"]:
            spec["scan"] = {"samples": SCAN_SAMPLES, "seed": SCAN_SEED}
        payload["cases"].append(spec)
    proc = subprocess.run(
        [NODE, os.path.join(HERE, "web_parity.js")],
        input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8", timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    result["cases"] = {entry["name"]: entry for entry in result["cases"]}
    result["texts"] = dict(zip(TEXTS, result["texts"]))
    return result


def score(problem, p, gammas, betas) -> float:
    """Python-side expected cost at fixed angles, as the demo computes it."""
    qc, meta = build_qaoa(problem, p)
    values = {meta["gammas"][k]: gammas[k] for k in range(p)}
    values.update({meta["betas"][k]: betas[k] for k in range(p)})
    return expectation(qc, problem.all_costs(), values)


def mulberry32(seed: int):
    """The page's random generator (web/qaoa.js), in 32-bit arithmetic."""
    mask = 0xFFFFFFFF
    state = seed & mask

    def imul(a, b):
        return (a * b) & mask

    def rand():
        nonlocal state
        state = (state + 0x6D2B79F5) & mask
        t = imul(state ^ (state >> 15), state | 1)
        t = ((t + imul(t ^ (t >> 7), t | 61)) & mask) ^ t
        return (t ^ (t >> 14)) / 4294967296

    return rand


# -------------------------------------------------------------------- tests

def test_constants_match(web):
    assert web["constants"] == {
        "EXACT_CLASSICAL_LIMIT": EXACT_CLASSICAL_LIMIT,
        "TIE_TOLERANCE": TIE_TOLERANCE,
        "SIMULATION_LIMIT": SIMULATION_LIMIT,
        "SYMMETRY_TOLERANCE": SYMMETRY_TOLERANCE,
        "ZERO_TOLERANCE": ZERO_TOLERANCE,
        "EXACT_SOLVER_MAX_N": 30,  # the browser solver's own bound; above the limit the page solves to
    }
    assert web["constants"]["EXACT_SOLVER_MAX_N"] >= EXACT_CLASSICAL_LIMIT


@pytest.mark.parametrize("name", list(REJECTED))
def test_bad_input_is_rejected_in_the_same_words(name, web):
    """The message pinned for the Python package in test_validation.py,
    from the web port, for the same input."""
    _, message = REJECTED[name]
    assert web["checks"][name] == {"error": message}


@pytest.mark.parametrize("name", list(ACCEPTED))
def test_good_input_is_accepted_with_the_same_result(name, web):
    _, result = ACCEPTED[name]
    assert web["checks"][name] == {"ok": result}


@pytest.mark.parametrize("name", list(TEXTS))
def test_text_typed_into_the_page(name, web):
    """The page's parsers: what they read, and which line they blame."""
    _, expected = TEXTS[name]
    assert web["texts"][name] == expected


def test_the_page_uses_the_limit_tested_here():
    with open(os.path.join(HERE, "..", "web", "app.js"), encoding="utf-8") as handle:
        assert f"const MAX_VARIABLES = {PAGE_LIMIT};" in handle.read()


def test_presets_are_the_same_problems(web):
    """Same names in the same order, same titles and descriptions, and
    the same matrix and offset once built."""
    assert [entry["name"] for entry in web["presets"]] == list(PRESETS)
    for entry in web["presets"]:
        preset = PRESETS[entry["name"]]
        problem = preset.build()
        assert (entry["title"], entry["kind"], entry["n"], entry["about"]) == (
            preset.title, preset.kind, preset.n, preset.about,
        )
        assert entry["matrix"] == problem.matrix.tolist()
        assert entry["offset"] == problem.offset


@pytest.mark.parametrize("name", names())
def test_density_and_recommendation(name, web):
    case, got = BY_NAME[name], web["cases"][name]
    problem = case["problem"]
    rec = recommend(problem, case["budget"])
    assert got["n"] == problem.n
    assert got["density"] == problem.density()  # same arithmetic, so bit for bit
    assert got["recommendation"] == {
        "algorithm": rec.algorithm, "params": rec.params, "rationale": rec.rationale,
    }


@pytest.mark.parametrize("name", names(lambda case: case["solve"]))
def test_exact_solve(name, web):
    """Same optimum and the same reported assignment from both solvers."""
    got = web["cases"][name]["exact"]
    value, x = BY_NAME[name]["problem"].brute_force()
    assert got["x"] == [int(v) for v in x]
    assert got["value"] == pytest.approx(value, abs=1e-9)


@pytest.mark.parametrize("name", names())
def test_ising_form(name, web):
    got = web["cases"][name]["ising"]
    problem = BY_NAME[name]["problem"]
    _, meta = build_qaoa(problem, 1)
    assert np.allclose(got["h"], meta["h"], rtol=0, atol=1e-12)
    assert np.allclose(np.array(got["J"]).reshape(problem.n, problem.n), meta["J"], rtol=0, atol=1e-12)
    assert got["offset"] == pytest.approx(meta["offset"], abs=1e-12)


@pytest.mark.parametrize("name", names(lambda case: case["depths"]))
def test_circuit_structure(name, web):
    """Qubits, depth, gate counts and parameter count, per layer count."""
    case, got = BY_NAME[name], web["cases"][name]["circuits"]
    for p in case["depths"]:
        qc, _ = build_qaoa(case["problem"], p)
        assert got[str(p)] == {
            "qubits": qc.num_qubits,
            "depth": qc.depth(),
            "total": qc.size(),
            "counts": dict(qc.count_ops()),
            "params": qc.num_parameters,
        }


@pytest.mark.parametrize("name", names(lambda case: case["angles"]))
def test_expectation_at_fixed_angles(name, web):
    """The browser's statevector simulator against Qiskit's."""
    case, got = BY_NAME[name], web["cases"][name]["expectations"]
    for p, sets in case["angles"].items():
        for (gammas, betas), value in zip(sets, got[str(p)]):
            assert value == pytest.approx(score(case["problem"], p, gammas, betas), abs=1e-9)


def test_demo_scan_is_reproduced_by_qiskit(web):
    """All 200 scanned values on the page, recomputed with Qiskit at the
    page's own angles."""
    problem = BY_NAME["preset-demo"]["problem"]
    scan = web["cases"]["preset-demo"]["scan"]
    p = recommend(problem).params["p"]
    assert scan["p"] == p

    rand = mulberry32(SCAN_SEED)
    stream = []
    for _ in range(SCAN_SAMPLES):
        gammas = [rand() * math.pi for _ in range(p)]
        betas = [rand() * math.pi for _ in range(p)]
        stream.append((gammas, betas))
    values = [score(problem, p, gammas, betas) for gammas, betas in stream]

    assert len(scan["trace"]) == SCAN_SAMPLES
    assert np.allclose(values, scan["trace"], rtol=0, atol=1e-9)
    best = int(np.argmin(values))
    assert scan["best"]["value"] == pytest.approx(values[best], abs=1e-9)
    assert scan["best"]["gammas"] == stream[best][0]
    assert scan["best"]["betas"] == stream[best][1]


def test_a_seed_of_zero_is_used_as_given(web):
    """The scan's generator seeded with 0, not swapped for the default
    seed: three samples at p = 1, recomputed with Qiskit."""
    problem = BY_NAME["preset-demo"]["problem"]
    rand = mulberry32(0)
    values = []
    for _ in range(3):
        gamma = rand() * math.pi
        beta = rand() * math.pi
        values.append(score(problem, 1, [gamma], [beta]))
    assert np.allclose(web["cases"]["preset-demo"]["seedZero"], values, rtol=0, atol=1e-9)


def test_demo_refinement_is_a_real_expectation(web):
    """The refined value is the Qiskit expectation at the refined angles,
    it improves on the scan, and it does not beat the certified optimum."""
    problem = BY_NAME["preset-demo"]["problem"]
    scan = web["cases"]["preset-demo"]["scan"]
    refined = scan["refined"]
    optimum, _ = problem.brute_force()
    assert refined["value"] == pytest.approx(
        score(problem, scan["p"], refined["gammas"], refined["betas"]), abs=1e-9
    )
    assert optimum - 1e-9 <= refined["value"] <= scan["best"]["value"] + 1e-12


def test_success_probability_is_the_weight_on_optimal_assignments(web):
    """The chance of measuring an optimum, shown by the guided demo:
    Qiskit's probabilities at the page's refined angles, summed over the
    assignments whose cost is the certified optimum."""
    problem = BY_NAME["preset-demo"]["problem"]
    scan = web["cases"]["preset-demo"]["scan"]
    refined, p = scan["refined"], scan["p"]
    qc, meta = build_qaoa(problem, p)
    values = {meta["gammas"][k]: refined["gammas"][k] for k in range(p)}
    values.update({meta["betas"][k]: refined["betas"][k] for k in range(p)})
    probabilities = Statevector(qc.assign_parameters(values)).probabilities()
    costs = problem.all_costs()
    optimum, _ = problem.brute_force()
    expected = float(probabilities[costs <= optimum + TIE_TOLERANCE].sum())
    assert 0.0 < expected < 1.0
    assert scan["success"] == pytest.approx(expected, abs=1e-9)


def test_resumable_forms_reach_the_same_numbers(web):
    """The guided demo runs the exact solve, the scan and the refinement
    a slice at a time so it can draw them. Sliced unevenly, they give
    exactly what the one-call forms give."""
    entry = web["cases"]["preset-demo"]
    stepped, scan = entry["stepped"], entry["scan"]
    assert stepped["exact"] == entry["exact"]
    assert stepped["visited"] == stepped["total"] == 2 ** entry["n"]
    assert stepped["slices"] == math.ceil((stepped["total"] - 1) / 7)
    costs = BY_NAME["preset-demo"]["problem"].all_costs()[1:]  # the slices start after assignment 0
    assert stepped["lo"] == pytest.approx(float(costs.min()), abs=1e-9)
    assert stepped["hi"] == pytest.approx(float(costs.max()), abs=1e-9)
    assert stepped["trace"] == scan["trace"]
    assert stepped["best"] == scan["best"]
    assert stepped["refined"] == scan["refined"]
    assert 0 < stepped["iterations"] <= 300
    assert len(stepped["history"]) == stepped["iterations"]
    assert all(b <= a + 1e-12 for a, b in zip(stepped["history"], stepped["history"][1:]))  # never gets worse
    assert stepped["history"][-1] == scan["refined"]["value"]


def test_web_demo_record(web):
    """The numbers the page shows for the demo preset, as in the README."""
    entry = web["cases"]["preset-demo"]
    assert entry["exact"] == {"value": -8.0, "x": [0, 1, 0, 0, 1, 0]}
    assert entry["circuits"]["3"] == {
        "qubits": 6, "depth": 19, "total": 48, "counts": {"h": 6, "rzz": 24, "rx": 18}, "params": 6,
    }
    assert format(entry["scan"]["best"]["value"], ".3f") == "-6.664"
    assert format(entry["scan"]["refined"]["value"], ".3f") == "-7.187"
    assert format(100 * entry["scan"]["success"], ".1f") == "43.8"  # chance of measuring an optimum, in percent
