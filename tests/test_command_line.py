"""The command line: python -m qopt_builder.

`run` takes a problem from a file, from standard input or from a preset
through the same pipeline as run_demo.py, so for the demo graph it must
print the demo record. The exit status says what happened: 0 a circuit
was built, 1 the builder refused, 2 the input could not be used.
"""

import io
import json
import os
import subprocess
import sys

import pytest
from qiskit import QuantumCircuit

import run_demo
from qopt_builder import PRESETS, __version__, build_qaoa, maxcut_qubo, recommend, scan_angles
from qopt_builder.cli import BAD_INPUT, BUILT, REFUSED, analyse, main
from qopt_builder.presets import DEMO_EDGES
from validation_cases import neither

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEMO_TEXT = "# the demo graph: two triangles joined by two edges\n" + "".join(
    f"{i}, {j}, {w:g}\n" for i, j, w in DEMO_EDGES
)

# what `run` prints for the demo graph after its first line, which names the source
RECORD = [
    "Coupling density: 0.53 (8 of 15 pairs)",
    "Recommendation: exact+qaoa, p = 3",
    " - n=6 <= 24: an exact classical solve is cheap and certifies the quantum result; "
    "recommending exact with QAOA built alongside for comparison.",
    " - coupling density 0.53 is high: deeper QAOA (p=3) earns its depth on dense graphs.",
    "Certified optimum: -8.000 at x = [0 1 0 0 1 0]",
    "As a cut: weight 8.000",
    "QAOA circuit: 6 qubits, depth 19, 48 gates (6 H, 24 RZZ, 18 RX), 6 free parameters",
    "Angle scan: best expected cost -7.308 over 200 samples (seed 20261110)",
    "Gap to the certified optimum: 0.692",
]


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "graph.txt"
    path.write_text(DEMO_TEXT, encoding="utf-8")
    return str(path)


def run(capsys, *args):
    """(exit status, lines printed, what went to the error stream)."""
    status = main(list(args))
    captured = capsys.readouterr()
    return status, captured.out.splitlines(), captured.err


# ------------------------------------------------------------- the record

def test_a_graph_file_gives_the_demo_record(capsys, graph):
    status, lines, err = run(capsys, "run", "--edges", graph)
    assert (status, err) == (BUILT, "")
    assert lines[0] == f"Problem: MaxCut on 6 nodes, 8 edges ({graph})"
    assert lines[1:] == RECORD


def test_the_preset_gives_the_same_record(capsys):
    status, lines, _ = run(capsys, "run", "--preset", "demo")
    assert status == BUILT
    assert lines[0] == "Problem: Demo graph: weighted MaxCut, 6 nodes (preset demo)"
    assert lines[1:] == RECORD


def test_it_reports_what_run_demo_reports(capsys, graph):
    """Same pipeline, same numbers: every figure in the six lines of
    run_demo.py is in the report."""
    run_demo.main([])
    demo = capsys.readouterr().out.splitlines()
    status, lines, _ = run(capsys, "run", "--edges", graph, "--json")
    report = json.loads("\n".join(lines))
    rec, exact, circuit, scan = report["recommendation"], report["exact"], report["circuit"], report["scan"]
    assert demo[0] == f"Recommendation: {rec['algorithm']} {rec['params']}"
    assert demo[1:3] == [f" - {reason}" for reason in rec["rationale"]]
    assert demo[3] == f"Brute-force optimum: {exact['value']:.3f} at x=[{' '.join(map(str, exact['x']))}]"
    assert demo[4] == f"QAOA circuit: {circuit['qubits']} qubits, depth {circuit['depth']}, params {circuit['parameters']}"
    assert demo[5] == f"Best scanned expectation: {scan['value']:.3f} (optimum {exact['value']:.3f})"


def test_json_report(capsys, graph):
    status, lines, _ = run(capsys, "run", "--edges", graph, "--json")
    report = json.loads("\n".join(lines))
    assert status == BUILT
    assert list(report) == ["problem", "recommendation", "refused", "exact", "circuit", "scan", "notes"]
    assert report["problem"] == {
        "kind": "maxcut", "source": graph, "n": 6, "couplings": 8, "pairs": 15, "density": 8 / 15, "offset": 0.0,
    }
    assert report["refused"] is False and report["notes"] == []
    assert report["exact"] == {"value": -8.0, "x": [0, 1, 0, 0, 1, 0]}
    assert report["circuit"] == {
        "p": 3, "recommended_p": 3, "qubits": 6, "depth": 19, "gates": 48,
        "counts": {"h": 6, "rzz": 24, "rx": 18}, "parameters": 6,
    }
    scan = report["scan"]
    expected = scan_angles(maxcut_qubo(DEMO_EDGES, 6), 3)
    assert (scan["samples"], scan["seed"]) == (200, 20261110)
    assert scan["value"] == expected.value
    assert scan["gammas"] == list(expected.gammas) and scan["betas"] == list(expected.betas)
    assert scan["gap"] == expected.value + 8.0


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_runs(capsys, name):
    problem = PRESETS[name].build()
    status, lines, _ = run(capsys, "run", "--preset", name, "--json", "--no-scan")
    report = json.loads("\n".join(lines))
    value, x = problem.brute_force()
    qc, _ = build_qaoa(problem, recommend(problem).params["p"])
    assert status == BUILT
    assert report["problem"]["kind"] == PRESETS[name].kind and report["problem"]["n"] == problem.n
    assert report["exact"] == {"value": value, "x": [int(v) for v in x]}
    assert (report["circuit"]["depth"], report["circuit"]["gates"]) == (qc.depth(), qc.size())
    assert report["scan"] is None and report["notes"] == ["angle scan not run: switched off"]


# ------------------------------------------------------------ the options

def test_a_budget_too_small_is_a_refusal_with_its_own_exit_status(capsys, graph):
    status, lines, err = run(capsys, "run", "--edges", graph, "--budget", "4")
    assert (status, err) == (REFUSED, "")
    assert lines[2] == "Recommendation: none"
    assert lines[-2] == (" - n=6 exceeds the qubit budget of 4: no full-width circuit fits; "
                         "the builder will refuse rather than silently truncate.")
    assert lines[-1] == "Refused: no circuit built."
    assert not any(line.startswith(("Certified", "QAOA circuit", "Angle scan")) for line in lines)

    status, lines, _ = run(capsys, "run", "--edges", graph, "--budget", "4", "--json")
    report = json.loads("\n".join(lines))
    assert status == REFUSED and report["refused"] is True
    assert report["exact"] is None and report["circuit"] is None and report["scan"] is None


def test_a_budget_that_fits_changes_nothing(capsys, graph):
    status, lines, _ = run(capsys, "run", "--edges", graph, "--budget", "6")
    assert status == BUILT and lines[1:] == RECORD


def test_layers_builds_another_depth_and_says_so(capsys, graph):
    problem = maxcut_qubo(DEMO_EDGES, 6)
    qc, _ = build_qaoa(problem, 1)
    status, lines, _ = run(capsys, "run", "--edges", graph, "--layers", "1")
    assert status == BUILT
    assert lines[2] == "Recommendation: exact+qaoa, p = 3"  # the recommendation itself is not altered
    assert "Building p = 1 as asked; the recommendation was p = 3." in lines
    assert f"QAOA circuit: 6 qubits, depth {qc.depth()}, {qc.size()} gates (6 H, 8 RZZ, 6 RX), 2 free parameters" in lines
    assert f"Angle scan: best expected cost {scan_angles(problem, 1).value:.3f} over 200 samples (seed 20261110)" in lines


def test_layers_equal_to_the_recommendation_is_not_remarked_on(capsys, graph):
    status, lines, _ = run(capsys, "run", "--edges", graph, "--layers", "3")
    assert lines[1:] == RECORD


def test_samples_and_seed_reach_the_scan(capsys, graph):
    expected = scan_angles(maxcut_qubo(DEMO_EDGES, 6), 3, samples=25, seed=7).value
    status, lines, _ = run(capsys, "run", "--edges", graph, "--samples", "25", "--seed", "7")
    assert f"Angle scan: best expected cost {expected:.3f} over 25 samples (seed 7)" in lines
    status, lines, _ = run(capsys, "run", "--edges", graph, "--samples", "1")
    assert any(line.endswith("over 1 sample (seed 20261110)") for line in lines)


def test_no_scan(capsys, graph):
    status, lines, _ = run(capsys, "run", "--edges", graph, "--no-scan")
    assert status == BUILT
    assert lines[1:] == RECORD[:-2] + ["Angle scan: skipped (--no-scan)"]


def test_draw_prints_the_circuit_after_the_report(capsys, graph):
    status, lines, _ = run(capsys, "run", "--edges", graph, "--draw", "--no-scan")
    assert status == BUILT
    drawing = lines[len(RECORD):]
    assert any(line.startswith("q_0:") for line in drawing) and any(line.startswith("q_5:") for line in drawing)


# -------------------------------------------------------- kinds of input

def test_a_matrix_file_with_an_offset(capsys, tmp_path):
    preset = PRESETS["vertex-cover-6"]
    path = tmp_path / "cover.txt"
    path.write_text("\n".join(" ".join(f"{v:g}" for v in row) for row in preset.matrix) + "\n", encoding="utf-8")
    status, lines, _ = run(capsys, "run", "--qubo", str(path), "--offset", "16")
    value, x = preset.build().brute_force()
    assert status == BUILT
    assert lines[0] == f"Problem: QUBO on 6 variables, offset 16 ({path})"
    assert f"Certified optimum: {value:.3f} at x = [{' '.join(str(int(v)) for v in x)}]" in lines
    assert not any(line.startswith("As a cut") for line in lines)


def test_standard_input(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(DEMO_TEXT))
    status, lines, _ = run(capsys, "run", "--edges", "-")
    assert status == BUILT
    assert lines[0] == "Problem: MaxCut on 6 nodes, 8 edges (standard input)"
    assert lines[1:] == RECORD


def test_a_file_from_windows_with_a_byte_order_mark(capsys, tmp_path):
    path = tmp_path / "graph.txt"
    path.write_bytes(b"\xef\xbb\xbf" + DEMO_TEXT.replace("\n", "\r\n").encode("utf-8"))
    status, lines, _ = run(capsys, "run", "--edges", str(path))
    assert status == BUILT and lines[1:] == RECORD


def test_nodes_adds_nodes_that_have_no_edges(capsys, tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("0 1\n", encoding="utf-8")
    status, lines, _ = run(capsys, "run", "--edges", str(path), "--nodes", "4", "--no-scan")
    assert status == BUILT
    assert lines[0] == f"Problem: MaxCut on 4 nodes, 1 edge ({path})"
    assert lines[1] == "Coupling density: 0.17 (1 of 6 pairs)"
    status, lines, err = run(capsys, "run", "--edges", str(path), "--nodes", "1")
    assert status == BAD_INPUT and lines == []
    assert err == f"error: {path}: --nodes 1 is fewer than the 2 nodes the edges use.\n"


def test_one_of_everything_is_singular(capsys, tmp_path):
    path = tmp_path / "single.txt"
    path.write_text("-1\n", encoding="utf-8")
    status, lines, _ = run(capsys, "run", "--qubo", str(path))
    assert status == BUILT
    assert lines[0] == f"Problem: QUBO on 1 variable ({path})"
    assert lines[1] == "Coupling density: 0.00 (0 of 0 pairs)"
    assert "QAOA circuit: 1 qubit, depth 3, 3 gates (1 H, 1 RZ, 1 RX), 2 free parameters" in lines


def test_beyond_both_limits_the_report_says_what_was_not_computed(capsys, tmp_path):
    path = tmp_path / "path.txt"
    path.write_text("".join(f"{i} {i + 1}\n" for i in range(25)), encoding="utf-8")
    status, lines, _ = run(capsys, "run", "--edges", str(path))
    assert status == BUILT
    assert lines[2] == "Recommendation: qaoa, p = 1"
    assert "Certified optimum: not computed (n=26 is above the exact-solve limit of 24)" in lines
    assert lines[-1] == "Angle scan: not simulated (n=26 is above the statevector limit of 14)"
    status, lines, _ = run(capsys, "run", "--edges", str(path), "--json")
    report = json.loads("\n".join(lines))
    assert report["exact"] is None and report["scan"] is None and report["circuit"]["qubits"] == 26
    assert report["notes"] == [
        "exact solve not run: n=26 is above the exact-solve limit of 24",
        "angle scan not run: n=26 is above the statevector limit of 14",
    ]


# ------------------------------------------------------------- bad input

def bad_file(tmp_path, content):
    path = tmp_path / "problem.txt"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("option, content, message", [
    ("--edges", "0 1\n1 x\n", 'Line 2: "x" is not a number.'),
    ("--edges", "0 1\n\n2 2\n", "Line 3: an edge cannot join node 2 to itself."),
    ("--edges", "# only a comment\n", "No edges given. Enter one edge per line: two node numbers and an optional weight."),
    ("--qubo", "1 2\n3\n", "Line 2 has 1 number, but the matrix has 2 rows. A QUBO matrix is square: "
                           "every row needs as many numbers as there are rows."),
    ("--qubo", "0 1\n3 0\n", neither(0, 1)),
    ("--qubo", "", "No matrix given. Enter one row per line, numbers separated by spaces or commas."),
], ids=["a-word", "a-self-loop", "no-edges", "a-short-row", "neither-symmetric-nor-triangular", "no-matrix"])
def test_bad_problem_text_is_reported_with_its_file_and_line(capsys, tmp_path, option, content, message):
    path = bad_file(tmp_path, content)
    status, lines, err = run(capsys, "run", option, path)
    assert (status, lines) == (BAD_INPUT, [])
    assert err == f"error: {path}: {message}\n"


def test_a_file_that_is_not_there(capsys, tmp_path):
    path = str(tmp_path / "missing.txt")
    status, lines, err = run(capsys, "run", "--edges", path)
    assert (status, lines) == (BAD_INPUT, [])
    assert err.startswith(f"error: cannot read {path}: ")


def test_a_file_that_is_not_text(capsys, tmp_path):
    path = bad_file(tmp_path, b"\xff\xfe\x00\x01 not text")
    status, lines, err = run(capsys, "run", "--edges", path)
    assert (status, lines) == (BAD_INPUT, [])
    assert err == f"error: {path} is not a text file in UTF-8.\n"


@pytest.mark.parametrize("arguments, complaint", [
    ([], "the following arguments are required: COMMAND"),
    (["run"], "one of the arguments --edges --qubo --preset is required"),
    (["run", "--preset", "demo", "--edges", "g.txt"], "not allowed with argument"),
    (["run", "--preset", "nope"], "invalid choice: 'nope'"),
    (["run", "--preset", "demo", "--nodes", "8"], "--nodes goes with --edges"),
    (["run", "--preset", "demo", "--offset", "2"], "--offset goes with --qubo"),
    (["run", "--preset", "demo", "--budget", "-1"], "a qubit budget must be 0 or more"),
    (["run", "--preset", "demo", "--budget", "2.5"], "'2.5' is not a whole number"),
    (["run", "--preset", "demo", "--layers", "0"], "the number of layers must be 1 or more"),
    (["run", "--preset", "demo", "--samples", "0"], "the number of samples must be 1 or more"),
    (["run", "--preset", "demo", "--seed", "-3"], "the seed must be 0 or more"),
    (["run", "--qubo", "m.txt", "--offset", "nan"], "'nan' is not a finite number"),
    (["run", "--qubo", "m.txt", "--offset", "much"], "'much' is not a number"),
    (["solve"], "invalid choice: 'solve'"),
], ids=["no-command", "no-problem", "two-problems", "unknown-preset", "nodes-without-edges", "offset-without-qubo",
        "negative-budget", "fractional-budget", "zero-layers", "zero-samples", "negative-seed", "offset-nan",
        "offset-a-word", "unknown-command"])
def test_wrong_arguments_are_usage_errors(capsys, arguments, complaint):
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)
    assert exit_info.value.code == BAD_INPUT
    assert complaint in capsys.readouterr().err


# ------------------------------------------------------ the other commands

def test_presets_lists_every_preset(capsys):
    status, lines, _ = run(capsys, "presets")
    assert status == BUILT
    for name, preset in PRESETS.items():
        assert any(line.startswith(name) and line.endswith(preset.title) for line in lines)
    run_demo.main(["--list"])
    assert capsys.readouterr().out.splitlines() == lines  # the same list as run_demo.py --list


def test_analyse_returns_the_circuit_itself():
    problem = maxcut_qubo(DEMO_EDGES, 6)
    report, circuit = analyse(problem, scan=False)
    assert isinstance(circuit, QuantumCircuit) and circuit.num_qubits == 6
    assert json.loads(json.dumps(report)) == report  # plain data
    report, circuit = analyse(problem, budget=5)
    assert report["refused"] and circuit is None


# ----------------------------------------------------------- as a program

def program(*args, stdin=None):
    return subprocess.run(
        [sys.executable, "-m", "qopt_builder", *args], cwd=ROOT, input=stdin,
        capture_output=True, text=True, encoding="utf-8", timeout=300,
    )


def test_run_as_a_program(graph):
    done = program("run", "--edges", graph)
    assert done.returncode == BUILT, done.stderr
    assert done.stdout.splitlines()[1:] == RECORD
    assert not done.stderr


def test_exit_statuses_as_a_program(graph):
    assert program("run", "--edges", graph, "--budget", "4", "--no-scan").returncode == REFUSED
    piped = program("run", "--edges", "-", "--no-scan", stdin="0 1\n1 1\n")
    assert piped.returncode == BAD_INPUT
    assert piped.stderr == "error: standard input: Line 2: an edge cannot join node 1 to itself.\n"
    assert program("run").returncode == BAD_INPUT


def test_version_as_a_program():
    done = program("--version")
    assert done.returncode == 0 and done.stdout.strip() == f"qopt-builder {__version__}"
