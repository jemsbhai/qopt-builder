"""The hardware package: its five files, and the archive that holds them.

A package has to be right in a way a simulator on this side cannot
forgive later: it is the thing a user sends to a quantum computer. So
the circuit file is read back by Qiskit and compared gate for gate with
the circuit build_qaoa() makes, the problem file is read back and
compared with the problem, and the archive is opened with Python's own
zipfile.

tests/test_run_on_ibm.py runs the script a package carries, and
tests/test_web_parity.py checks that the web page writes the same bytes.
"""

import io
import json
import math
import random
import struct
import time
import zipfile

import numpy as np
import pytest
from qiskit import qasm2
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import Statevector

import qopt_builder
from qopt_builder import (
    PRESETS, QUBO, build_qaoa, maxcut_qubo, package_files, recommend, run_on_ibm, write_folder, zip_bytes,
)
from qopt_builder.package import (
    FILES, FORMAT, PACKAGE_NAME, README_TEMPLATE, REQUIREMENTS, RUN_SCRIPT, circuit_qasm, describe_problem,
    exact_cost, number_text,
)
from qopt_builder.recommend import EXACT_CLASSICAL_LIMIT
from qopt_builder.simulate import SIMULATION_LIMIT

WHEN = (2026, 11, 11, 12, 34, 56)
NATIVE_RZZ = [qasm2.CustomInstruction("rzz", 1, 2, RZZGate)]


def angles(p: int):
    """Fixed angles for p layers, all different."""
    return [0.3 + 0.17 * k for k in range(p)], [1.1 - 0.23 * k for k in range(p)]


def make(problem: QUBO, *, kind="qubo", edges=None, layers=None, solve=True, angles_from="fixed test angles"):
    """A package for `problem` at fixed angles: (files, layers, gammas, betas, optimum)."""
    rec = recommend(problem)
    p = rec.params["p"] if layers is None else layers
    gammas, betas = angles(p)
    optimum = problem.brute_force()[1] if solve else None
    files = package_files(problem, kind=kind, edges=edges, recommendation=rec, optimum=optimum, layers=p,
                          gammas=gammas, betas=betas, angles_from=angles_from)
    return files, p, gammas, betas, optimum


def preset_package(name: str):
    preset = PRESETS[name]
    return make(preset.build(), kind=preset.kind, edges=len(preset.edges) if preset.kind == "maxcut" else None,
                solve=preset.n <= SIMULATION_LIMIT)


def random_problem(seed: int, n: int) -> QUBO:
    rng = np.random.default_rng(seed)
    return QUBO(np.triu(rng.normal(size=(n, n))), offset=float(rng.normal()))


# ----------------------------------------------------------------- numbers

@pytest.mark.parametrize("value, text", [
    (0.0, "0.0"), (-0.0, "0.0"), (1, "1.0"), (-1.0, "-1.0"), (8.0, "8.0"), (0.1, "0.1"), (-0.25, "-0.25"),
    (123.456, "123.456"), (100.0, "100.0"), (0.004, "0.004"),
    (1e-6, "0.000001"), (1.5e-6, "0.0000015"), (1e-7, "1.0e-7"), (9.999999e-7, "9.999999e-7"),
    (1e15, "1000000000000000.0"), (1234567890123456.0, "1234567890123456.0"), (1e16, "1.0e+16"),
    (1.25e21, "1.25e+21"), (5e-324, "5.0e-324"), (1.7976931348623157e308, "1.7976931348623157e+308"),
    (0.1 + 0.2, "0.30000000000000004"), (1 / 3, "0.3333333333333333"), (np.float64(2.5), "2.5"),
])
def test_number_text(value, text):
    assert number_text(value) == text


def test_number_text_reads_back_as_the_same_number():
    rng = random.Random(20261111)
    checked = 0
    while checked < 20000:
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isfinite(x) and x != 0:
            text = number_text(x)
            assert float(text) == x, (x, text)
            assert json.loads(text) == x  # the text is a JSON number
            assert "." in text  # and a real number in OpenQASM 2, which wants the point
            checked += 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "1.0", None, True, [1.0]])
def test_number_text_wants_a_finite_number(value):
    with pytest.raises(ValueError, match="must be finite"):
        number_text(value)


# ------------------------------------------------------------ the five files

def test_a_package_is_five_files_in_a_fixed_order():
    files, *_ = preset_package("demo")
    assert tuple(files) == FILES == ("README.txt", "problem.json", "circuit.qasm", "run_on_ibm.py", "requirements.txt")
    for name, text in files.items():
        assert text.endswith("\n") and "\r" not in text, name
        text.encode("ascii")  # every file is plain ASCII


def test_the_script_and_requirements_are_carried_unchanged():
    files, *_ = preset_package("demo")
    assert files["run_on_ibm.py"] == RUN_SCRIPT
    assert files["requirements.txt"] == REQUIREMENTS
    with open(run_on_ibm.__file__, encoding="utf-8") as handle:
        assert handle.read() == RUN_SCRIPT
    assert "qopt_builder" not in RUN_SCRIPT  # it runs without this package


@pytest.mark.parametrize("name", list(PRESETS))
def test_problem_file_states_the_problem(name):
    preset, problem = PRESETS[name], PRESETS[name].build()
    files, p, gammas, betas, optimum = preset_package(name)
    data = json.loads(files["problem.json"])

    assert data["format"] == FORMAT
    assert data["made_by"] == f"qopt-builder {qopt_builder.__version__}"
    stated = data["problem"]
    assert stated["kind"] == preset.kind
    assert stated["variables"] == problem.n
    assert stated["offset"] == problem.offset
    assert stated["matrix"] == problem.matrix.tolist()  # every entry, exactly

    if optimum is None:
        assert data["certified_optimum"] is None
    else:
        value, x = problem.brute_force()
        assert data["certified_optimum"]["assignment"] == [int(v) for v in x]
        assert data["certified_optimum"]["cost"] == pytest.approx(value, abs=1e-12)

    rec = recommend(problem)
    assert data["recommendation"] == {"algorithm": rec.algorithm, "layers": rec.params["p"], "reasons": rec.rationale}
    assert data["qaoa"] == {"layers": p, "gammas": gammas, "betas": betas, "angles_from": "fixed test angles"}

    circuit, _ = build_qaoa(problem, p)
    counts = circuit.count_ops()
    assert data["circuit"] == {
        "file": "circuit.qasm", "qubits": problem.n, "depth": circuit.depth(), "gates": circuit.size(),
        "h": counts.get("h", 0), "rz": counts.get("rz", 0), "rzz": counts.get("rzz", 0), "rx": counts.get("rx", 0),
        "measurements": problem.n,
    }


def test_problem_file_for_the_demo_graph():
    """The numbers of the demo record, as the package states them."""
    files, *_ = preset_package("demo")
    data = json.loads(files["problem.json"])
    assert data["problem"]["description"] == "MaxCut on 6 nodes, 8 edges"
    assert data["certified_optimum"] == {"cost": -8.0, "assignment": [0, 1, 0, 0, 1, 0]}
    assert data["circuit"] == {"file": "circuit.qasm", "qubits": 6, "depth": 19, "gates": 48,
                               "h": 6, "rz": 0, "rzz": 24, "rx": 18, "measurements": 6}


def test_description():
    assert describe_problem("maxcut", 6, 8) == "MaxCut on 6 nodes, 8 edges"
    assert describe_problem("maxcut", 2, 1) == "MaxCut on 2 nodes, 1 edge"
    assert describe_problem("qubo", 5) == "QUBO on 5 variables"
    assert describe_problem("qubo", 1) == "QUBO on 1 variable"


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("layout", ["upper", "symmetric"])
def test_exact_cost_is_the_cost(seed, layout):
    problem = random_problem(seed, 7)
    if layout == "symmetric":
        problem = QUBO((problem.matrix + problem.matrix.T) / 2, offset=problem.offset)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        x = rng.integers(0, 2, size=7)
        assert exact_cost(problem, x) == pytest.approx(problem.cost([int(v) for v in x]), abs=1e-12)


# ------------------------------------------------------------- the circuit

def bound(problem: QUBO, p: int, gammas, betas):
    """The circuit of build_qaoa() with the angles set."""
    circuit, meta = build_qaoa(problem, p)
    values = {meta["gammas"][k]: gammas[k] for k in range(p)}
    values.update({meta["betas"][k]: betas[k] for k in range(p)})
    return circuit.assign_parameters(values, strict=False)


def gates(circuit):
    """(name, qubits, angles) of every gate, in order."""
    return [
        (item.operation.name, [circuit.find_bit(q).index for q in item.qubits],
         [float(v) for v in item.operation.params])
        for item in circuit.data
    ]


CIRCUIT_CASES = {name: PRESETS[name].build() for name in PRESETS if PRESETS[name].n <= SIMULATION_LIMIT}
CIRCUIT_CASES.update({
    "linear-terms": QUBO(np.diag([-1.0, -2.0, 3.0, 4.0])),
    "random-7": random_problem(3, 7),
    "one-variable": QUBO([[2.0]]),
    "no-terms": QUBO(np.zeros((3, 3))),
})


@pytest.mark.parametrize("name", list(CIRCUIT_CASES))
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_circuit_file_is_the_circuit_the_builder_makes(name, layers):
    """Read back by Qiskit, the file holds exactly the gates of
    build_qaoa() with the angles set, followed by one measurement of
    each qubit into the bit of the same number."""
    problem = CIRCUIT_CASES[name]
    files, p, gammas, betas, _ = make(problem, layers=layers)
    loaded = qasm2.loads(files["circuit.qasm"], custom_instructions=NATIVE_RZZ)

    n = problem.n
    assert loaded.num_qubits == n and loaded.num_clbits == n
    measurements = loaded.data[-n:]
    assert [(m.operation.name, loaded.find_bit(m.qubits[0]).index, loaded.find_bit(m.clbits[0]).index)
            for m in measurements] == [("measure", i, i) for i in range(n)]

    written = [g for g in gates(loaded) if g[0] != "measure"]
    built = gates(bound(problem, p, gammas, betas))
    assert [(g[0], g[1]) for g in written] == [(g[0], g[1]) for g in built]
    for (_, _, mine), (_, _, theirs) in zip(written, built):
        assert mine == pytest.approx(theirs, rel=1e-15, abs=1e-300)


@pytest.mark.parametrize("name", ["demo", "vertex-cover-6", "linear-terms", "random-7"])
def test_circuit_file_needs_nothing_from_qiskit(name):
    """The file defines rzz itself, so a reader that knows only the
    standard OpenQASM 2 gates accepts it, in strict mode too, and gets
    the same state as the builder's circuit."""
    problem = CIRCUIT_CASES[name]
    files, p, gammas, betas, _ = make(problem)
    for options in ({}, {"strict": True}):
        loaded = qasm2.loads(files["circuit.qasm"], **options)
        assert "rzz" not in loaded.count_ops() or loaded.count_ops()["rzz"] > 0
        loaded.remove_final_measurements()
        assert Statevector.from_instruction(loaded).equiv(Statevector.from_instruction(bound(problem, p, gammas, betas)))


def test_circuit_file_layout():
    files, *_ = preset_package("demo")
    lines = files["circuit.qasm"].split("\n")
    assert lines[:9] == [
        "// QAOA circuit made by qopt-builder 0.1.0",
        "// Problem: MaxCut on 6 nodes, 8 edges",
        "// 3 layers, angles filled in; problem.json lists them",
        "// Qubit q[i] stands for variable x_i and is measured into bit c[i]",
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }",
        "qreg q[6];",
        "creg c[6];",
    ]
    assert lines[9:15] == [f"h q[{i}];" for i in range(6)]
    assert lines[15] == "// layer 1 of 3: cost"
    assert lines[16] == "rzz(0.3) q[0],q[1];"  # 2 x gamma 0.3 x J 0.5
    assert lines[-7:-1] == [f"measure q[{i}] -> c[{i}];" for i in range(6)]
    assert [line for line in lines if line.startswith("//")][4:] == [
        f"// layer {k} of 3: {part}" for k in (1, 2, 3) for part in ("cost", "mixer")
    ]


def test_circuit_text_alone():
    problem = maxcut_qubo([(0, 1)], 2)
    text = circuit_qasm(problem, 1, [0.5], [0.25], "two nodes")
    assert text.split("\n")[9:] == [
        "h q[0];", "h q[1];", "// layer 1 of 1: cost", "rzz(0.5) q[0],q[1];", "// layer 1 of 1: mixer",
        "rx(0.5) q[0];", "rx(0.5) q[1];", "measure q[0] -> c[0];", "measure q[1] -> c[1];", "",
    ]


# -------------------------------------------------------------- the README

def test_readme_names_the_problem_and_leaves_no_blank_to_fill():
    files, *_ = make(PRESETS["demo"].build(), kind="maxcut", edges=8, angles_from="from a test {layers}")
    readme = files["README.txt"]
    assert "Problem:  MaxCut on 6 nodes, 8 edges\n" in readme
    assert "Circuit:  QAOA with 3 layers on 6 qubits, angles filled in\n" in readme
    assert "Angles:   from a test {layers}\n" in readme  # text that is put in is not read for blanks again
    assert f"Made by:  qopt-builder {qopt_builder.__version__}," in readme
    assert readme.replace("from a test {layers}", "").count("{") == 0
    single, *_ = make(QUBO([[2.0]]), layers=1)
    assert "Circuit:  QAOA with 1 layer on 1 qubit, angles filled in\n" in single["README.txt"]


def test_readme_agrees_with_the_script_and_the_package():
    """The commands, the default number of shots, the default simulated
    device and the two limits the README states are the ones in force."""
    for flag in ("--simulate", "--rehearse", "--hardware", "--job ID", "--backend", "--shots"):
        assert flag in README_TEMPLATE, flag
        assert flag.split(" ")[0] in RUN_SCRIPT
    assert f"one job of {run_on_ibm.SHOTS} shots" in README_TEMPLATE
    assert f"The copy\nis {run_on_ibm.REHEARSAL_DEVICE} unless" in README_TEMPLATE
    assert f"of up to {EXACT_CLASSICAL_LIMIT} variables" in README_TEMPLATE
    assert f"For problems of up to {run_on_ibm.IDEAL_LIMIT} variables" in README_TEMPLATE
    assert run_on_ibm.IDEAL_LIMIT == SIMULATION_LIMIT
    assert run_on_ibm.FORMAT == FORMAT
    assert all(f"\n{name}" in README_TEMPLATE for name in FILES)
    assert "run_on_ibm.run(backend, shots=1024)" in README_TEMPLATE
    assert "It has not been run on a real quantum computer." in README_TEMPLATE.replace("\n", " ")
    assert max(len(line) for line in README_TEMPLATE.split("\n")) <= 78


def test_requirements_name_the_three_packages():
    wanted = [line.split(">=")[0] for line in REQUIREMENTS.split("\n") if line and not line.startswith("#")]
    assert wanted == ["qiskit", "qiskit-ibm-runtime", "qiskit-aer"]


# ------------------------------------------------------------- what it refuses

def test_a_refused_problem_has_no_package():
    problem = PRESETS["demo"].build()
    with pytest.raises(ValueError, match="refused this problem, so there is no circuit to package"):
        package_files(problem, kind="maxcut", edges=8, recommendation=recommend(problem, qubit_budget=4), optimum=None,
                      layers=3, gammas=[0.1] * 3, betas=[0.1] * 3, angles_from="x")


@pytest.mark.parametrize("gammas, betas, message", [
    ([0.1, 0.2], [0.1, 0.2, 0.3], "gammas must be a list of 3 numbers, one per layer."),
    ([0.1, 0.2, 0.3], [0.1], "betas must be a list of 3 numbers, one per layer."),
    ([0.1, float("nan"), 0.3], [0.1, 0.2, 0.3], "gammas must be a list of 3 numbers, one per layer."),
    ([0.1, 0.2, 0.3], [0.1, "0.2", 0.3], "betas must be a list of 3 numbers, one per layer."),
    (None, [0.1, 0.2, 0.3], "gammas must be a list of 3 numbers, one per layer."),
])
def test_angles_must_fit_the_layers(gammas, betas, message):
    problem = PRESETS["demo"].build()
    with pytest.raises(ValueError) as error:
        package_files(problem, kind="maxcut", edges=8, recommendation=recommend(problem), optimum=None, layers=3,
                      gammas=gammas, betas=betas, angles_from="x")
    assert str(error.value) == message


@pytest.mark.parametrize("layers", [0, -1, 1.5, "2", None])
def test_layers_must_be_a_positive_whole_number(layers):
    problem = PRESETS["demo"].build()
    with pytest.raises(ValueError, match="QAOA depth p must be a positive whole number."):
        package_files(problem, kind="maxcut", edges=8, recommendation=recommend(problem), optimum=None, layers=layers,
                      gammas=[0.1], betas=[0.1], angles_from="x")


def test_kind_is_maxcut_or_qubo():
    problem = PRESETS["demo"].build()
    with pytest.raises(ValueError, match='kind must be "maxcut" or "qubo".'):
        package_files(problem, kind="graph", recommendation=recommend(problem), optimum=None, layers=1,
                      gammas=[0.1], betas=[0.1], angles_from="x")


# ---------------------------------------------------------------- on disk

def test_folder_holds_the_same_bytes_on_every_system(tmp_path):
    files, *_ = preset_package("demo")
    write_folder(files, tmp_path / "deep" / "package")
    for name, text in files.items():
        assert (tmp_path / "deep" / "package" / name).read_bytes() == text.encode("utf-8")
    assert sorted(path.name for path in (tmp_path / "deep" / "package").iterdir()) == sorted(FILES)


def test_archive_is_a_zip_python_can_open():
    files, *_ = preset_package("demo")
    data = zip_bytes(files, when=WHEN)
    archive = zipfile.ZipFile(io.BytesIO(data))
    assert archive.testzip() is None  # every checksum is right
    assert archive.namelist() == [f"{PACKAGE_NAME}/{name}" for name in FILES]
    for info in archive.infolist():
        name = info.filename.split("/")[1]
        assert archive.read(info).decode("utf-8") == files[name]
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == WHEN
        assert info.file_size == info.compress_size == len(files[name].encode("utf-8"))
    assert data[:4] == b"PK\x03\x04" and data[-22:-18] == b"PK\x05\x06"


def test_archive_is_the_same_bytes_every_time():
    files, *_ = preset_package("demo")
    assert zip_bytes(files, when=WHEN) == zip_bytes(dict(files), when=WHEN)
    later = (2026, 11, 11, 12, 34, 58)
    assert zip_bytes(files, when=WHEN) != zip_bytes(files, when=later)


def test_archive_is_stamped_with_the_time_it_was_made():
    files, *_ = preset_package("demo")
    before = time.time()
    stamped = zipfile.ZipFile(io.BytesIO(zip_bytes(files))).infolist()[0].date_time
    after = time.time()
    made = time.mktime(stamped + (0, 0, -1))
    assert before - 2 <= made <= after  # a zip keeps even seconds only


def test_archive_folder_name():
    files, *_ = preset_package("demo")
    assert zipfile.ZipFile(io.BytesIO(zip_bytes(files, folder="mine", when=WHEN))).namelist()[0] == "mine/README.txt"
    accented = zipfile.ZipFile(io.BytesIO(zip_bytes(files, folder="démo", when=WHEN)))
    assert accented.testzip() is None
    assert accented.namelist()[0] == "démo/README.txt"  # marked as UTF-8, so the name survives


@pytest.mark.parametrize("year", [1979, 2108])
def test_archive_dates_stay_in_the_range_a_zip_can_hold(year):
    files, *_ = preset_package("demo")
    with pytest.raises(ValueError, match="1980 to 2107"):
        zip_bytes(files, when=(year, 1, 1, 0, 0, 0))
