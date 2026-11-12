"""The script a hardware package carries: run_on_ibm.py.

What can be run without a quantum computer is run here for real: the
ideal simulator, Qiskit's generic simulated device (the path any other
provider's backend takes), and, where qiskit-ibm-runtime and qiskit-aer
are installed, a simulated copy of an IBM device through IBM's own
Sampler.

The steps that talk to IBM (signing in, choosing a device, fetching a
job) cannot be run here. They are exercised with a stand-in for the IBM
service, which checks what the script asks for and in what order, not
what IBM answers. No test here can reach a real account: every test of
--hardware and --job replaces the sign-in step first.
"""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector

from qopt_builder import QUBO, PRESETS, build_qaoa, expectation, package_files, recommend, run_on_ibm, write_folder
from qopt_builder.problems import TIE_TOLERANCE



def has_ibm() -> bool:
    """qiskit-aer, and qiskit-ibm-runtime at the version the script was written for or later."""
    if any(importlib.util.find_spec(name) is None for name in ("qiskit_ibm_runtime", "qiskit_aer")):
        return False
    major, minor = importlib.metadata.version("qiskit-ibm-runtime").split(".")[:2]
    return (int(major), int(minor)) >= (0, 52)


needs_ibm = pytest.mark.skipif(not has_ibm(), reason="needs qiskit-aer and qiskit-ibm-runtime 0.52 or later")

# Angles close to the best of the builder's own scan of the demo graph:
# an ideal run puts about half its shots on a best cut.
DEMO_ANGLES = ([0.3519, 0.6958, 0.0772], [1.148, 0.9838, 1.9168])

# Four variables with nothing coupling them: the best answer sets the
# first two and clears the last two. Read backwards it is a different
# assignment, so a slip in the bit order cannot hide. At these angles an
# ideal run gives the best answer about half the time, and its mirror
# image almost never.
ONE_SIDED = QUBO(np.diag([-1.0, -2.0, 3.0, 4.0]))
ONE_SIDED_ANGLES = ([0.42], [2.12])


def write_package(folder, problem, *, kind="qubo", edges=None, gammas=None, betas=None, layers=None, solve=True):
    rec = recommend(problem)
    p = rec.params["p"] if layers is None else layers
    gammas = [0.3 + 0.17 * k for k in range(p)] if gammas is None else gammas
    betas = [1.1 - 0.23 * k for k in range(p)] if betas is None else betas
    files = package_files(
        problem, kind=kind, edges=edges, recommendation=rec, optimum=problem.brute_force()[1] if solve else None,
        layers=p, gammas=gammas, betas=betas, angles_from="test",
    )
    write_folder(files, folder)
    return p, gammas, betas


@pytest.fixture
def demo(tmp_path):
    """The demo graph, packaged with fixed angles."""
    folder = tmp_path / "demo"
    p, gammas, betas = write_package(folder, PRESETS["demo"].build(), kind="maxcut", edges=8,
                                     gammas=DEMO_ANGLES[0], betas=DEMO_ANGLES[1])
    return folder, p, gammas, betas


@pytest.fixture
def one_sided(tmp_path):
    folder = tmp_path / "one-sided"
    write_package(folder, ONE_SIDED, gammas=ONE_SIDED_ANGLES[0], betas=ONE_SIDED_ANGLES[1])
    return folder


def lopsided() -> QUBO:
    """Five variables with a full symmetric matrix of decimals and an
    offset: every part of the cost has something to get wrong."""
    rng = np.random.default_rng(5)
    half = rng.normal(size=(5, 5))
    return QUBO((half + half.T) / 2, offset=1.75)


@pytest.fixture
def symmetric(tmp_path):
    folder = tmp_path / "symmetric"
    problem = lopsided()
    p, gammas, betas = write_package(folder, problem)
    return folder, problem, p, gammas, betas


def run_script(*args):
    """main() of the script, as a call: its exit status."""
    return run_on_ibm.main([str(arg) for arg in args])


# ------------------------------------------------------- reading the results

def test_bit_order():
    """Qiskit writes bit c[0] last, and c[0] holds x_0."""
    assert run_on_ibm.assignment("0011") == [1, 1, 0, 0]
    assert run_on_ibm.assignment("1") == [1]
    assert run_on_ibm.assignment("01 0") == [0, 1, 0]  # a space between registers is not a bit


def test_cost_is_the_cost_of_the_problem(demo, symmetric):
    for folder, problem in ((demo[0], PRESETS["demo"].build()), (symmetric[0], symmetric[1])):
        package, _ = run_on_ibm.load_package(folder)
        for number in range(2 ** problem.n):  # every assignment there is
            x = [(number >> i) & 1 for i in range(problem.n)]
            assert run_on_ibm.cost(package, x) == pytest.approx(problem.cost(x), abs=1e-12)
        value, best = problem.brute_force()
        assert package["certified_optimum"]["assignment"] == [int(v) for v in best]
        assert run_on_ibm.cost(package, package["certified_optimum"]["assignment"]) == pytest.approx(value, abs=1e-12)
    package, _ = run_on_ibm.load_package(demo[0])
    assert run_on_ibm.cost(package, package["certified_optimum"]["assignment"]) == -8.0


def test_summary_of_counts(demo):
    package, _ = run_on_ibm.load_package(demo[0])
    # as Qiskit prints them: x = 010010 and 110010 are optimal (-8), 000000 costs 0, 100000 costs -2
    counts = {"010010": 30, "010011": 30, "000000": 25, "000001": 15}
    summary = run_on_ibm.summarise(package, counts)
    assert summary["shots"] == 100
    assert summary["shots_at_optimum"] == 60
    assert summary["average_cost"] == pytest.approx((60 * -8.0 + 25 * 0.0 + 15 * -2.0) / 100)
    assert summary["best"] == {"x": [0, 1, 0, 0, 1, 0], "shots": 30, "cost": -8.0}  # the first of two equally good
    assert [row["shots"] for row in summary["outcomes"]] == [30, 30, 25, 15]
    assert [row["x"] for row in summary["outcomes"]][:2] == [[0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0]]


def test_summary_rejects_results_of_another_size(demo):
    package, _ = run_on_ibm.load_package(demo[0])
    with pytest.raises(run_on_ibm.PackageError, match="do not fit the problem"):
        run_on_ibm.summarise(package, {"0101": 10})
    with pytest.raises(run_on_ibm.PackageError, match="returned no results"):
        run_on_ibm.summarise(package, {})


def test_ideal_values_are_those_of_the_builder(demo):
    """The script works out what an ideal run gives from the circuit
    file. It must agree with the package's own expectation at the same
    angles, and with a direct count of the optimal assignments."""
    folder, p, gammas, betas = demo
    package, circuit = run_on_ibm.load_package(folder)
    found = run_on_ibm.ideal(package, circuit)

    problem = PRESETS["demo"].build()
    parametric, meta = build_qaoa(problem, p)
    values = {meta["gammas"][k]: gammas[k] for k in range(p)}
    values.update({meta["betas"][k]: betas[k] for k in range(p)})
    costs = problem.all_costs()
    assert found["average_cost"] == pytest.approx(expectation(parametric, costs, values), abs=1e-9)
    assert found["random_average_cost"] == pytest.approx(costs.mean(), abs=1e-12)
    optimal = costs <= problem.brute_force()[0] + TIE_TOLERANCE
    assert found["random_share_at_optimum"] == optimal.mean() == 6 / 64
    probabilities = Statevector.from_instruction(parametric.assign_parameters(values)).probabilities()
    assert found["share_at_optimum"] == pytest.approx(probabilities[optimal].sum(), abs=1e-9)
    assert 0.5 < found["share_at_optimum"] < 0.6


def test_ideal_values_with_an_offset_and_a_full_matrix(symmetric):
    folder, problem, p, gammas, betas = symmetric
    package, circuit = run_on_ibm.load_package(folder)
    found = run_on_ibm.ideal(package, circuit)
    parametric, meta = build_qaoa(problem, p)
    values = {meta["gammas"][k]: gammas[k] for k in range(p)}
    values.update({meta["betas"][k]: betas[k] for k in range(p)})
    costs = problem.all_costs()
    assert found["average_cost"] == pytest.approx(expectation(parametric, costs, values), abs=1e-9)
    assert found["random_average_cost"] == pytest.approx(costs.mean(), abs=1e-12)
    probabilities = Statevector.from_instruction(parametric.assign_parameters(values)).probabilities()
    optimal = costs <= problem.brute_force()[0] + TIE_TOLERANCE
    assert found["share_at_optimum"] == pytest.approx(probabilities[optimal].sum(), abs=1e-9)
    assert found["random_share_at_optimum"] == optimal.mean()


def test_simulate_a_full_matrix_with_an_offset(symmetric, capsys):
    folder, problem, *_ = symmetric
    assert run_script("--simulate", "--seed", 2, "--shots", 4000, "--package", folder, "--out", folder / "r.json") == 0
    capsys.readouterr()
    results = json.loads((folder / "r.json").read_text())
    value, _ = problem.brute_force()
    for row in results["outcomes"]:
        assert row["cost"] == pytest.approx(problem.cost(row["x"]), abs=1e-12)
        assert row["cost"] >= value - 1e-12  # nothing beats the certified optimum
    assert abs(results["average_cost"] - results["ideal"]["average_cost"]) < 0.25


def test_no_ideal_values_for_a_wide_circuit(tmp_path):
    folder = tmp_path / "wide"
    preset = PRESETS["chorded-ring-24"]
    write_package(folder, preset.build(), kind="maxcut", edges=len(preset.edges), solve=False)
    package, circuit = run_on_ibm.load_package(folder)
    assert circuit.num_qubits == 24 > run_on_ibm.IDEAL_LIMIT
    assert run_on_ibm.ideal(package, circuit) is None
    assert package["certified_optimum"] is None


# ----------------------------------------------------------- ideal simulator

def test_simulate_the_demo_graph(demo, capsys):
    folder = demo[0]
    out = folder / "results.json"
    assert run_script("--simulate", "--seed", 7, "--package", folder, "--out", out) == 0
    lines = capsys.readouterr().out.split("\n")
    assert lines[:4] == [
        "Package: MaxCut on 6 nodes, 8 edges",
        "Circuit: QAOA with 3 layers on 6 qubits",
        "Target: an ideal simulator on this computer",
        "Shots: 1024",
    ]
    assert lines[5].startswith("Average cost of the shots (lower is better): -")
    assert lines[6].startswith("  An ideal run of this circuit averages -")
    assert lines[6].endswith("Assignments picked at random average -5.000.")
    assert lines[7].startswith("Shots at the certified optimum of -8.000: ") and lines[7].endswith("%)")
    assert lines[8].endswith("Assignments picked at random give 9.4%.")
    assert "Most frequent results" in lines
    assert lines[-2] == f"Saved: {out}"

    results = json.loads(out.read_text())
    assert results["package"] == "MaxCut on 6 nodes, 8 edges"
    assert results["target"] == "an ideal simulator on this computer"
    assert results["shots"] == 1024 == sum(row["shots"] for row in results["outcomes"])
    assert results["sent"] == {}
    assert results["certified_optimum"] == {"cost": -8.0, "assignment": [0, 1, 0, 0, 1, 0]}
    problem = PRESETS["demo"].build()
    for row in results["outcomes"]:
        assert row["cost"] == pytest.approx(problem.cost(row["x"]), abs=1e-12)
    assert results["shots_at_optimum"] == sum(row["shots"] for row in results["outcomes"] if row["cost"] == -8.0)
    assert results["average_cost"] == pytest.approx(
        sum(row["shots"] * row["cost"] for row in results["outcomes"]) / 1024
    )
    # 1024 shots land near the ideal values: within five standard errors
    ideal = results["ideal"]
    assert abs(results["average_cost"] - ideal["average_cost"]) < 5 * 2.0 / 32
    assert abs(results["shots_at_optimum"] / 1024 - ideal["share_at_optimum"]) < 5 * 0.5 / 32


def test_a_seed_makes_the_simulation_repeatable(demo):
    folder = demo[0]
    for name in ("a.json", "b.json"):
        assert run_script("--simulate", "--seed", 11, "--shots", 300, "--package", folder, "--out", folder / name) == 0
    first, second = (json.loads((folder / name).read_text()) for name in ("a.json", "b.json"))
    assert first["outcomes"] == second["outcomes"]
    assert first["shots"] == 300


def test_the_best_answer_is_read_the_right_way_round(one_sided, capsys):
    """A run on the one-sided problem finds x = [1 1 0 0], not its mirror image."""
    out = one_sided / "r.json"
    assert run_script("--simulate", "--seed", 3, "--package", one_sided, "--out", out) == 0
    text = capsys.readouterr().out
    assert "Best assignment found: x = [1 1 0 0], cost -3.000" in text
    assert "  It is a certified optimum." in text
    assert "As a cut" not in text  # that line is for MaxCut
    results = json.loads(out.read_text())
    assert results["outcomes"][0]["x"] == [1, 1, 0, 0]  # also the most frequent one
    assert results["shots_at_optimum"] > 1024 * 0.4


def test_the_packaged_copy_runs_on_its_own(one_sided, tmp_path):
    """The file in the package is run as a user runs it: as a script, from
    another folder, with this repository off the path. It imports
    nothing of qopt_builder, so it does not miss it."""
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    done = subprocess.run(
        [os.path.abspath(sys.executable), str(one_sided / "run_on_ibm.py"), "--simulate", "--seed", "3", "--out",
         str(tmp_path / "out.json")],
        capture_output=True, text=True, cwd=tmp_path, env=environment, timeout=300,
    )
    assert "qopt_builder" not in (one_sided / "run_on_ibm.py").read_text(encoding="utf-8")
    assert done.returncode == 0, done.stderr
    assert "Best assignment found: x = [1 1 0 0], cost -3.000" in done.stdout
    assert json.loads((tmp_path / "out.json").read_text())["best"]["x"] == [1, 1, 0, 0]


def test_results_are_saved_in_the_package_by_default(one_sided, capsys):
    assert run_script("--simulate", "--seed", 3, "--shots", 50, "--package", one_sided) == 0
    saved = [path.name for path in one_sided.glob("results-*.json")]
    assert len(saved) == 1 and saved[0].endswith("-simulator.json")
    assert f"Saved: {one_sided / saved[0]}" in capsys.readouterr().out


def test_a_package_without_a_certified_optimum(tmp_path, capsys):
    folder = tmp_path / "unsolved"
    write_package(folder, ONE_SIDED, gammas=ONE_SIDED_ANGLES[0], betas=ONE_SIDED_ANGLES[1], solve=False)
    assert run_script("--simulate", "--seed", 3, "--package", folder, "--out", folder / "r.json") == 0
    text = capsys.readouterr().out
    assert "This package has no certified optimum: the problem is too large to solve exactly." in text
    assert "Shots at the certified optimum" not in text and "optimal" not in text
    results = json.loads((folder / "r.json").read_text())
    assert results["shots_at_optimum"] is None and results["ideal"]["share_at_optimum"] is None


# ------------------------------------------------- any other Qiskit backend

def test_a_generic_device(one_sided, capsys):
    """The path for another provider: a backend object handed to run().
    Qiskit's generic simulated device has a limited set of gates and of
    connections, so the circuit is rewritten for it before it is run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # without qiskit-aer the device says it runs without noise
        backend = GenericBackendV2(num_qubits=6, seed=1)
        results = run_on_ibm.run(backend, shots=400, seed=5, folder=one_sided, out=one_sided / "g.json")
    lines = capsys.readouterr().out.split("\n")
    assert lines[2] == "Target: generic_backend_6q, 6 qubits"
    assert lines[3].startswith("Circuit as sent: depth ") and lines[3].endswith(" two-qubit gates")
    assert results["shots"] == 400
    assert results["sent"].keys() == {"depth", "two_qubit_gates"}
    assert results["best"]["x"] == [1, 1, 0, 0]
    assert json.loads((one_sided / "g.json").read_text())["target"] == "generic_backend_6q, 6 qubits"


class RecordingDevice(GenericBackendV2):
    """A generic device that keeps the circuit it is asked to run."""

    def run(self, run_input, **options):
        self.received = run_input
        return super().run(run_input, **options)


def test_a_coupled_problem_on_a_device_with_few_connections(demo):
    """Eight qubits in a line, each wired to its neighbours only. The
    demo graph couples qubits that are not neighbours, so the circuit
    has to be rewritten. What reaches the device must use only its
    gates, on connected qubits, and the results must still be scored
    as the problem's."""
    line = [[i, i + 1] for i in range(7)] + [[i + 1, i] for i in range(7)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = RecordingDevice(num_qubits=8, coupling_map=line, seed=2)
        results = run_on_ibm.run(backend, shots=600, seed=5, folder=demo[0], out=demo[0] / "g.json")

    sent = backend.received
    assert sent.num_qubits == 8
    for item in sent.data:
        assert item.operation.name in backend.operation_names, item.operation.name
        if item.operation.num_qubits == 2:
            assert [sent.find_bit(q).index for q in item.qubits] in line
    assert results["sent"] == {"depth": sent.depth(), "two_qubit_gates": sent.num_nonlocal_gates()}
    assert results["sent"]["two_qubit_gates"] >= 24  # the circuit had 24 RZZ gates before it was rewritten

    assert results["best"]["cost"] == -8.0
    assert results["shots_at_optimum"] / 600 > 0.25  # an ideal run gives 54%, guessing 9%


def test_a_plain_simulator_backend(one_sided):
    results = run_on_ibm.run(BasicSimulator(), shots=200, seed=5, folder=one_sided, out=one_sided / "b.json")
    assert results["target"] == "basic_simulator"
    assert results["best"]["x"] == [1, 1, 0, 0]


# --------------------------------------------------- a package that is wrong

def edit(folder, name, old, new):
    path = folder / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_files_that_do_not_belong_together_are_refused(demo, capsys):
    folder = demo[0]
    edit(folder, "circuit.qasm", "rx(", "rz(")  # one gate swapped for another
    assert run_script("--simulate", "--package", folder) == 2
    assert capsys.readouterr().err == (
        "error: circuit.qasm and problem.json do not belong together: rz is 1 in the circuit, "
        "but problem.json says 0.\n"
    )


def test_a_changed_optimum_is_refused(demo, capsys):
    folder = demo[0]
    edit(folder, "problem.json", '"assignment": [0, 1, 0, 0, 1, 0]', '"assignment": [0, 1, 0, 0, 1, 1]')
    assert run_script("--simulate", "--package", folder) == 2
    assert "the certified optimum does not have the cost it states" in capsys.readouterr().err


@pytest.mark.parametrize("damage, message", [
    (lambda folder: (folder / "problem.json").unlink(), "cannot read problem.json"),
    (lambda folder: (folder / "problem.json").write_text("{not json"), "problem.json is not valid JSON."),
    (lambda folder: (folder / "problem.json").write_text('{"format": "something else"}'),
     "problem.json is not a qopt-builder hardware package, version 1."),
    (lambda folder: (folder / "circuit.qasm").unlink(), "cannot read"),
    (lambda folder: (folder / "circuit.qasm").write_text("OPENQASM 2.0;\nqreg q[6];\nnonsense q[0];\n"),
     "circuit.qasm is not a circuit this script can read"),
])
def test_a_damaged_package_is_reported_in_one_line(demo, capsys, damage, message):
    damage(demo[0])
    assert run_script("--simulate", "--package", demo[0]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ") and message in captured.err


# ------------------------------------------------------------ the arguments

@pytest.mark.parametrize("args, message", [
    ([], "one of the arguments --simulate --rehearse --hardware --job is required"),
    (["--simulate", "--rehearse"], "not allowed with argument"),
    (["--simulate", "--backend", "fake_fez"], "--backend goes with --hardware or --rehearse"),
    (["--job", "abc", "--shots", "10"], "--shots and --seed go with a run, not with --job"),
    (["--simulate", "--shots", "0"], "the number of shots must be 1 or more"),
    (["--simulate", "--shots", "many"], "'many' is not a whole number"),
    (["--simulate", "--seed", "-1"], "the seed must be 0 or more"),
])
def test_arguments_that_are_refused(args, message, capsys):
    with pytest.raises(SystemExit) as stop:
        run_on_ibm.main(args)
    assert stop.value.code == 2
    assert message in capsys.readouterr().err


def test_help_names_the_four_targets(capsys):
    with pytest.raises(SystemExit) as stop:
        run_on_ibm.main(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--simulate", "--rehearse", "--hardware", "--job ID", "--backend NAME", "--shots N", "--seed S"):
        assert flag in text


# --------------------------------------- the steps that talk to IBM, stood in

class StandInService:
    """What the script asks of IBM's service, recorded."""

    def __init__(self, backend=None, counts=None):
        self.asked = []
        self._backend = backend
        self._counts = counts

    def backend(self, name):
        self.asked.append(("backend", name))
        return self._backend

    def least_busy(self, **wanted):
        self.asked.append(("least_busy", wanted))
        return self._backend

    def job(self, job_id):
        self.asked.append(("job", job_id))
        counts = self._counts

        class Bits:
            def get_counts(self):
                return counts

        class Data:
            c = Bits()

        class Result:
            data = Data()

        class Job:
            def result(self):
                return [Result()]

            def backend(self):
                class Device:
                    name = "ibm_stand_in"
                return Device()

        return Job()


def test_hardware_asks_for_the_least_busy_device_with_enough_qubits(one_sided, monkeypatch, capsys):
    service = StandInService(backend=BasicSimulator())
    monkeypatch.setattr(run_on_ibm, "ibm_service", lambda: service)
    assert run_script("--hardware", "--seed", 5, "--package", one_sided, "--out", one_sided / "h.json") == 0
    assert service.asked == [("least_busy", {"operational": True, "simulator": False, "min_num_qubits": 4})]
    assert "Best assignment found: x = [1 1 0 0]" in capsys.readouterr().out


def test_hardware_uses_the_device_that_is_named(one_sided, monkeypatch):
    service = StandInService(backend=BasicSimulator())
    monkeypatch.setattr(run_on_ibm, "ibm_service", lambda: service)
    assert run_script("--hardware", "--backend", "ibm_somewhere", "--shots", 64, "--seed", 5, "--package", one_sided,
                      "--out", one_sided / "h.json") == 0
    assert service.asked == [("backend", "ibm_somewhere")]
    assert json.loads((one_sided / "h.json").read_text())["shots"] == 64


def test_an_earlier_job_is_fetched_and_scored(one_sided, monkeypatch, capsys):
    service = StandInService(counts={"0011": 70, "0010": 20, "1100": 10})
    monkeypatch.setattr(run_on_ibm, "ibm_service", lambda: service)
    assert run_script("--job", "job-123", "--package", one_sided, "--out", one_sided / "j.json") == 0
    assert service.asked == [("job", "job-123")]
    text = capsys.readouterr().out
    assert "Target: ibm_stand_in (job job-123)" in text
    assert "Shots at the certified optimum of -3.000: 70 of 100 (70.0%)" in text
    assert "Best assignment found: x = [1 1 0 0], cost -3.000, in 70 shots" in text
    results = json.loads((one_sided / "j.json").read_text())
    assert results["sent"] == {} and results["shots"] == 100


def test_a_failed_sign_in_is_reported_and_nothing_is_run(one_sided, monkeypatch, capsys):
    def refuse():
        raise run_on_ibm.PackageError("could not sign in to IBM Quantum: no account")

    monkeypatch.setattr(run_on_ibm, "ibm_service", refuse)
    for args in (["--hardware"], ["--job", "job-123"]):
        assert run_script(*args, "--package", one_sided) == 2
        captured = capsys.readouterr()
        assert captured.err == "error: could not sign in to IBM Quantum: no account\n"
        assert captured.out == ""
    assert not list(one_sided.glob("results-*.json"))


def test_a_bad_package_stops_before_any_sign_in(demo, monkeypatch, capsys):
    def never():
        raise AssertionError("the script signed in for a package it cannot use")

    monkeypatch.setattr(run_on_ibm, "ibm_service", never)
    (demo[0] / "problem.json").write_text("{}")
    for args in (["--hardware"], ["--job", "job-123"]):
        assert run_script(*args, "--package", demo[0]) == 2
        assert "is not a qopt-builder hardware package" in capsys.readouterr().err


class StandInSampler:
    """In place of IBM's Sampler: a job that is sent and then fails."""

    def __init__(self, mode=None, options=None):
        self.mode = mode

    def run(self, pubs, shots=None):
        class Job:
            def job_id(self):
                return "job-456"

            def result(self):
                raise RuntimeError("the job was cancelled")

        return Job()


def test_a_job_that_fails_is_reported_with_its_id(one_sided, monkeypatch, capsys):
    import types

    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", types.SimpleNamespace(SamplerV2=StandInSampler))
    monkeypatch.setattr(run_on_ibm, "from_ibm", lambda backend: True)
    monkeypatch.setattr(run_on_ibm, "ibm_service", lambda: StandInService(backend=BasicSimulator()))
    assert run_script("--hardware", "--package", one_sided) == 2
    captured = capsys.readouterr()
    assert captured.out == "Job job-456 sent to basic_simulator. Waiting for the results.\n"
    assert captured.err == "error: job job-456 did not return results: the job was cancelled\n"
    assert not list(one_sided.glob("results-*.json"))


def test_stopping_the_script_says_what_happens_to_the_job(one_sided, monkeypatch, capsys):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_on_ibm, "sample", interrupted)
    assert run_script("--simulate", "--package", one_sided) == 2
    assert "Stopped. A job that was already sent still runs" in capsys.readouterr().err


def test_missing_packages_are_named(one_sided, monkeypatch, capsys):
    def missing(module, install):
        raise run_on_ibm.PackageError(f"this needs {install}: pip install {install}")

    monkeypatch.setattr(run_on_ibm, "need", missing)
    assert run_script("--rehearse", "--package", one_sided) == 2
    assert capsys.readouterr().err == "error: this needs qiskit-ibm-runtime: pip install qiskit-ibm-runtime\n"
    assert run_script("--hardware", "--package", one_sided) == 2
    assert capsys.readouterr().err == "error: this needs qiskit-ibm-runtime: pip install qiskit-ibm-runtime\n"


# --------------------------------- a simulated copy of an IBM device, for real

@needs_ibm
def test_rehearse_on_a_simulated_ibm_device(one_sided, capsys):
    """The whole hardware path except the account: the circuit is laid
    out for a 156-qubit IBM device and run through IBM's Sampler on a
    simulated copy of it, noise included."""
    out = one_sided / "r.json"
    assert run_script("--rehearse", "--seed", 5, "--package", one_sided, "--out", out) == 0
    lines = capsys.readouterr().out.split("\n")
    assert lines[2] == "Target: fake_fez, 156 qubits: a simulated copy of an IBM quantum computer, on this computer"
    assert lines[3].startswith("Circuit as sent: depth ")
    results = json.loads(out.read_text())
    assert results["shots"] == 1024
    assert "job" not in results["sent"]
    assert results["best"]["x"] == [1, 1, 0, 0]
    assert results["outcomes"][0]["x"] == [1, 1, 0, 0]
    assert results["shots_at_optimum"] > 1024 * 0.3


@needs_ibm
def test_rehearse_names_the_copy_three_ways(one_sided):
    for name in ("fake_torino", "ibm_torino", "torino"):
        assert run_on_ibm.rehearsal_backend(name).name == "fake_torino"
    assert run_on_ibm.rehearsal_backend().name == run_on_ibm.REHEARSAL_DEVICE


@needs_ibm
def test_rehearse_with_an_unknown_copy_lists_the_copies(one_sided, capsys):
    assert run_script("--rehearse", "--backend", "ibm_nowhere", "--package", one_sided) == 2
    error = capsys.readouterr().err
    assert error.startswith("error: there is no simulated copy named ibm_nowhere. The copies are: fake_")
    assert "fake_fez" in error and "fake_torino" in error
