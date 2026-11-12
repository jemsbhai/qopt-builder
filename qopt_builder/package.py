"""Hardware package: one problem and its circuit, ready for a quantum computer.

A package is five text files:

    README.txt        what the package is and how to run it
    problem.json      the problem, its certified optimum, and the angles
    circuit.qasm      the QAOA circuit in OpenQASM 2, angles filled in,
                      ending in a measurement of every qubit
    run_on_ibm.py     runs the circuit through Qiskit and scores the
                      results against the problem
    requirements.txt  the Python packages run_on_ibm.py needs

package_files() makes the five texts; write_folder() and zip_bytes() put
them on disk as a folder or as a zip archive.

The web page builds the same package (web/package.js). The two ports
write every file with the same rules, so one problem with one set of
angles gives the same bytes from both; tests/test_web_parity.py holds
them to that. For that reason nothing here is left to a library whose
output could differ between the ports: numbers are written by
number_text(), JSON line by line, and the zip archive byte by byte.
"""

from __future__ import annotations

import json
import re
import struct
import time
import zlib
from pathlib import Path

from . import __version__
from .build import _ising_terms, build_qaoa
from .problems import QUBO, is_finite_number, whole_number
from .recommend import Recommendation

FORMAT = "qopt-builder hardware package, version 1"
PACKAGE_NAME = "qopt-package"
FILES = ("README.txt", "problem.json", "circuit.qasm", "run_on_ibm.py", "requirements.txt")
GATES = ("h", "rz", "rzz", "rx")  # the order the gates appear in the circuit

README_TEMPLATE = """\
qopt-builder hardware package
=============================

Problem:  {description}
Circuit:  QAOA with {layers} on {qubits}, angles filled in
Angles:   {angles_from}
Made by:  qopt-builder {version}, https://github.com/jemsbhai/qopt-builder

This folder holds one optimisation problem and the quantum circuit built
for it, ready to run through Qiskit: on a simulator, on a simulated copy
of an IBM quantum computer, or on an IBM quantum computer.

Files
-----
README.txt        this file
problem.json      the problem, its certified optimum, and the angles
circuit.qasm      the circuit in OpenQASM 2, with a measurement of every qubit
run_on_ibm.py     runs the circuit and scores the results against the problem
requirements.txt  the Python packages run_on_ibm.py needs

Step 1: try it on a simulator
-----------------------------
    pip install -r requirements.txt
    python run_on_ibm.py --simulate

No account is needed. The circuit runs on an ideal simulator on this
computer, and the script prints the report you will later get from
hardware.

Step 2: rehearse on a simulated IBM quantum computer
----------------------------------------------------
    python run_on_ibm.py --rehearse

No account is needed and no quantum time is used. The script prepares
the circuit for the layout of a real device and runs it on a simulated
copy of that device on this computer, with the device's noise. The copy
is fake_fez unless you name another, for example --backend fake_torino.

Step 3: run it on an IBM quantum computer
-----------------------------------------
You need an IBM Quantum account and your API key saved on this computer.
The account: https://quantum.cloud.ibm.com
Saving the key: https://quantum.cloud.ibm.com/docs/en/guides/save-credentials

    python run_on_ibm.py --hardware

The script signs in with the saved key, picks the least busy device that
has enough qubits, sends one job of 1024 shots, waits for the results
and prints the report. The run uses quantum time from your IBM plan.

    python run_on_ibm.py --hardware --backend NAME    use the device NAME
    python run_on_ibm.py --hardware --shots 4096      measure more often
    python run_on_ibm.py --job ID                     score a job sent earlier

The id of the job is printed as soon as the job is sent. If you stop the
script while it waits, the job still runs. Fetch its results later with
--job and the id.

The report
----------
Each shot is one run of the circuit, measured: it gives a 0 or a 1 for
every variable of the problem, which is one candidate answer. The script
works out the cost of each candidate from problem.json. Lower is better.

  Average cost of the shots
      How good the candidates are on average.
  Shots at the certified optimum
      How often a run gave a best possible answer. qopt-builder
      certifies the optimum by checking every assignment, for problems
      of up to 24 variables. A larger problem has no certified optimum,
      and the report says so.
  Best assignment found
      The best candidate among the shots.

For problems of up to 14 variables two comparisons are printed next to
these: what an ideal run of the same circuit gives, with no noise, and
what picking assignments at random gives. They are the two ends to judge
a real run against. Every run also saves its results as a JSON file in
this folder.

Another quantum computer that works with Qiskit
-----------------------------------------------
circuit.qasm is plain OpenQASM 2 and can be loaded by other tools. From
Python, make a backend object with the provider's own Qiskit package and
hand it to the script:

    import run_on_ibm
    backend = ...                          # from the provider's package
    run_on_ibm.run(backend, shots=1024)

What was tested
---------------
This package was tested on simulators only: Qiskit's ideal simulator, a
generic simulated device from Qiskit, and simulated copies of IBM
devices, with qiskit 2.6.2, qiskit-ibm-runtime 0.52.0 and qiskit-aer
0.18.2. It has not been run on a real quantum computer. The steps that
sign in to IBM, choose a device and fetch a job follow IBM's
documentation as it stood in November 2026.
"""

REQUIREMENTS = """\
# What run_on_ibm.py needs. The versions are the ones it was tested with.
qiskit>=2.6
# for --rehearse, --hardware and --job
qiskit-ibm-runtime>=0.52
# for --rehearse
qiskit-aer>=0.18
"""

RUN_SCRIPT = Path(__file__).with_name("run_on_ibm.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- numbers

def number_text(value) -> str:
    """A finite number as text that reads back as the same number.

    The digits are the shortest that identify the number, written with a
    decimal point: 8.0, -0.25, 1234.5. A number with a decimal exponent
    below -6 or above 15 is written as 1.5e-7 or 1.0e+21. Zero is 0.0
    whatever its sign. The text is valid in JSON and in OpenQASM 2.
    """
    if not is_finite_number(value):
        raise ValueError("A number written into a package must be finite.")
    x = float(value)
    if x == 0.0:
        return "0.0"
    sign = "-" if x < 0 else ""
    mantissa, _, exponent = repr(abs(x)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    # the number is 0.d1d2d3... x 10^point
    point = len(digits) - len(fraction) + int(exponent or 0)
    digits = digits.rstrip("0")
    power = point - 1  # the number is d1.d2d3... x 10^power
    if power < -6 or power > 15:
        return f"{sign}{digits[0]}.{digits[1:] or '0'}e{'-' if power < 0 else '+'}{abs(power)}"
    if point <= 0:
        return f"{sign}0.{'0' * -point}{digits}"
    if point >= len(digits):
        return f"{sign}{digits}{'0' * (point - len(digits))}.0"
    return f"{sign}{digits[:point]}.{digits[point:]}"


def count(number: int, thing: str) -> str:
    """'1 edge', '8 edges'."""
    return f"{number} {thing}" if number == 1 else f"{number} {thing}s"


def describe_problem(kind: str, n: int, edges=None) -> str:
    """One line naming the problem: its kind and its size."""
    if kind == "maxcut":
        return f"MaxCut on {count(n, 'node')}, {count(edges, 'edge')}"
    return f"QUBO on {count(n, 'variable')}"


def exact_cost(problem: QUBO, x) -> float:
    """cost(x) summed term by term in a fixed order, so that both ports
    arrive at the same number down to the last bit."""
    total = problem.offset
    for i in range(problem.n):
        if x[i]:
            for j in range(problem.n):
                if x[j]:
                    total += float(problem.matrix[i, j])
    return total


# ------------------------------------------------------------ the circuit

def circuit_qasm(problem: QUBO, layers: int, gammas, betas, description: str) -> str:
    """The QAOA circuit of qopt_builder/build.py as OpenQASM 2 text, with
    the angles filled in and every qubit measured.

    The gates and their order are those of build_qaoa(): H on every
    qubit, then per layer RZ(2 gamma h_i) and RZZ(2 gamma J_ij) for the
    cost and RX(2 beta) on every qubit for the mixer. The file defines
    rzz itself, because the standard gate file of OpenQASM 2 does not
    have it.
    """
    n = problem.n
    h, J, _ = _ising_terms(problem)
    lines = [
        f"// QAOA circuit made by qopt-builder {__version__}",
        f"// Problem: {description}",
        f"// {count(layers, 'layer')}, angles filled in; problem.json lists them",
        "// Qubit q[i] stands for variable x_i and is measured into bit c[i]",
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }",
        f"qreg q[{n}];",
        f"creg c[{n}];",
    ]
    lines += [f"h q[{i}];" for i in range(n)]
    for layer in range(layers):
        g, b = float(gammas[layer]), float(betas[layer])
        lines.append(f"// layer {layer + 1} of {layers}: cost")
        for i in range(n):
            if abs(h[i]) > 1e-12:
                lines.append(f"rz({number_text(2.0 * g * float(h[i]))}) q[{i}];")
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    lines.append(f"rzz({number_text(2.0 * g * float(J[i, j]))}) q[{i}],q[{j}];")
        lines.append(f"// layer {layer + 1} of {layers}: mixer")
        lines += [f"rx({number_text(2.0 * b)}) q[{i}];" for i in range(n)]
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ the package

def _angles(values, layers: int, name: str) -> list:
    try:
        values = list(values)
    except TypeError:
        raise ValueError(f"{name} must be a list of {count(layers, 'number')}, one per layer.") from None
    if len(values) != layers or not all(is_finite_number(v) for v in values):
        raise ValueError(f"{name} must be a list of {count(layers, 'number')}, one per layer.")
    return [float(v) for v in values]


def package_files(problem: QUBO, *, kind: str, edges=None, recommendation: Recommendation, optimum,
                  layers: int, gammas, betas, angles_from: str) -> dict:
    """The five files of a hardware package, as {name: text}.

    problem         the QUBO
    kind            "maxcut" or "qubo"; with "maxcut", `edges` is the
                    number of edges, for the description
    recommendation  what recommend() returned for the problem
    optimum         the assignment brute_force() returned, or None when
                    the problem was not solved exactly
    layers          the number of QAOA layers built
    gammas, betas   the angles, one of each per layer
    angles_from     one line saying where the angles came from

    Raises ValueError for a refused problem (there is no circuit to
    package) and for angles that do not fit the layers.
    """
    if recommendation.algorithm == "none":
        raise ValueError("The builder refused this problem, so there is no circuit to package.")
    depth = whole_number(layers)
    if depth is None or depth < 1:
        raise ValueError("QAOA depth p must be a positive whole number.")
    layers = depth
    gammas, betas = _angles(gammas, layers, "gammas"), _angles(betas, layers, "betas")
    if kind not in ("maxcut", "qubo"):
        raise ValueError('kind must be "maxcut" or "qubo".')
    n = problem.n
    description = describe_problem(kind, n, edges)

    circuit, _ = build_qaoa(problem, layers)
    counts = circuit.count_ops()

    out = ["{"]
    out.append(f'  "format": {json.dumps(FORMAT)},')
    out.append(f'  "made_by": {json.dumps("qopt-builder " + __version__)},')
    out.append('  "problem": {')
    out.append(f'    "kind": {json.dumps(kind)},')
    out.append(f'    "description": {json.dumps(description)},')
    out.append(f'    "variables": {n},')
    out.append(f'    "offset": {number_text(problem.offset)},')
    out.append('    "matrix": [')
    rows = ["      [" + ", ".join(number_text(v) for v in row) + "]" for row in problem.matrix]
    out.append(",\n".join(rows))
    out.append("    ]")
    out.append("  },")
    if optimum is None:
        out.append('  "certified_optimum": null,')
    else:
        x = [int(v) for v in optimum]
        out.append('  "certified_optimum": {')
        out.append(f'    "cost": {number_text(exact_cost(problem, x))},')
        out.append(f'    "assignment": [{", ".join(str(v) for v in x)}]')
        out.append("  },")
    out.append('  "recommendation": {')
    out.append(f'    "algorithm": {json.dumps(recommendation.algorithm)},')
    out.append(f'    "layers": {recommendation.params["p"]},')
    out.append('    "reasons": [')
    out.append(",\n".join("      " + json.dumps(reason) for reason in recommendation.rationale))
    out.append("    ]")
    out.append("  },")
    out.append('  "qaoa": {')
    out.append(f'    "layers": {layers},')
    out.append(f'    "gammas": [{", ".join(number_text(g) for g in gammas)}],')
    out.append(f'    "betas": [{", ".join(number_text(b) for b in betas)}],')
    out.append(f'    "angles_from": {json.dumps(angles_from)}')
    out.append("  },")
    out.append('  "circuit": {')
    out.append('    "file": "circuit.qasm",')
    out.append(f'    "qubits": {circuit.num_qubits},')
    out.append(f'    "depth": {circuit.depth()},')
    out.append(f'    "gates": {circuit.size()},')
    out += [f'    "{gate}": {counts.get(gate, 0)},' for gate in GATES]
    out.append(f'    "measurements": {n}')
    out.append("  }")
    out.append("}")

    values = {
        "description": description,
        "layers": count(layers, "layer"),
        "qubits": count(n, "qubit"),
        "angles_from": angles_from,
        "version": __version__,
    }
    return {
        "README.txt": re.sub(r"\{(\w+)\}", lambda match: values[match.group(1)], README_TEMPLATE),
        "problem.json": "\n".join(out) + "\n",
        "circuit.qasm": circuit_qasm(problem, layers, gammas, betas, description),
        "run_on_ibm.py": RUN_SCRIPT,
        "requirements.txt": REQUIREMENTS,
    }


# --------------------------------------------------------------- on disk

def write_folder(files: dict, folder) -> None:
    """Write the files into `folder`, creating it if needed. The bytes
    are those of the zip archive: UTF-8, with unix line ends on every
    system."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name).write_bytes(text.encode("utf-8"))


def zip_bytes(files: dict, folder: str = PACKAGE_NAME, when=None) -> bytes:
    """The files as a zip archive that unpacks to one folder.

    The files are stored without compression. `when` is the date and
    time stamped on them, as (year, month, day, hour, minute, second);
    None means now, in local time. The same files, folder name and time
    always give the same bytes.
    """
    if when is None:
        when = time.localtime(time.time())[:6]
    year, month, day, hour, minute, second = when
    if not 1980 <= year <= 2107:
        raise ValueError("A zip archive can only hold dates from 1980 to 2107.")
    dos_time = (hour << 11) | (minute << 5) | (second // 2)
    dos_date = ((year - 1980) << 9) | (month << 5) | day

    body, directory = b"", b""
    for name, text in files.items():
        path = f"{folder}/{name}".encode("utf-8")
        data = text.encode("utf-8")
        flags = 0 if path.isascii() else 0x800  # bit 11: the name is UTF-8
        # the fields a file's own header and its entry in the directory share
        common = struct.pack(
            "<HHHHHIIIHH", 20, flags, 0, dos_time, dos_date, zlib.crc32(data), len(data), len(data), len(path), 0
        )
        directory += (
            b"PK\x01\x02" + struct.pack("<H", 20) + common + struct.pack("<HHHII", 0, 0, 0, 0, len(body)) + path
        )
        body += b"PK\x03\x04" + common + path + data
    end = b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, len(files), len(files), len(directory), len(body), 0)
    return body + directory + end
