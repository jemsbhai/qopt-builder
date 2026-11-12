"""The command line: python -m qopt_builder

    python -m qopt_builder run --edges graph.txt
    python -m qopt_builder run --qubo matrix.txt --offset 16
    python -m qopt_builder run --preset demo --budget 4
    python -m qopt_builder package --edges graph.txt --out my-package
    python -m qopt_builder presets

`run` takes one problem through the pipeline and prints each stage: the
recommendation with its reasons, the exact classical solve, the QAOA
circuit, and a coarse angle scan scored against the certified optimum.
With --json it prints the same report as one JSON object.

`package` does the same and then writes a hardware package: a folder, or
a zip archive, with the problem, the circuit with its angles filled in,
and a script that runs the circuit through Qiskit on a simulator or on
an IBM quantum computer (see qopt_builder/package.py).

A problem comes from a text file (see qopt_builder/textio.py for the two
formats, which are also the formats of the web page's text boxes), from
standard input when the file name is -, or from a built-in preset.

Exit status: 0 when a circuit was built (and, for `package`, written), 1
when the builder refused the problem because it needs more qubits than
the budget, 2 when the input could not be used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

from . import __version__
from .build import build_qaoa
from .package import FILES, PACKAGE_NAME, package_files, write_folder, zip_bytes
from .presets import PRESETS, get_preset
from .problems import QUBO, ZERO_TOLERANCE, maxcut_qubo
from .recommend import EXACT_CLASSICAL_LIMIT, Recommendation, recommend
from .simulate import SCAN_SAMPLES, SCAN_SEED, SIMULATION_LIMIT, expectation, scan_angles
from .textio import parse_edge_list, parse_matrix

BUILT, REFUSED, BAD_INPUT = 0, 1, 2  # exit statuses

GATE_ORDER = ("h", "rz", "rzz", "rx")  # the order the gates appear in the circuit


class InputError(Exception):
    """The problem could not be read or used. The message is for the user."""


# ------------------------------------------------------------ the problem

def read_text(path: str) -> str:
    """The text of a file, or of standard input when the name is -."""
    if path == "-":
        return sys.stdin.read()
    try:
        with open(path, encoding="utf-8-sig") as handle:
            return handle.read()
    except UnicodeDecodeError:
        raise InputError(f"{path} is not a text file in UTF-8.") from None
    except OSError as error:
        raise InputError(f"cannot read {path}: {error.strerror or error}.") from None


def load_problem(args) -> tuple[QUBO, dict]:
    """The problem the arguments name, and how to describe it."""
    if args.preset is not None:
        preset = get_preset(args.preset)
        return preset.build(), {"kind": preset.kind, "source": f"preset {preset.name}", "title": preset.title,
                                "edges": len(preset.edges)}
    path = args.edges if args.edges is not None else args.qubo
    source = "standard input" if path == "-" else path
    text = read_text(path)
    try:
        if args.edges is not None:
            edges, n = parse_edge_list(text)
            if args.nodes is not None:
                if args.nodes < n:
                    raise ValueError(f"--nodes {args.nodes} is fewer than the {n} nodes the edges use.")
                n = args.nodes
            return maxcut_qubo(edges, n), {"kind": "maxcut", "source": source, "edges": len(edges)}
        return QUBO(parse_matrix(text), offset=args.offset), {"kind": "qubo", "source": source}
    except ValueError as error:
        raise InputError(f"{source}: {error}") from None
    except MemoryError:
        raise InputError(f"{source}: the problem is too large to hold in memory as a matrix.") from None


# ------------------------------------------------------------- the report

def analyse(problem: QUBO, *, budget=None, layers=None, samples=SCAN_SAMPLES, seed=SCAN_SEED, scan=True):
    """Run the pipeline on one problem.

    Returns (report, circuit). The report is plain data, ready for JSON:
    every stage that ran, and a note for every stage that did not. The
    circuit is the Qiskit QuantumCircuit, or None after a refusal.
    """
    n = problem.n
    pairs = n * (n - 1) // 2
    folded = problem.matrix + problem.matrix.T
    couplings = int(np.count_nonzero(np.abs(folded[np.triu_indices(n, k=1)]) > ZERO_TOLERANCE))
    rec = recommend(problem, budget)
    report = {
        "problem": {"n": n, "couplings": couplings, "pairs": pairs, "density": problem.density(),
                    "offset": problem.offset},
        "recommendation": {"algorithm": rec.algorithm, "params": rec.params, "rationale": rec.rationale},
        "refused": rec.algorithm == "none",
        "exact": None,
        "circuit": None,
        "scan": None,
        "notes": [],
    }
    if report["refused"]:
        return report, None

    if rec.algorithm == "exact+qaoa":
        value, x = problem.brute_force()
        report["exact"] = {"value": value, "x": [int(v) for v in x]}
    else:
        report["notes"].append(
            f"exact solve not run: n={n} is above the exact-solve limit of {EXACT_CLASSICAL_LIMIT}"
        )

    recommended = rec.params["p"]
    circuit, meta = build_qaoa(problem, recommended if layers is None else layers)
    p = meta["p"]
    counts = dict(circuit.count_ops())
    report["circuit"] = {
        "p": p, "recommended_p": recommended, "qubits": circuit.num_qubits, "depth": circuit.depth(),
        "gates": circuit.size(), "counts": {gate: counts[gate] for gate in GATE_ORDER if gate in counts},
        "parameters": circuit.num_parameters,
    }

    if not scan:
        report["notes"].append("angle scan not run: switched off")
    elif n > SIMULATION_LIMIT:
        report["notes"].append(
            f"angle scan not run: n={n} is above the statevector limit of {SIMULATION_LIMIT}"
        )
    else:
        result = scan_angles(problem, p, samples=samples, seed=seed)
        report["scan"] = {
            "samples": len(result.trace), "seed": seed, "value": result.value,
            "gammas": [float(g) for g in result.gammas], "betas": [float(b) for b in result.betas],
            "gap": None if report["exact"] is None else result.value - report["exact"]["value"],
        }
    return report, circuit


def count(number: int, thing: str) -> str:
    """'1 edge', '8 edges'."""
    return f"{number} {thing}" if number == 1 else f"{number} {thing}s"


def describe(report: dict, about: dict, scan_lines: bool = True) -> list[str]:
    """The report as the lines `run` prints. With scan_lines=False the
    lines about the angle scan are left to the caller."""
    problem, rec = report["problem"], report["recommendation"]
    n = problem["n"]
    if "title" in about:
        lines = [f"Problem: {about['title']} ({about['source']})"]
    elif about["kind"] == "maxcut":
        lines = [f"Problem: MaxCut on {count(n, 'node')}, {count(about['edges'], 'edge')} ({about['source']})"]
    else:
        offset = f", offset {problem['offset']:g}" if problem["offset"] else ""
        lines = [f"Problem: QUBO on {count(n, 'variable')}{offset} ({about['source']})"]
    lines.append(
        f"Coupling density: {problem['density']:.2f} ({problem['couplings']} of {count(problem['pairs'], 'pair')})"
    )

    if report["refused"]:
        lines.append("Recommendation: none")
        lines += [f" - {reason}" for reason in rec["rationale"]]
        lines.append("Refused: no circuit built.")
        return lines

    lines.append(f"Recommendation: {rec['algorithm']}, p = {rec['params']['p']}")
    lines += [f" - {reason}" for reason in rec["rationale"]]

    exact = report["exact"]
    if exact is not None:
        lines.append(f"Certified optimum: {exact['value']:.3f} at x = [{' '.join(str(v) for v in exact['x'])}]")
        if about["kind"] == "maxcut":
            lines.append(f"As a cut: weight {-exact['value']:.3f}")
    else:
        lines.append(f"Certified optimum: not computed (n={n} is above the exact-solve limit of {EXACT_CLASSICAL_LIMIT})")

    circuit = report["circuit"]
    if circuit["p"] != circuit["recommended_p"]:
        lines.append(f"Building p = {circuit['p']} as asked; the recommendation was p = {circuit['recommended_p']}.")
    gates = ", ".join(f"{number} {gate.upper()}" for gate, number in circuit["counts"].items())
    lines.append(
        f"QAOA circuit: {count(circuit['qubits'], 'qubit')}, depth {circuit['depth']}, "
        f"{count(circuit['gates'], 'gate')} ({gates}), {count(circuit['parameters'], 'free parameter')}"
    )

    scan = report["scan"]
    if not scan_lines:
        pass
    elif scan is not None:
        lines.append(
            f"Angle scan: best expected cost {scan['value']:.3f} over {count(scan['samples'], 'sample')} "
            f"(seed {scan['seed']})"
        )
        if scan["gap"] is not None:
            lines.append(f"Gap to the certified optimum: {scan['gap']:.3f}")
    elif n > SIMULATION_LIMIT:
        lines.append(f"Angle scan: not simulated (n={n} is above the statevector limit of {SIMULATION_LIMIT})")
    else:
        lines.append("Angle scan: skipped (--no-scan)")
    return lines


# ------------------------------------------------------------ the commands

def command_run(args) -> int:
    problem, about = load_problem(args)
    report, circuit = analyse(
        problem, budget=args.budget, layers=args.layers, samples=args.samples, seed=args.seed,
        scan=not args.no_scan,
    )
    report["problem"] = {"kind": about["kind"], "source": about["source"], **report["problem"]}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n".join(describe(report, about)))
        if args.draw and circuit is not None:
            print_drawing(circuit)
    return REFUSED if report["refused"] else BUILT


def command_package(args) -> int:
    problem, about = load_problem(args)
    given = args.gammas is not None
    report, circuit = analyse(
        problem, budget=args.budget, layers=args.layers, samples=args.samples, seed=args.seed, scan=not given,
    )
    print("\n".join(describe(report, about, scan_lines=not given)))
    if report["refused"]:
        print("No package written.")
        return REFUSED

    n, p = problem.n, report["circuit"]["p"]
    if given:
        for name, values in (("--gammas", args.gammas), ("--betas", args.betas)):
            if len(values) != p:
                raise InputError(
                    f"{name} has {count(len(values), 'number')}, but the circuit has {count(p, 'layer')}: "
                    "give one per layer."
                )
        gammas, betas = args.gammas, args.betas
        angles_from = "given on the command line"
        print("Angles: given on the command line")
        if n > SIMULATION_LIMIT:
            print(
                f"Expected cost at these angles: not simulated "
                f"(n={n} is above the statevector limit of {SIMULATION_LIMIT})"
            )
        else:
            parametric, meta = build_qaoa(problem, p)
            values = {meta["gammas"][k]: gammas[k] for k in range(p)}
            values.update({meta["betas"][k]: betas[k] for k in range(p)})
            value = expectation(parametric, problem.all_costs(), values)
            print(f"Expected cost at these angles: {value:.3f}")
            if report["exact"] is not None:
                print(f"Gap to the certified optimum: {value - report['exact']['value']:.3f}")
    elif report["scan"] is None:
        raise InputError(
            f"no angles to package: n={n} is above the statevector limit of {SIMULATION_LIMIT}, so the scan "
            "cannot choose them. Give your own with --gammas and --betas."
        )
    else:
        scan = report["scan"]
        gammas, betas = scan["gammas"], scan["betas"]
        angles_from = (
            f"command line angle scan: the best of {count(scan['samples'], 'random setting')} (seed {scan['seed']})"
        )

    rec = report["recommendation"]
    files = package_files(
        problem, kind=about["kind"], edges=about.get("edges"),
        recommendation=Recommendation(rec["algorithm"], rec["params"], rec["rationale"]),
        optimum=None if report["exact"] is None else report["exact"]["x"],
        layers=p, gammas=gammas, betas=betas, angles_from=angles_from,
    )
    write_package(files, args.out, args.force)
    print(f"Hardware package written: {args.out} ({', '.join(FILES)})")
    if is_zip(args.out):
        print("Unpack it and start with README.txt: step 1 runs the circuit on a simulator.")
    else:
        print("Start with README.txt in that folder: step 1 runs the circuit on a simulator.")
    return BUILT


def is_zip(path: str) -> bool:
    return path.lower().endswith(".zip")


def write_package(files: dict, out: str, force: bool) -> None:
    """Write the package to `out`: a zip archive when the name ends in
    .zip, otherwise a folder. Nothing that exists is replaced unless
    `force` is set."""
    path = Path(out)
    try:
        if is_zip(out):
            if path.is_dir():
                raise InputError(f"{out} is a folder; give the archive another name.")
            if path.exists() and not force:
                raise InputError(f"{out} already exists. Pass --force to replace it, or choose another --out.")
            path.write_bytes(zip_bytes(files, folder=path.stem or PACKAGE_NAME))
        else:
            if path.exists() and not path.is_dir():
                raise InputError(f"{out} is a file; give the folder another name.")
            if path.is_dir() and any(path.iterdir()) and not force:
                raise InputError(
                    f"{out} already exists and is not empty. Pass --force to write into it, or choose another --out."
                )
            write_folder(files, path)
    except OSError as error:
        raise InputError(f"cannot write {out}: {error.strerror or error}.") from None


def print_drawing(circuit) -> None:
    """The circuit as text. The drawing uses box-drawing characters; a
    console that cannot show them gets question marks in their place
    instead of an error."""
    drawing = str(circuit.draw(output="text", fold=100))
    encoding = sys.stdout.encoding or "utf-8"
    print(drawing.encode(encoding, errors="replace").decode(encoding))


def command_presets(args) -> int:
    print_presets()
    return BUILT


def print_presets() -> None:
    indent = " " * (max(len(name) for name in PRESETS) + 2)
    for name, preset in PRESETS.items():
        print(f"{name:<{len(indent)}}{preset.title}")
        print(textwrap.fill(preset.about, width=96, initial_indent=indent, subsequent_indent=indent))


# ----------------------------------------------------------- the arguments

def whole_number(minimum: int, what: str):
    """An argparse type: a whole number, at least `minimum`."""
    def convert(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{what} must be {minimum} or more")
        return value
    return convert


def finite_number(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number")
    return value


def angle_list(text: str) -> list:
    """An argparse type: numbers separated by commas or spaces."""
    values = [finite_number(token) for token in re.split(r"[,\s]+", text.strip()) if token]
    if not values:
        raise argparse.ArgumentTypeError("no numbers given")
    return values


FILE_NOTE = ("FILE may be - to read standard input. In a file, numbers are separated by spaces, tabs or "
             "commas; blank lines and anything after a # are ignored.")


def add_problem_options(command) -> None:
    """The options `run` and `package` share: the problem, and how it is built and scanned."""
    source = command.add_argument_group("the problem (exactly one)").add_mutually_exclusive_group(required=True)
    source.add_argument("--edges", metavar="FILE",
                        help="a graph for MaxCut: one edge per line as two node numbers and an optional weight")
    source.add_argument("--qubo", metavar="FILE",
                        help="a QUBO matrix, one row per line, symmetric or triangular")
    source.add_argument("--preset", choices=list(PRESETS), metavar="NAME",
                        help="a built-in problem (see the presets command)")
    command.add_argument("--nodes", type=whole_number(1, "the number of nodes"), metavar="N",
                         help="with --edges: the number of nodes, when some at the end have no edges")
    command.add_argument("--offset", type=finite_number, default=0.0, metavar="X",
                         help="with --qubo: a constant added to every cost (default 0)")
    command.add_argument("--budget", type=whole_number(0, "a qubit budget"), metavar="QUBITS",
                         help="hardware qubit budget; a problem larger than this is refused")
    command.add_argument("--layers", type=whole_number(1, "the number of layers"), metavar="P",
                         help="build this many QAOA layers instead of the recommended number")
    command.add_argument("--samples", type=whole_number(1, "the number of samples"), default=SCAN_SAMPLES,
                         metavar="N", help=f"angle settings tried by the scan (default {SCAN_SAMPLES})")
    command.add_argument("--seed", type=whole_number(0, "the seed"), default=SCAN_SEED, metavar="S",
                         help=f"seed of the scan's random settings (default {SCAN_SEED})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qopt_builder",
        description="Recommend, certify and build a QAOA circuit for an optimisation problem.",
        epilog="Exit status: 0 a circuit was built, 1 the builder refused the problem, 2 the input could not be used.",
    )
    parser.add_argument("--version", action="version", version=f"qopt-builder {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    run = commands.add_parser(
        "run",
        help="take one problem through the pipeline",
        description="Recommend an algorithm for one problem, solve it exactly when it is small enough, "
                    "build the QAOA circuit, and score a coarse angle scan against the certified optimum.",
        epilog=FILE_NOTE,
    )
    add_problem_options(run)
    run.add_argument("--no-scan", action="store_true", help="build the circuit and do not simulate it")
    run.add_argument("--json", action="store_true", help="print the report as JSON")
    run.add_argument("--draw", action="store_true", help="also print the circuit as text")
    run.set_defaults(handler=command_run)

    package = commands.add_parser(
        "package",
        help="write a hardware package for one problem",
        description="Take one problem through the pipeline, as `run` does, and write a hardware package: "
                    "the problem with its certified optimum, the QAOA circuit with its angles filled in, and "
                    "a script that runs the circuit through Qiskit on a simulator or on an IBM quantum "
                    "computer. The angles are the best of the scan, unless --gammas and --betas give them.",
        epilog=FILE_NOTE + " Write --gammas=-0.5,0.2 with the equals sign when the first angle is negative.",
    )
    add_problem_options(package)
    package.add_argument("--gammas", type=angle_list, metavar="ANGLES",
                         help="the cost angles to package, one per layer, separated by commas; goes with --betas")
    package.add_argument("--betas", type=angle_list, metavar="ANGLES",
                         help="the mixer angles to package, one per layer, separated by commas")
    package.add_argument("--out", default=PACKAGE_NAME, metavar="PATH",
                         help=f"the folder to write, or a zip archive when the name ends in .zip "
                              f"(default {PACKAGE_NAME})")
    package.add_argument("--force", action="store_true", help="replace a package that is already there")
    package.set_defaults(handler=command_package)

    presets = commands.add_parser("presets", help="list the built-in problems",
                                  description="List the built-in problems by name.")
    presets.set_defaults(handler=command_presets)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("run", "package"):
        if args.nodes is not None and args.edges is None:
            parser.error("--nodes goes with --edges")
        if args.offset != 0.0 and args.qubo is None:
            parser.error("--offset goes with --qubo")
    if args.command == "package" and (args.gammas is None) != (args.betas is None):
        parser.error("--gammas and --betas go together")
    try:
        return args.handler(args)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return BAD_INPUT
