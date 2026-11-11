"""End-to-end demo: MaxCut on a 6-node graph, and the named presets.

Pipeline: problem -> features -> recommendation -> exact classical
solve -> QAOA circuit -> a coarse angle scan scored by exact statevector
expectation, reported against the certified optimum.

    python run_demo.py                  the 6-node demo graph
    python run_demo.py --list           the named presets
    python run_demo.py --preset NAME    one preset
    python run_demo.py --all            every preset
    python run_demo.py --budget 4       with a hardware qubit budget (here: a refusal)
"""

import argparse
import textwrap

from qopt_builder import PRESETS, build_qaoa, get_preset, recommend, scan_angles
from qopt_builder.presets import DEMO_EDGES
from qopt_builder.simulate import SIMULATION_LIMIT

EDGES = list(DEMO_EDGES)
N = 6


def run(problem, qubit_budget=None) -> None:
    """Run the pipeline on one problem and print each stage."""
    rec = recommend(problem, qubit_budget)
    print("Recommendation:", rec.algorithm, rec.params)
    for r in rec.rationale:
        print(" -", r)
    if rec.algorithm == "none":
        print("No circuit built.")
        return

    opt = None
    if rec.algorithm == "exact+qaoa":
        opt, x = problem.brute_force()
        print(f"Brute-force optimum: {opt:.3f} at x=[{' '.join(str(int(v)) for v in x)}]")
    else:
        print("Brute-force optimum: not computed (beyond the exact-solve limit)")

    p = rec.params["p"]
    qc, _meta = build_qaoa(problem, p)
    print(f"QAOA circuit: {qc.num_qubits} qubits, depth {qc.depth()}, params {qc.num_parameters}")

    if problem.n > SIMULATION_LIMIT:
        print(f"Best scanned expectation: not simulated (n={problem.n} is above the "
              f"statevector limit of {SIMULATION_LIMIT})")
        return
    best = scan_angles(problem, p).value
    reference = f"optimum {opt:.3f}" if opt is not None else "no certified optimum"
    print(f"Best scanned expectation: {best:.3f} ({reference})")


def list_presets() -> None:
    indent = " " * (max(len(name) for name in PRESETS) + 2)
    for name, preset in PRESETS.items():
        print(f"{name:<{len(indent)}}{preset.title}")
        print(textwrap.fill(preset.about, width=96, initial_indent=indent, subsequent_indent=indent))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Recommend, certify, build and score QAOA for a preset problem.",
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--preset", choices=list(PRESETS), metavar="NAME",
                       help="run one named preset (see --list)")
    which.add_argument("--all", action="store_true", help="run every preset")
    which.add_argument("--list", action="store_true", help="list the presets and exit")
    parser.add_argument("--budget", type=int, metavar="QUBITS",
                        help="hardware qubit budget; a problem larger than this is refused")
    args = parser.parse_args(argv)

    if args.list:
        list_presets()
        return
    if args.preset is None and not args.all:
        run(get_preset("demo").build(), args.budget)  # the demo record, no header
        return
    names = list(PRESETS) if args.all else [args.preset]
    for k, name in enumerate(names):
        preset = get_preset(name)
        if k:
            print()
        print(f"[{name}] {preset.title}")
        run(preset.build(), args.budget)


if __name__ == "__main__":
    main()
