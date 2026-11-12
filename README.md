# Quantum Optimisation Circuit Builder

**Describe the problem; get the right circuit, with the reasoning shown.** A recommendation engine plus QAOA circuit factory for quantum optimisation, built with Qiskit. You describe the problem; the builder extracts the features the algorithm decision actually turns on (size, coupling density, hardware qubit budget), recommends an algorithm and parameters with the rationale printed next to the choice, builds the circuit, and solves the instance exactly when that is cheap, so the quantum result is scored against a certified answer. When the honest answer is "do not use a quantum computer for this instance", or the problem exceeds the qubit budget, it says so and refuses rather than silently truncating.

Built by Muntaser Syed for the DeepStation edition of Qiskit Fall Fest 2026.

## Live demo

**https://qopt-builder.vercel.app/**

One-page summary (PDF): [qopt-builder-onepager.pdf](qopt-builder-onepager.pdf)

Guided demo, an animated walkthrough made for watching or recording: **https://qopt-builder.vercel.app/demo.html**. It plays the pipeline on one screen: the recommendation and its reasons, the exact solve, the QAOA circuit assembled gate by gate, with each coupling in the problem flying to the RZZ gate that encodes it, and the run scored against the certified optimum. To record it, open `demo.html?controls=0`, start the screen recorder and press Space: the page waits on the title card until then, and nothing but the demo is on screen. `?autoplay=1` starts on load instead. `&preset=NAME` chooses the problem (names from `python run_demo.py --list`), `&budget=4` shows a refusal, and `&speed=0.5` or `&speed=2` changes the pace. Keys: Space pauses and resumes, Right arrow skips to the end of the current stage, R restarts, H shows or hides the controls, C shows or hides the captions.

New to quantum computing? The site carries a [primer](https://qopt-builder.vercel.app/learn.html), a [glossary](https://qopt-builder.vercel.app/glossary.html) and a [user guide](https://qopt-builder.vercel.app/guide.html), written for programmers who have not met the subject before. Every number those pages quote is recomputed from the code by `tests/test_docs.py`.

Two more pages are for people who want to use the tool, not only watch it: [the command line](https://qopt-builder.vercel.app/cli.html), with every command and what it prints, and [running on hardware](https://qopt-builder.vercel.app/hardware.html). The commands shown on those pages are run by the tests, which require the output printed there.

**Hardware package.** The builder page and the command line can both write a hardware package for a problem: the problem with its certified optimum, the QAOA circuit as OpenQASM 2 with its angles filled in, and a script that runs the circuit through Qiskit and scores the measured answers against the optimum. The script has three targets: an ideal simulator, a simulated copy of an IBM device (through IBM's own client, noise included), and an IBM quantum computer under your own account. The package has been tested on simulators only, including simulated copies of IBM devices. It has not been run on a real quantum computer, and the site says so wherever the package is offered.

The full pipeline runs in the browser: problem intake (MaxCut edge lists or a raw QUBO matrix, plus presets), the recommendation engine with its rationales, the exact solve, the QAOA circuit diagram with gate counts and depth, and a statevector QAOA run (200-point angle scan plus Nelder-Mead refinement) scored against the certified optimum. The JavaScript in `web/` is a dependency-free port of the Python package below.

## Verified results (demo graph, 6 nodes, 8 weighted edges)

Every number here is reproduced by `run_demo.py` and pinned by `tests/test_demo.py`.

| Check | Result |
|---|---|
| Recommendation | exact + QAOA, p=3, both rationales printed |
| Coupling density | 0.53: 8 of the 15 node pairs are coupled (high band, hence p=3) |
| Brute-force optimum | -8.000 at x = [0 1 0 0 1 0] |
| QAOA circuit | 6 qubits, depth 19, 6 free parameters |
| Python demo scan (200 angles) | best expectation -7.308 against the -8.000 optimum |
| Browser demo | scan -6.664, Nelder-Mead refinement -7.187 |
| Qubit budget set below n | recommendation "none": refusal, no circuit built |

## Run it

Python package (Qiskit, Aer, NumPy; see `requirements.txt`):

```bash
pip install -r requirements.txt
python run_demo.py        # intake -> recommendation -> circuit -> scored scan
python tests/test_demo.py # pins the table above; prints "7 passed"
```

Your own problem, from a text file in the same formats as the text boxes on the site:

```bash
python -m qopt_builder run --edges graph.txt     # a graph: one edge per line, "i j weight"
python -m qopt_builder run --qubo matrix.txt     # a QUBO matrix: one row per line
python -m qopt_builder run --preset demo --json  # a built-in problem, reported as JSON
python -m qopt_builder run --help                # every option
```

The exit status is 0 when a circuit was built, 1 when the builder refused the problem, and 2 when the input could not be used.

A hardware package for a problem, and a first run of it on a simulator:

```bash
python -m qopt_builder package --preset demo --out demo-package   # or --edges / --qubo, as for run
cd demo-package
pip install -r requirements.txt
python run_on_ibm.py --simulate    # an ideal simulator; needs no account
python run_on_ibm.py --rehearse    # a simulated copy of an IBM device, noise included; needs no account
python run_on_ibm.py --hardware    # an IBM quantum computer, with the IBM Quantum account saved on your computer
```

The page writes the same package as a download. For the same problem and angles the files in the two are identical byte for byte, which `tests/test_web_parity.py` checks.

Browser demo locally, with any static file server:

```bash
python3 -m http.server --directory web
```

The repository root carries a `vercel.json` that deploys `web/` as a static site (that is how the live demo above is hosted).

## How it works

1. **Intake** (`qopt_builder/problems.py`): the problem as a QUBO. MaxCut on a weighted graph converts exactly; anything else enters as a QUBO matrix directly. Input is checked where it enters. A matrix that is empty, is not square, holds something that is not a finite number, or is neither symmetric nor triangular is rejected with a `ValueError` that says what is wrong and where; so is an edge that names a node outside the graph, joins a node to itself, or has a weight that is not a finite number. The browser port makes the same checks in the same words, and its text boxes report the line at fault. Exact brute-force solve for n up to 24.
2. **Recommend** (`qopt_builder/recommend.py`): stated, conservative rules. Small instances get an exact classical solve that certifies the quantum result, with QAOA built alongside. A problem that exceeds the hardware qubit budget is refused, not truncated. QAOA depth follows coupling density, the fraction of variable pairs with a nonzero coupling (for a graph, its edge density): sparse graphs (density up to 0.15) get p=1, moderate p=2, dense (above 0.5) p=3.
3. **Build** (`qopt_builder/build.py`): the QAOA circuit for the QUBO via its Ising form, angles left as free parameters.

## Status

Hackathon prototype (v0.1.0). The MaxCut path is complete end to end in both Python and the browser port. The hardware package is tested on simulators and on simulated IBM devices, and has not yet been run on a real one. On the roadmap: that run, a VQE ansatz family in the factory, a recommendation that takes the target device's coupling map and native gates into account (today the circuit is fitted to the device when the package is run), problem templates beyond MaxCut (TSP, scheduling, portfolio), and a variational optimiser in the Python demo in place of the angle scan (the browser demo already refines).

## Layout

- `qopt_builder/problems.py`: QUBO model, MaxCut conversion, exact solver for small n.
- `qopt_builder/recommend.py`: the recommendation rules and their rationales.
- `qopt_builder/build.py`: QAOA circuit construction.
- `qopt_builder/simulate.py`: statevector scoring and the angle scan.
- `qopt_builder/cli.py`, `qopt_builder/textio.py`: the command line and its two file formats.
- `qopt_builder/package.py`, `qopt_builder/run_on_ibm.py`: the hardware package and the script it carries.
- `run_demo.py`: the end-to-end MaxCut demo.
- `tests/test_demo.py`: smoke tests pinning the verified results above.
- `web/`: the site (static files, nothing to build): the builder page and its modules (`qubo.js`, `qaoa.js`, `circuit.js`, `package.js`, `app.js`), the guided demo, and the text pages.

## License

MIT.
