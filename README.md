# Quantum Optimisation Circuit Builder

**Describe the problem; get the right circuit, with the reasoning shown.** A recommendation engine plus QAOA circuit factory for quantum optimisation, built with Qiskit. You describe the problem; the builder extracts the features the algorithm decision actually turns on (size, coupling density, hardware qubit budget), recommends an algorithm and parameters with the rationale printed next to the choice, builds the circuit, and solves the instance exactly when that is cheap, so the quantum result is scored against a certified answer. When the honest answer is "do not use a quantum computer for this instance", or the problem exceeds the qubit budget, it says so and refuses rather than silently truncating.

Built by Muntaser Syed for the DeepStation edition of Qiskit Fall Fest 2026.

## Live demo

**https://qopt-builder.vercel.app/**

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

Browser demo locally, with any static file server:

```bash
python3 -m http.server --directory web
```

The repository root carries a `vercel.json` that deploys `web/` as a static site (that is how the live demo above is hosted).

## How it works

1. **Intake** (`qopt_builder/problems.py`): the problem as a QUBO. MaxCut on a weighted graph converts exactly; anything else enters as a QUBO matrix directly. Exact brute-force solve for n up to 24.
2. **Recommend** (`qopt_builder/recommend.py`): stated, conservative rules. Small instances get an exact classical solve that certifies the quantum result, with QAOA built alongside. A problem that exceeds the hardware qubit budget is refused, not truncated. QAOA depth follows coupling density, the fraction of variable pairs with a nonzero coupling (for a graph, its edge density): sparse graphs (density up to 0.15) get p=1, moderate p=2, dense (above 0.5) p=3.
3. **Build** (`qopt_builder/build.py`): the QAOA circuit for the QUBO via its Ising form, angles left as free parameters.

## Status

Hackathon prototype (v0.1.0). The MaxCut path is complete end to end in both Python and the browser port. On the roadmap: a VQE ansatz family in the factory, hardware-aware transpilation targets (coupling map and native gates), problem templates beyond MaxCut (TSP, scheduling, portfolio), and a variational optimiser in the Python demo in place of the angle scan (the browser demo already refines).

## Layout

- `qopt_builder/problems.py`: QUBO model, MaxCut conversion, exact solver for small n.
- `qopt_builder/recommend.py`: the recommendation rules and their rationales.
- `qopt_builder/build.py`: QAOA circuit construction.
- `run_demo.py`: the end-to-end MaxCut demo.
- `tests/test_demo.py`: smoke tests pinning the verified results above.
- `web/`: the browser demo (static site; `qubo.js`, `qaoa.js`, `circuit.js`, `app.js`).

## License

MIT.
