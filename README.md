# Quantum Optimisation Circuit Builder

**Describe the problem; get the right circuit, with the reasoning shown.** A recommendation engine plus circuit factory for quantum optimisation, built with Qiskit for the DeepStation edition of Qiskit Fall Fest 2026.

## The problem it solves

Picking a quantum optimisation algorithm is a specialist decision that most potential users cannot make: QAOA or VQE or exact classical, at what depth, on how many qubits, and when is the honest answer "do not use a quantum computer for this instance"? This builder makes that decision inspectable. You describe the problem; it extracts the features the decision actually turns on (size, coupling density, hardware budget), recommends an algorithm and parameters with the rationale printed next to the choice, and builds the circuit.

## Pipeline

1. **Intake** (`qopt_builder/problems.py`): the problem as a QUBO. MaxCut on a weighted graph converts exactly; anything else enters as a QUBO matrix directly.
2. **Recommend** (`qopt_builder/recommend.py`): stated, conservative rules. Small instances get an exact classical solve that certifies the quantum result. A problem that exceeds the hardware qubit budget is refused, not silently truncated. QAOA depth follows coupling density: sparse graphs start at p=1, dense graphs earn p=3.
3. **Build** (`qopt_builder/build.py`): the QAOA circuit for the QUBO, via its Ising form, with angles left as free parameters.

## Status

Hackathon prototype (v0.1.0). The MaxCut path is complete end to end: `run_demo.py` takes a 6-node graph from intake to a built circuit and scores a coarse angle scan against the brute-force optimum. On the roadmap: a VQE ansatz family in the factory, hardware-aware transpilation targets (coupling map and native gates), problem templates beyond MaxCut (TSP, scheduling, portfolio), and a proper variational optimiser in place of the demo's angle scan.

## Layout

- `qopt_builder/problems.py`: QUBO model, MaxCut conversion, exact solver for small n.
- `qopt_builder/recommend.py`: the recommendation rules and their rationales.
- `qopt_builder/build.py`: QAOA circuit construction.
- `run_demo.py`: the end-to-end MaxCut demo.

## License

MIT.
