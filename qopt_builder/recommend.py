"""Recommendation engine: problem features -> algorithm + parameters.

Rules are stated, inspectable, and deliberately conservative. Each
recommendation carries its rationale so a user can override any choice
with the reason in view. Features used: problem size n, coupling
density, whether an exact classical answer is cheap, and an optional
hardware qubit budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .problems import QUBO

EXACT_CLASSICAL_LIMIT = 24  # brute force is 2^n; beyond this, do not promise exact


@dataclass
class Recommendation:
    algorithm: str  # "exact" | "qaoa" | "vqe"
    params: dict = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)


def recommend(problem: QUBO, qubit_budget: int | None = None) -> Recommendation:
    n = problem.n
    density = problem.density()
    reasons: list[str] = []

    if n <= EXACT_CLASSICAL_LIMIT:
        reasons.append(
            f"n={n} <= {EXACT_CLASSICAL_LIMIT}: an exact classical solve is cheap "
            "and certifies the quantum result; recommending exact with QAOA built "
            "alongside for comparison."
        )
        base = "exact"
    else:
        base = "quantum"
        reasons.append(f"n={n}: exact enumeration is not feasible (2^{n} states).")

    if qubit_budget is not None and n > qubit_budget:
        reasons.append(
            f"n={n} exceeds the qubit budget of {qubit_budget}: no full-width "
            "circuit fits; the builder will refuse rather than silently truncate."
        )
        return Recommendation("none", {"n": n}, reasons)

    if density > 0.5:
        p = 3
        reasons.append(
            f"coupling density {density:.2f} is high: deeper QAOA (p={p}) earns "
            "its depth on dense graphs."
        )
    elif density > 0.15:
        p = 2
        reasons.append(f"coupling density {density:.2f} is moderate: QAOA p={p}.")
    else:
        p = 1
        reasons.append(
            f"coupling density {density:.2f} is sparse: QAOA p={p} first; depth "
            "buys little on near-tree coupling graphs."
        )

    if base == "exact":
        return Recommendation("exact+qaoa", {"p": p, "also_build": "qaoa"}, reasons)
    return Recommendation("qaoa", {"p": p}, reasons)
