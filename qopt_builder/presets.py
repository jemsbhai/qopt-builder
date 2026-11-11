"""Named preset problems.

A small set of problems that between them cover both input forms (a
MaxCut edge list and a QUBO matrix), all three QAOA depth bands of the
recommender, and sizes from 6 variables up to the exact-solve limit of
24. The certified optimum of each is pinned in tests/test_presets.py,
and `python run_demo.py --all` runs them end to end.

The browser demo carries the same list in web/presets.js; the parity
test checks that the two describe the same problems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problems import QUBO, maxcut_qubo

# The 6-node weighted graph of the demo record.
DEMO_EDGES = (
    (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 2.0),
    (2, 4, 1.0), (3, 4, 1.0), (3, 5, 1.0), (4, 5, 2.0),
)


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    kind: str  # "maxcut": an edge list; "qubo": a matrix and an offset
    n: int
    about: str
    edges: tuple = ()  # kind "maxcut": (i, j, weight) with i < j
    matrix: tuple = ()  # kind "qubo": the rows of Q
    offset: float = 0.0

    def build(self) -> QUBO:
        """The preset as a QUBO, ready for recommend(), brute_force() and build_qaoa()."""
        if self.kind == "maxcut":
            problem = maxcut_qubo(list(self.edges), self.n)
        else:
            problem = QUBO(np.array(self.matrix, dtype=float), offset=self.offset)
        problem.name = self.name
        return problem


def _ring(n: int, step: int) -> tuple:
    """Edges joining every node i to node i + step, around a ring of n nodes."""
    return tuple((min(i, (i + step) % n), max(i, (i + step) % n), 1.0) for i in range(n))


def _vertex_cover(edges, n: int, penalty: float = 2.0):
    """Minimum vertex cover as a QUBO: (matrix rows, offset).

    cost(x) = sum_i x_i + penalty * sum_{(i,j)} (1 - x_i)(1 - x_j)

    x_i = 1 puts node i in the cover. Each chosen node costs 1 and each
    edge left uncovered costs `penalty`. With penalty > 1 it is never
    worth leaving an edge uncovered, so the optimum is a valid cover and
    its cost is the size of the smallest one. Expanding the product
    gives linear terms 1 - penalty * degree(i) on the diagonal, a
    coupling of `penalty` per edge, and the constant penalty * |edges|.
    """
    q = np.zeros((n, n))
    for i in range(n):
        q[i, i] = 1.0
    for i, j, _weight in edges:
        q[i, i] -= penalty
        q[j, j] -= penalty
        q[i, j] += penalty
    return tuple(tuple(float(v) for v in row) for row in q), penalty * len(edges)


_COVER_MATRIX, _COVER_OFFSET = _vertex_cover(DEMO_EDGES, 6)

_ALL = (
    Preset(
        name="demo",
        title="Demo graph: weighted MaxCut, 6 nodes",
        kind="maxcut",
        n=6,
        about=(
            "Two triangles joined by two edges, with weights 1 and 2. A triangle always "
            "leaves one of its edges uncut, so the best cut misses one weight-1 edge in each."
        ),
        edges=DEMO_EDGES,
    ),
    Preset(
        name="tree-14",
        title="Sparse tree: MaxCut, 14 nodes",
        kind="maxcut",
        n=14,
        about=(
            "A binary tree with unit weights. A tree can be two-coloured level by level, so "
            "every edge can be cut. This is the largest size the browser simulates."
        ),
        edges=tuple(((child - 1) // 2, child, 1.0) for child in range(1, 14)),
    ),
    Preset(
        name="ring-10",
        title="Ring: MaxCut, 10 nodes",
        kind="maxcut",
        n=10,
        about="A ring of ten nodes with unit weights. An even ring alternates sides and cuts every edge.",
        edges=_ring(10, 1),
    ),
    Preset(
        name="weighted-8",
        title="Dense weighted graph: MaxCut, 8 nodes",
        kind="maxcut",
        n=8,
        about="Sixteen weighted edges on eight nodes, more than half of all possible pairs.",
        edges=(
            (0, 1, 2.0), (0, 3, 1.0), (1, 2, 3.0), (1, 4, 1.0), (2, 5, 2.0), (3, 4, 2.0),
            (3, 6, 1.0), (4, 5, 1.0), (4, 7, 3.0), (5, 7, 2.0), (6, 7, 1.0), (2, 3, 1.0),
            (5, 6, 2.0), (0, 5, 1.0), (1, 6, 1.0), (3, 7, 2.0),
        ),
    ),
    Preset(
        name="chorded-ring-24",
        title="Chorded ring: MaxCut, 24 nodes",
        kind="maxcut",
        n=24,
        about=(
            "A problem at the exact-solve limit of 24 variables. Every node is joined to its "
            "neighbours one and two steps around a ring: 48 unit edges, full of triangles. "
            "Each triangle leaves an edge uncut, which bounds the cut at 36, and the pattern "
            "1100 repeated reaches it. Too large to simulate, so the builder certifies the "
            "optimum and builds the circuit only."
        ),
        edges=_ring(24, 1) + _ring(24, 2),
    ),
    Preset(
        name="vertex-cover-6",
        title="Vertex cover: QUBO, 6 variables",
        kind="qubo",
        n=6,
        about=(
            "Minimum vertex cover, a different problem on the demo graph: pick the fewest "
            "nodes that touch every edge. Each chosen node costs 1 and each uncovered edge "
            "costs 2, which puts linear terms on the diagonal and a constant offset of 16. "
            "The optimum cost is the size of the smallest cover."
        ),
        matrix=_COVER_MATRIX,
        offset=_COVER_OFFSET,
    ),
    Preset(
        name="dense-qubo-6",
        title="Dense symmetric QUBO, 6 variables",
        kind="qubo",
        n=6,
        about=(
            "A matrix entered in symmetric form, with every pair of variables coupled and "
            "linear terms on the diagonal."
        ),
        matrix=(
            (-2.0, 1.0, 2.0, 0.5, 1.0, 2.0),
            (1.0, -3.0, 1.0, 2.0, 0.5, 1.0),
            (2.0, 1.0, -2.0, 1.0, 2.0, 0.5),
            (0.5, 2.0, 1.0, -3.0, 1.0, 2.0),
            (1.0, 0.5, 2.0, 1.0, -2.0, 1.0),
            (2.0, 1.0, 0.5, 2.0, 1.0, -3.0),
        ),
    ),
)

PRESETS: dict[str, Preset] = {preset.name: preset for preset in _ALL}


def get_preset(name: str) -> Preset:
    """The preset called `name`, or a ValueError that lists the names."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from: {', '.join(PRESETS)}") from None
