"""Named presets: the record of each one, and each optimum checked a second way.

RECORD pins what the pipeline produces for every preset. The tests below
it do not rely on the solver being right: where the answer can be argued
by hand (a tree, an even ring, the triangle bound of the chorded ring, a
vertex cover) the argument is checked directly, and every preset small
enough is also solved by the independent reference enumeration.
"""

from itertools import combinations

import numpy as np
import pytest

from qopt_builder import PRESETS, QUBO, Preset, build_qaoa, get_preset, recommend, scan_angles
from qopt_builder.presets import DEMO_EDGES
from qopt_builder.simulate import SIMULATION_LIMIT
from reference_impl import cut_weight, reference_minimum

# name: variables, coupled pairs, QAOA depth p, certified optimum, optimal x
# (x_0 first), circuit depth, free parameters, best scanned expectation
RECORD = {
    "demo":            (6, 8, 3, -8.0, "010010", 19, 6, "-7.308"),
    "tree-14":         (14, 13, 1, -13.0, "10011110000000", 7, 2, "-9.373"),
    "ring-10":         (10, 10, 2, -10.0, "1010101010", 14, 4, "-7.743"),
    "weighted-8":      (8, 16, 3, -22.0, "01010100", 25, 6, "-16.513"),
    "chorded-ring-24": (24, 48, 2, -36.0, "1100" * 6, 53, 4, None),
    "vertex-cover-6":  (6, 8, 3, 4.0, "110110", 22, 6, "4.617"),
    "dense-qubo-6":    (6, 15, 3, -4.0, "100100", 27, 6, "-0.462"),
}
NAMES = list(RECORD)


def bits(x) -> str:
    return "".join(str(int(v)) for v in x)


def test_the_preset_list():
    assert list(PRESETS) == NAMES
    assert len({preset.title for preset in PRESETS.values()}) == len(PRESETS)
    for name, preset in PRESETS.items():
        assert isinstance(preset, Preset)
        assert preset.name == name
        assert preset.kind in ("maxcut", "qubo")
        assert preset.about


@pytest.mark.parametrize("name", NAMES)
def test_record(name):
    n, coupled, p, optimum, x, depth, params, _scan = RECORD[name]
    problem = get_preset(name).build()
    assert problem.n == n
    assert problem.density() == coupled / (n * (n - 1) / 2)

    rec = recommend(problem)
    assert rec.algorithm == "exact+qaoa"
    assert rec.params["p"] == p

    value, best_x = problem.brute_force()
    assert value == optimum
    assert bits(best_x) == x

    qc, _ = build_qaoa(problem, p)
    assert (qc.num_qubits, qc.depth(), qc.num_parameters) == (n, depth, params)


@pytest.mark.parametrize("name", [name for name in NAMES if RECORD[name][7] is not None])
def test_scan_record(name):
    n, _coupled, p, optimum, _x, _depth, _params, scan = RECORD[name]
    result = scan_angles(get_preset(name).build(), p)
    assert format(result.value, ".3f") == scan
    assert result.value >= optimum  # the certified optimum is a floor


def test_presets_above_the_simulation_limit_have_no_scan_record():
    for name in NAMES:
        assert (RECORD[name][7] is None) == (RECORD[name][0] > SIMULATION_LIMIT)


@pytest.mark.parametrize("name", [name for name in NAMES if RECORD[name][0] <= 14])
def test_optimum_agrees_with_the_reference_enumeration(name):
    problem = get_preset(name).build()
    value, x, _ = reference_minimum(problem.matrix.tolist(), problem.offset)
    assert value == RECORD[name][3]
    assert bits(x) == RECORD[name][4]


# --------------------------------------------------- the preset data itself

@pytest.mark.parametrize("name", [name for name in NAMES if PRESETS[name].kind == "maxcut"])
def test_edge_lists_are_well_formed(name):
    preset = PRESETS[name]
    pairs = [(i, j) for i, j, _ in preset.edges]
    assert all(0 <= i < j < preset.n for i, j in pairs)
    assert len(set(pairs)) == len(pairs)
    assert {node for pair in pairs for node in pair} == set(range(preset.n))  # no isolated node


@pytest.mark.parametrize("name", [name for name in NAMES if PRESETS[name].kind == "maxcut"])
def test_reported_cut_is_what_the_assignment_cuts(name):
    preset = PRESETS[name]
    value, x = preset.build().brute_force()
    assert cut_weight(preset.edges, [int(v) for v in x]) == -value


@pytest.mark.parametrize("name", [name for name in NAMES if PRESETS[name].kind == "qubo"])
def test_matrices_are_well_formed(name):
    preset = PRESETS[name]
    matrix = np.array(preset.matrix)
    assert matrix.shape == (preset.n, preset.n)
    assert np.array_equal(matrix, np.triu(matrix)) or np.array_equal(matrix, matrix.T)


def test_build_returns_a_fresh_named_problem():
    first, second = get_preset("demo").build(), get_preset("demo").build()
    assert isinstance(first, QUBO) and first is not second
    assert first.name == "demo"
    first.matrix[0, 0] = 99.0  # changing one copy leaves the preset alone
    assert get_preset("demo").build().matrix[0, 0] == second.matrix[0, 0]


def test_unknown_preset_names_the_choices():
    with pytest.raises(ValueError) as error:
        get_preset("nope")
    assert "unknown preset 'nope'" in str(error.value)
    assert all(name in str(error.value) for name in NAMES)


# ------------------------------------------- the optima, argued a second way

def test_demo_graph():
    """Two triangles: each leaves an edge uncut, so at most 10 - 2 = 8.
    The graph is the one tests/test_demo.py records."""
    import test_demo

    assert PRESETS["demo"].edges == DEMO_EDGES
    assert list(DEMO_EDGES) == test_demo.EDGES and PRESETS["demo"].n == test_demo.N
    assert sum(w for _, _, w in DEMO_EDGES) == 10.0
    assert RECORD["demo"][3] == -8.0


@pytest.mark.parametrize("name", ["tree-14", "ring-10"])
def test_two_colourable_graphs_cut_every_edge(name):
    """A tree and an even ring are bipartite, and no cut can exceed the
    total edge weight."""
    preset = PRESETS[name]
    assert RECORD[name][3] == -sum(w for _, _, w in preset.edges)


def test_tree_14_is_a_tree():
    edges = PRESETS["tree-14"].edges
    assert len(edges) == 13  # n - 1 edges ...
    reached = {0}
    for parent, child, _ in edges:  # ... each joining a new node to one already reached
        assert parent in reached and child not in reached
        reached.add(child)
    assert reached == set(range(14))


def test_chorded_ring_bound():
    """Why 36 is the maximum, from the premises of the argument.

    The 24 triangles {i, i+1, i+2} each contain two ring edges and one
    chord; every ring edge lies in exactly two of them and every chord
    in exactly one. A triangle has at most two cut edges, so with R cut
    ring edges and C cut chords, 2R + C <= 48. With C <= 24 this gives
    R + C <= 36.
    """
    n = 24
    edges = {(i, j) for i, j, _ in PRESETS["chorded-ring-24"].edges}
    assert len(edges) == 48

    def edge(a, b):
        a, b = a % n, b % n
        return (min(a, b), max(a, b))

    ring = {edge(i, i + 1) for i in range(n)}
    chords = {edge(i, i + 2) for i in range(n)}
    assert ring | chords == edges and not ring & chords

    membership = {e: 0 for e in edges}
    for i in range(n):
        for e in (edge(i, i + 1), edge(i + 1, i + 2), edge(i, i + 2)):
            membership[e] += 1
    assert all(membership[e] == 2 for e in ring)
    assert all(membership[e] == 1 for e in chords)

    best = max(r + c for r in range(25) for c in range(25) if 2 * r + c <= 48)
    assert best == 36 == -RECORD["chorded-ring-24"][3]


def test_vertex_cover_is_valid_and_minimal():
    """The optimal x is a cover, its size is the optimum, and no three
    nodes touch all eight edges."""
    preset = PRESETS["vertex-cover-6"]
    value, x = preset.build().brute_force()
    cover = {i for i, v in enumerate(x) if v}
    assert all(i in cover or j in cover for i, j, _ in DEMO_EDGES)
    assert len(cover) == value == 4.0
    for smaller in combinations(range(6), 3):
        assert not all(i in smaller or j in smaller for i, j, _ in DEMO_EDGES)


def test_vertex_cover_cost_is_size_plus_penalties():
    """cost(x) = (nodes chosen) + 2 * (edges left uncovered), for every x."""
    problem = PRESETS["vertex-cover-6"].build()
    assert problem.offset == 16.0
    for index in range(1 << 6):
        x = [(index >> i) & 1 for i in range(6)]
        uncovered = sum(1 for i, j, _ in DEMO_EDGES if not x[i] and not x[j])
        assert problem.cost(x) == sum(x) + 2 * uncovered
