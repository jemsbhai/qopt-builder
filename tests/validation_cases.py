"""Inputs for the input checks, shared by two test files.

Every case here can be written as JSON, so the very same input reaches
the Python package (tests/test_validation.py) and, through Node.js, the
web port (tests/test_web_parity.py). A rejected input must draw the
message given here from both; an accepted input must give the result
given here in both. Inputs that exist only in Python, such as numpy
arrays, are tested in test_validation.py alone.

A case is a call plus its arguments:

    {"call": "qubo",    "matrix": ..., "offset": ...}   offset optional
    {"call": "maxcut",  "edges": ..., "n": ...}
    {"call": "cost",    "matrix": ..., "x": ...}
    {"call": "circuit", "matrix": ..., "p": ...}
    {"call": "scan",    "matrix": ..., "p": ..., "samples": ...}
"""

import math

from qopt_builder import QUBO, build_qaoa, maxcut_qubo, scan_angles

NAN, INF = float("nan"), float("inf")

# ------------------------------------------------------------ the wording

NOT_A_TABLE = "QUBO matrix must be a square table of numbers, given as a list of rows."
EMPTY = "QUBO matrix is empty; a problem needs at least one variable."
OFFSET = "QUBO offset must be a finite number."
NODE_COUNT = "Number of nodes must be a positive whole number."
NOT_A_LIST = "Edges must be a list of (i, j) or (i, j, weight) entries."
NOT_AN_ASSIGNMENT = "Assignment must be a list of 0s and 1s, one per variable."
DEPTH = "QAOA depth p must be a positive whole number."
SAMPLES = "Number of samples must be a positive whole number."


def not_square(rows: int, row: int, length: int) -> str:
    return f"QUBO matrix must be square: number of rows is {rows}, but row {row} has length {length}."


def not_a_number(i: int, j: int) -> str:
    return f"QUBO matrix entry [{i}][{j}] is not a number."


def not_finite(i: int, j: int) -> str:
    return f"QUBO matrix entry [{i}][{j}] is not finite."


def neither(i: int, j: int) -> str:
    return (
        f"QUBO matrix must be symmetric or triangular. Entries [{i}][{j}] and [{j}][{i}] "
        "differ, and there are nonzero entries on both sides of the diagonal."
    )


def edge_shape(k: int) -> str:
    return f"edges[{k}] must be (i, j) or (i, j, weight)."


def edge_nodes(k: int) -> str:
    return f"edges[{k}]: node numbers must be whole numbers."


def edge_range(k: int, node: int, n: int) -> str:
    return f"edges[{k}]: node {node} is outside the range 0 to {n - 1}."


def edge_loop(k: int, node: int) -> str:
    return f"edges[{k}] joins node {node} to itself."


def edge_weight(k: int) -> str:
    return f"edges[{k}]: weight must be a finite number."


def wrong_length(n: int, length: int) -> str:
    return f"Assignment must have one entry per variable: this problem has {n}, but x has {length}."


def not_binary(i: int) -> str:
    return f"Assignment entries must be 0 or 1; entry {i} is not."


# ---------------------------------------------------------------- rejected

SMALL = [[0, 1], [1, 0]]  # a valid matrix, for cases that are about something else


def qubo(matrix, **more):
    return {"call": "qubo", "matrix": matrix, **more}


def maxcut(edges, n=3):
    return {"call": "maxcut", "edges": edges, "n": n}


def cost(x):
    return {"call": "cost", "matrix": SMALL, "x": x}


def circuit(p):
    return {"call": "circuit", "matrix": SMALL, "p": p}


def scan(samples, p=1):
    return {"call": "scan", "matrix": SMALL, "p": p, "samples": samples}


REJECTED = {
    # the matrix is not a table of rows
    "matrix-is-a-number": (qubo(5), NOT_A_TABLE),
    "matrix-is-a-flat-list": (qubo([1, 2, 3]), NOT_A_TABLE),
    "matrix-is-text": (qubo("0 1; 1 0"), NOT_A_TABLE),
    "matrix-is-missing": (qubo(None), NOT_A_TABLE),
    "matrix-is-a-mapping": (qubo({"0": [0, 1], "1": [1, 0]}), NOT_A_TABLE),
    "matrix-row-is-a-number": (qubo([[0, 1], 5]), NOT_A_TABLE),
    "matrix-row-is-text": (qubo([[0, 1], "10"]), NOT_A_TABLE),
    # empty problem
    "matrix-has-no-rows": (qubo([]), EMPTY),
    "matrix-has-empty-rows": (qubo([[], []]), EMPTY),
    "matrix-has-one-empty-row": (qubo([[]]), EMPTY),
    # wrong dimensions
    "matrix-ragged": (qubo([[1, 2], [3]]), not_square(2, 1, 1)),
    "matrix-ragged-first-row": (qubo([[1], [2, 3]]), not_square(2, 0, 1)),
    "matrix-wide": (qubo([[1, 2, 3], [4, 5, 6]]), not_square(2, 0, 3)),
    "matrix-tall": (qubo([[1], [2], [3]]), not_square(3, 0, 1)),
    "matrix-one-row-too-long": (qubo([[1, 2]]), not_square(1, 0, 2)),
    "matrix-empty-row-among-others": (qubo([[], [1, 2]]), not_square(2, 0, 0)),
    # entries that are not numbers
    "entry-is-text": (qubo([[0, "1"], ["1", 0]]), not_a_number(0, 1)),
    "entry-is-missing": (qubo([[0, 1], [None, 0]]), not_a_number(1, 0)),
    "entry-is-a-boolean": (qubo([[True, 0], [0, 1]]), not_a_number(0, 0)),
    "entry-is-a-list": (qubo([[0, 0], [0, [1]]]), not_a_number(1, 1)),
    "entry-is-nan": (qubo([[0, NAN], [0, 0]]), not_finite(0, 1)),
    "entry-is-infinite": (qubo([[0, 1], [0, -INF]]), not_finite(1, 1)),
    "entry-text-before-infinity": (qubo([[INF, "x"], [0, 0]]), not_finite(0, 0)),
    "ragged-reported-before-entries": (qubo([["x", 2], [3]]), not_square(2, 1, 1)),
    # neither symmetric nor triangular
    "entries-on-both-sides": (qubo([[0, 1, 0], [0, 0, 0], [2, 0, 0]]), neither(0, 1)),
    "a-conflicting-pair": (qubo([[0, 1], [3, 0]]), neither(0, 1)),
    "symmetric-but-for-one-pair": (qubo([[0, 1, 2], [1, 0, 3], [2, 4, 0]]), neither(1, 2)),
    "triangular-but-for-one-entry": (qubo([[1, 2, 3], [0, 1, 4], [0, 5, 1]]), neither(0, 1)),
    "just-outside-the-tolerance": (qubo([[0, 1.0], [1.00000001, 0]]), neither(0, 1)),
    "entries-not-finite-reported-first": (qubo([[0, 1], [3, NAN]]), not_finite(1, 1)),
    # the offset
    "offset-is-text": (qubo(SMALL, offset="1.5"), OFFSET),
    "offset-is-missing": (qubo(SMALL, offset=None), OFFSET),
    "offset-is-a-boolean": (qubo(SMALL, offset=True), OFFSET),
    "offset-is-nan": (qubo(SMALL, offset=NAN), OFFSET),
    "offset-is-infinite": (qubo(SMALL, offset=INF), OFFSET),
    "offset-is-a-list": (qubo(SMALL, offset=[1]), OFFSET),
    "matrix-reported-before-offset": (qubo([[0, 1], [3, 0]], offset="x"), neither(0, 1)),
    # the number of nodes
    "n-is-zero": (maxcut([], 0), NODE_COUNT),
    "n-is-negative": (maxcut([[0, 1, 1]], -2), NODE_COUNT),
    "n-is-a-fraction": (maxcut([[0, 1, 1]], 2.5), NODE_COUNT),
    "n-is-text": (maxcut([[0, 1, 1]], "3"), NODE_COUNT),
    "n-is-missing": (maxcut([[0, 1, 1]], None), NODE_COUNT),
    "n-is-a-boolean": (maxcut([], True), NODE_COUNT),
    "n-is-nan": (maxcut([[0, 1, 1]], NAN), NODE_COUNT),
    "n-is-infinite": (maxcut([[0, 1, 1]], INF), NODE_COUNT),
    # the edge list
    "edges-missing": (maxcut(None), NOT_A_LIST),
    "edges-is-a-number": (maxcut(5), NOT_A_LIST),
    "edges-is-text": (maxcut("0 1"), NOT_A_LIST),
    "edges-is-a-mapping": (maxcut({"0": [0, 1, 1]}), NOT_A_LIST),
    "n-reported-before-edges": (maxcut(None, 0), NODE_COUNT),
    "edge-too-short": (maxcut([[0]]), edge_shape(0)),
    "edge-too-long": (maxcut([[0, 1, 1], [0, 1, 2, 3]]), edge_shape(1)),
    "edge-is-a-number": (maxcut([5]), edge_shape(0)),
    "edge-is-text": (maxcut(["01"]), edge_shape(0)),
    "edge-is-missing": (maxcut([[0, 1], None]), edge_shape(1)),
    "edge-is-a-mapping": (maxcut([{"i": 0, "j": 1}]), edge_shape(0)),
    "node-is-a-fraction": (maxcut([[0.5, 1, 1]]), edge_nodes(0)),
    "node-is-text": (maxcut([[0, 1], ["0", 1, 1]]), edge_nodes(1)),
    "node-is-missing": (maxcut([[None, 1]]), edge_nodes(0)),
    "node-is-a-boolean": (maxcut([[True, 2]]), edge_nodes(0)),
    "node-is-nan": (maxcut([[0, NAN]]), edge_nodes(0)),
    "node-past-the-end": (maxcut([[0, 3, 1]]), edge_range(0, 3, 3)),
    "node-is-negative": (maxcut([[0, 1], [-1, 2, 1]]), edge_range(1, -1, 3)),
    "node-far-past-the-end": (maxcut([[100000, 0]], 6), edge_range(0, 100000, 6)),
    "edge-joins-a-node-to-itself": (maxcut([[2, 2, 1]]), edge_loop(0, 2)),
    "pair-joins-a-node-to-itself": (maxcut([[0, 1], [1, 1]]), edge_loop(1, 1)),
    "weight-is-text": (maxcut([[0, 1, "2"]]), edge_weight(0)),
    "weight-is-missing": (maxcut([[0, 1, None]]), edge_weight(0)),
    "weight-is-a-boolean": (maxcut([[0, 1, True]]), edge_weight(0)),
    "weight-is-nan": (maxcut([[0, 1, 1], [1, 2, NAN]]), edge_weight(1)),
    "weight-is-infinite": (maxcut([[0, 1, INF]]), edge_weight(0)),
    "node-reported-before-weight": (maxcut([[0, 7, "x"]]), edge_range(0, 7, 3)),
    "whole-number-reported-before-range": (maxcut([[0.5, 9, 1]]), edge_nodes(0)),
    "first-node-out-of-range-is-reported": (maxcut([[5, 9]]), edge_range(0, 5, 3)),
    "range-reported-before-loop": (maxcut([[7, 7]]), edge_range(0, 7, 3)),
    "loop-reported-before-weight": (maxcut([[1, 1, "x"]]), edge_loop(0, 1)),
    "first-bad-edge-is-reported": (maxcut([[0, 1], [1, 1], [0, 9]]), edge_loop(1, 1)),
    # an assignment
    "assignment-is-text": (cost("01"), NOT_AN_ASSIGNMENT),
    "assignment-is-a-number": (cost(5), NOT_AN_ASSIGNMENT),
    "assignment-is-missing": (cost(None), NOT_AN_ASSIGNMENT),
    "assignment-is-a-mapping": (cost({"0": 1, "1": 0}), NOT_AN_ASSIGNMENT),
    "assignment-too-short": (cost([0]), wrong_length(2, 1)),
    "assignment-too-long": (cost([0, 1, 1]), wrong_length(2, 3)),
    "assignment-is-empty": (cost([]), wrong_length(2, 0)),
    "length-reported-before-entries": (cost([2, 0, 0]), wrong_length(2, 3)),
    "assignment-entry-is-two": (cost([0, 2]), not_binary(1)),
    "assignment-entry-is-a-fraction": (cost([0.5, 1]), not_binary(0)),
    "assignment-entry-is-negative": (cost([1, -1]), not_binary(1)),
    "assignment-entry-is-text": (cost(["1", 0]), not_binary(0)),
    "assignment-entry-is-missing": (cost([1, None]), not_binary(1)),
    "assignment-entry-is-nan": (cost([NAN, 1]), not_binary(0)),
    "assignment-entry-is-a-list": (cost([[0], [1]]), not_binary(0)),
    # circuit depth and scan size
    "depth-is-zero": (circuit(0), DEPTH),
    "depth-is-negative": (circuit(-1), DEPTH),
    "depth-is-a-fraction": (circuit(1.5), DEPTH),
    "depth-is-text": (circuit("2"), DEPTH),
    "depth-is-missing": (circuit(None), DEPTH),
    "depth-is-a-boolean": (circuit(True), DEPTH),
    "depth-is-nan": (circuit(NAN), DEPTH),
    "samples-is-zero": (scan(0), SAMPLES),
    "samples-is-negative": (scan(-5), SAMPLES),
    "samples-is-a-fraction": (scan(2.5), SAMPLES),
    "samples-is-text": (scan("10"), SAMPLES),
    "samples-is-missing": (scan(None), SAMPLES),
    "samples-is-a-boolean": (scan(True), SAMPLES),
    "depth-reported-before-samples": (scan(0, p=0), DEPTH),
}

# ---------------------------------------------------------------- accepted


def problem(matrix, offset=0):
    """The result of a "qubo" or "maxcut" call: the stored matrix and offset."""
    return {"matrix": matrix, "offset": offset}


ACCEPTED = {
    # layouts
    "symmetric": (qubo([[1, 2], [2, -1]]), problem([[1, 2], [2, -1]])),
    "upper-triangular": (qubo([[1, 4, 0], [0, -1, 3], [0, 0, 2]]), problem([[1, 4, 0], [0, -1, 3], [0, 0, 2]])),
    "lower-triangular": (qubo([[1, 0, 0], [4, -1, 0], [0, 3, 2]]), problem([[1, 0, 0], [4, -1, 0], [0, 3, 2]])),
    "diagonal": (qubo([[1, 0], [0, -2]]), problem([[1, 0], [0, -2]])),
    "all-zero": (qubo([[0, 0], [0, 0]]), problem([[0, 0], [0, 0]])),
    "one-variable": (qubo([[-1.5]]), problem([[-1.5]])),
    # 0.1 + 0.2 and 0.3 differ in the last bit; that is not an asymmetry
    "symmetric-to-rounding": (qubo([[0, 0.1 + 0.2], [0.3, 0]]), problem([[0, 0.1 + 0.2], [0.3, 0]])),
    "symmetric-within-the-tolerance": (
        qubo([[0, 1000000.0], [1000000.0001, 0]]), problem([[0, 1000000.0], [1000000.0001, 0]]),
    ),
    # an entry of 1e-13 counts as zero, so this is still triangular
    "triangular-with-a-negligible-entry": (qubo([[0, 1], [1e-13, 0]]), problem([[0, 1], [1e-13, 0]])),
    "offset-given": (qubo(SMALL, offset=-2.5), problem(SMALL, -2.5)),
    "offset-left-out": (qubo(SMALL), problem(SMALL, 0)),
    # edges
    "weighted-edges": (maxcut([[0, 1, 2], [1, 2, 0.5]]), problem([[-2, 4, 0], [0, -2.5, 1], [0, 0, -0.5]])),
    "pairs-have-weight-one": (maxcut([[0, 1], [1, 2]]), problem([[-1, 2, 0], [0, -2, 2], [0, 0, -1]])),
    "either-direction": (maxcut([[2, 0, 1.5], [1, 0]]), problem([[-2.5, 2, 3], [0, -1, 0], [0, 0, -1.5]])),
    "repeated-edges-add": (maxcut([[0, 1, 1], [1, 0, 2]], 2), problem([[-3, 6], [0, -3]])),
    "no-edges": (maxcut([], 2), problem([[0, 0], [0, 0]])),
    "whole-numbers-written-as-decimals": (maxcut([[0.0, 1.0, 2.0]], 2.0), problem([[-2, 4], [0, -2]])),
    "negative-and-zero-weights": (maxcut([[0, 1, -1], [1, 2, 0]]), problem([[1, -2, 0], [0, 1, 0], [0, 0, 0]])),
    # assignments
    "assignment-of-numbers": (cost([1, 1]), 2),
    "assignment-of-booleans": (cost([True, False]), 0),
    "assignment-of-decimals": (cost([1.0, 0.0]), 0),
    # depth and samples
    "depth-one": (circuit(1), {"depth": 5, "gates": 7}),
    "depth-written-as-a-decimal": (circuit(2.0), {"depth": 8, "gates": 12}),
    "three-samples": (scan(3), 3),
}


# ------------------------------------------------------------ running them

def run_python(spec):
    """What the Python package does with one case: ("ok", result) or
    ("error", message)."""
    try:
        return "ok", _call(spec)
    except ValueError as error:
        return "error", str(error)


def _call(spec):
    call = spec["call"]
    if call == "qubo":
        built = QUBO(spec["matrix"], offset=spec["offset"]) if "offset" in spec else QUBO(spec["matrix"])
        return {"matrix": built.matrix.tolist(), "offset": built.offset}
    if call == "maxcut":
        built = maxcut_qubo(spec["edges"], spec["n"])
        return {"matrix": built.matrix.tolist(), "offset": built.offset}
    if call == "cost":
        return QUBO(spec["matrix"]).cost(spec["x"])
    if call == "circuit":
        qc, _ = build_qaoa(QUBO(spec["matrix"]), spec["p"])
        return {"depth": qc.depth(), "gates": qc.size()}
    if call == "scan":
        return len(scan_angles(QUBO(spec["matrix"]), spec["p"], samples=spec["samples"], seed=1).trace)
    raise AssertionError(f"unknown call {call!r}")


def to_json(value):
    """The case with NaN and the infinities spelled out, since JSON has no
    way to write them. tests/web_parity.js turns them back."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"$": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    return value
