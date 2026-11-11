"""Problems written as text, and what a parser must make of them.

The same text formats are read in two places: by the text boxes of the
builder page (parseEdgeList, parseMatrix, parseOffset and parseBudget
in web/qubo.js) and by the command line (qopt_builder/textio.py). The
cases are pinned here once. tests/test_web_parity.py runs every one
through the page's code under Node.js, tests/test_textio.py runs them
through the Python parsers, and tests/test_docs.py checks that the user
guide quotes these messages word for word.

A case with a "limit" is about the page's own cap on problem size,
which the Python package does not have.
"""

from qopt_builder import PRESETS
from validation_cases import neither

PAGE_LIMIT = 64  # MAX_VARIABLES in web/app.js

NO_EDGES = "No edges given. Enter one edge per line: two node numbers and an optional weight."
NO_MATRIX = "No matrix given. Enter one row per line, numbers separated by spaces or commas."


def edge_form(line: int) -> str:
    return f"Line {line}: an edge is two node numbers and an optional weight, for example 0 1 2.5."


def node_form(line: int) -> str:
    return f"Line {line}: node numbers must be whole numbers, 0 or more."


def not_a_number_text(where: str, token: str) -> str:
    return f'{where}: "{token}" is not a number.'


def too_large(where: str, token: str) -> str:
    return f'{where}: "{token}" is too large to use.'


def too_many(n: int, limit: int = PAGE_LIMIT) -> str:
    return (f"This page handles problems of up to {limit} variables, and this one has {n}. "
            "The Python package has no such limit.")


def row_length(line: int, numbers: str, rows: str) -> str:
    return (f"Line {line} has {numbers}, but the matrix has {rows}. A QUBO matrix is square: "
            "every row needs as many numbers as there are rows.")


def not_a_budget(token: str) -> str:
    return f'Qubit budget: "{token}" is not a whole number, 0 or more.'


def ok(value):
    return {"ok": value}


def error(message):
    return {"error": message}


def text(kind, typed, expected, **more):
    return {"kind": kind, "text": typed, **more}, expected


# what is typed -> what the page's parsers make of it
TEXTS = {
    # edge lists
    "edges-plain": text("edges", "0 1\n1 2 2.5\n", ok({"edges": [[0, 1, 1], [1, 2, 2.5]], "n": 3})),
    "edges-commas-blank-lines-crlf": text(
        "edges", "0, 1, 2\r\n\r\n   2,0 , -1  \r\n", ok({"edges": [[0, 1, 2], [2, 0, -1]], "n": 3})),
    "edges-tabs-and-signs": text("edges", "0\t3\t+1.5e1\n3 1 -.5", ok({"edges": [[0, 3, 15], [3, 1, -0.5]], "n": 4})),
    "edges-repeated-and-reversed": text("edges", "0 1\n1 0 2", ok({"edges": [[0, 1, 1], [1, 0, 2]], "n": 2})),
    "edges-comments": text(
        "edges", "# a triangle\n0 1  # first edge\n\n1 2 2.5#weighted\n   # nothing here\n2 0\n",
        ok({"edges": [[0, 1, 1], [1, 2, 2.5], [2, 0, 1]], "n": 3})),
    "edges-only-comments": text("edges", "# nothing\n  # at all\n", error(NO_EDGES)),
    "edges-error-after-comments": text("edges", "# header\n0 1\n# note\n1 x\n", error(not_a_number_text("Line 4", "x"))),
    "edges-form-feed-and-vertical-tab": text("edges", "0\x0b1\x0c2", ok({"edges": [[0, 1, 2]], "n": 2})),
    "edges-no-break-space-is-not-a-separator": text("edges", "0\u00a01", error(edge_form(1))),
    "edges-empty": text("edges", "", error(NO_EDGES)),
    "edges-only-blank-lines": text("edges", "  \n\n\t\n", error(NO_EDGES)),
    "edges-one-number": text("edges", "0 1\n2", error(edge_form(2))),
    "edges-four-numbers": text("edges", "0 1 2 3", error(edge_form(1))),
    "edges-a-word": text("edges", "0 1\n\n1 x", error(not_a_number_text("Line 3", "x"))),
    "edges-hex": text("edges", "0 0x1", error(not_a_number_text("Line 1", "0x1"))),
    "edges-infinity": text("edges", "0 1 Infinity", error(not_a_number_text("Line 1", "Infinity"))),
    "edges-nan": text("edges", "0 1 NaN", error(not_a_number_text("Line 1", "NaN"))),
    "edges-two-points": text("edges", "0 1 1.2.3", error(not_a_number_text("Line 1", "1.2.3"))),
    "edges-overflow": text("edges", "0 1 1e999", error(too_large("Line 1", "1e999"))),
    "edges-fractional-node": text("edges", "0 1.5", error(node_form(1))),
    "edges-negative-node": text("edges", "0 1\n-1 2", error(node_form(2))),
    "edges-self-loop": text("edges", "0 1\n3 3 2", error("Line 2: an edge cannot join node 3 to itself.")),
    "edges-at-the-page-limit": text("edges", "0 63", ok({"edges": [[0, 63, 1]], "n": 64}), limit=PAGE_LIMIT),
    "edges-past-the-page-limit": text("edges", "0 64", error(too_many(65)), limit=PAGE_LIMIT),
    "edges-far-past-the-page-limit": text("edges", "0 1\n5 100000", error(too_many(100001)), limit=PAGE_LIMIT),
    "edges-no-limit-given": text("edges", "0 200", ok({"edges": [[0, 200, 1]], "n": 201})),
    # matrices
    "matrix-plain": text("matrix", "0 1\n1 0", ok([[0, 1], [1, 0]])),
    "matrix-commas-blank-lines": text("matrix", "-1, 2.5\r\n\r\n0,3\n", ok([[-1, 2.5], [0, 3]])),
    "matrix-one-entry": text("matrix", " 7 ", ok([[7]])),
    "matrix-comments": text("matrix", "# Q\n0 1 # row 0\n\n1 0\n", ok([[0, 1], [1, 0]])),
    "matrix-only-comments": text("matrix", "# no rows yet\n", error(NO_MATRIX)),
    "matrix-short-row-after-comments": text("matrix", "# Q\n1 2\n# gap\n3\n", error(row_length(4, "1 number", "2 rows"))),
    "matrix-empty": text("matrix", "", error(NO_MATRIX)),
    "matrix-only-blank-lines": text("matrix", "\n \n", error(NO_MATRIX)),
    "matrix-short-row": text("matrix", "1 2\n\n3", error(row_length(3, "1 number", "2 rows"))),
    "matrix-long-rows": text("matrix", "1 2 3\n4 5 6", error(row_length(1, "3 numbers", "2 rows"))),
    "matrix-one-long-row": text("matrix", "1 2", error(row_length(1, "2 numbers", "1 row"))),
    "matrix-a-word": text("matrix", "1 2\n3 four", error(not_a_number_text("Line 2", "four"))),
    "matrix-overflow": text("matrix", "1e400", error(too_large("Line 1", "1e400"))),
    "matrix-past-the-page-limit": text("matrix", "1 0 0\n0 1 0\n0 0 1", error(too_many(3, 2)), limit=2),
    # the offset box
    "offset-empty": text("offset", "", ok(0)),
    "offset-blank": text("offset", "   ", ok(0)),
    "offset-number": text("offset", " -2.5 ", ok(-2.5)),
    "offset-exponent": text("offset", "1e-3", ok(0.001)),
    "offset-with-line-breaks": text("offset", "\n\t3.5\r\n", ok(3.5)),
    "offset-a-word": text("offset", "abc", error(not_a_number_text("Offset", "abc"))),
    "offset-decimal-comma": text("offset", "1,5", error(not_a_number_text("Offset", "1,5"))),
    "offset-overflow": text("offset", "-1e999", error(too_large("Offset", "-1e999"))),
    # the budget box
    "budget-empty": text("budget", "", ok(None)),
    "budget-blank": text("budget", "  ", ok(None)),
    "budget-number": text("budget", " 12 ", ok(12)),
    "budget-zero": text("budget", "0", ok(0)),
    "budget-leading-zeros": text("budget", "007", ok(7)),
    "budget-negative": text("budget", "-1", error(not_a_budget("-1"))),
    "budget-fraction": text("budget", "2.5", error(not_a_budget("2.5"))),
    "budget-exponent": text("budget", "1e3", error(not_a_budget("1e3"))),
    "budget-a-word": text("budget", "many", error(not_a_budget("many"))),
    "budget-beyond-exact-integers": text("budget", "9" * 20, error(not_a_budget("9" * 20))),
    # text all the way to a problem
    "typed-matrix-neither-symmetric-nor-triangular": text("matrix-problem", "0 1\n3 0", error(neither(0, 1)), offset=""),
    "typed-matrix-bad-offset": text("matrix-problem", "0 1\n1 0", error(not_a_number_text("Offset", "x")), offset="x"),
    "typed-matrix-with-offset": text("matrix-problem", "0 1\n1 0", ok({"matrix": [[0, 1], [1, 0]], "offset": 4}), offset="4"),
    # the exact solver's own bound
    "solver-at-its-bound": ({"kind": "solver", "n": 30}, ok(2 ** 30)),
    "solver-past-its-bound": (
        {"kind": "solver", "n": 31},
        error("The exact solver in the browser stops at n = 30; this problem has n = 31."),
    ),
}

# every preset written out the way the page writes it into its text boxes
for _name, _preset in PRESETS.items():
    _built = _preset.build()
    _expected = ok({"matrix": _built.matrix.tolist(), "offset": _built.offset})
    if _preset.kind == "maxcut":
        _typed = "\n".join(", ".join(str(v) for v in edge) for edge in _preset.edges)
        TEXTS[f"preset-as-text-{_name}"] = text("edges-problem", _typed, _expected, limit=PAGE_LIMIT)
    else:
        _typed = "\n".join(" ".join(str(v) for v in row) for row in _preset.matrix)
        TEXTS[f"preset-as-text-{_name}"] = text(
            "matrix-problem", _typed, _expected, limit=PAGE_LIMIT, offset=str(_preset.offset) if _preset.offset else "")
