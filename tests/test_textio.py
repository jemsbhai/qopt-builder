"""The Python text parsers (qopt_builder/textio.py) on the shared cases.

tests/page_text_cases.py pins what each piece of text must be read as.
The page's parsers pass those cases in tests/test_web_parity.py; here
the Python parsers pass the same ones, so a file that works on the
command line works when pasted into the page, and an error reads the
same in both.
"""

import numpy as np
import pytest

from page_text_cases import TEXTS
from qopt_builder import QUBO, maxcut_qubo, parse_edge_list, parse_matrix, parse_offset


def read(spec):
    """What the Python side makes of one case: {"ok": value} or {"error": message}."""
    kind, typed = spec["kind"], spec["text"]
    try:
        if kind == "edges":
            edges, n = parse_edge_list(typed)
            return {"ok": {"edges": [list(edge) for edge in edges], "n": n}}
        if kind == "matrix":
            return {"ok": parse_matrix(typed)}
        if kind == "offset":
            return {"ok": parse_offset(typed)}
        if kind == "edges-problem":
            built = maxcut_qubo(*parse_edge_list(typed))
        else:
            built = QUBO(parse_matrix(typed), offset=parse_offset(spec["offset"]))
        return {"ok": {"matrix": built.matrix.tolist(), "offset": built.offset}}
    except ValueError as error:
        return {"error": str(error)}


def applies(spec, expected) -> bool:
    """Cases the Python parsers answer: text of a problem or an offset,
    apart from the ones about the page's own size limit."""
    if spec["kind"] not in ("edges", "matrix", "offset", "edges-problem", "matrix-problem"):
        return False
    return not ("limit" in spec and "error" in expected and expected["error"].startswith("This page handles"))


SHARED = [name for name, (spec, expected) in TEXTS.items() if applies(spec, expected)]


def test_most_cases_are_shared():
    assert len(SHARED) >= 50


@pytest.mark.parametrize("name", SHARED)
def test_python_reads_text_as_the_page_does(name):
    spec, expected = TEXTS[name]
    assert read(spec) == expected


def test_edges_come_back_as_whole_numbers_and_a_weight():
    edges, n = parse_edge_list("3 1 2.5\n0 3")
    assert edges == [(3, 1, 2.5), (0, 3, 1.0)] and n == 4
    assert all(type(i) is int and type(j) is int and type(w) is float for i, j, w in edges)


def test_what_is_parsed_is_accepted_by_the_checks():
    """Text that parses always builds: the parser leaves nothing for
    maxcut_qubo() to turn away."""
    edges, n = parse_edge_list("# a path\n0 1\n1 2 -0.5\n2 3 1e1\n")
    problem = maxcut_qubo(edges, n)
    assert problem.n == 4 and np.count_nonzero(problem.matrix) == 7
