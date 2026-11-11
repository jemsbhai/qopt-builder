"""Problems as text: the file formats of the command line, which are
also the formats of the text boxes on the web page.

An edge list has one edge per line: two node numbers and an optional
weight (1 when left out). A matrix has one row per line. Numbers are
separated by spaces, tabs or commas. Blank lines are skipped, and so is
anything after a # on a line. Only plain decimal numbers are read: a
sign, digits, a decimal point, an exponent. Anything else is rejected
with a ValueError that names the line.

parseEdgeList(), parseMatrix() and parseOffset() in web/qubo.js read
the same formats and word their errors the same way.
tests/page_text_cases.py holds the cases that both must pass.
"""

from __future__ import annotations

import math
import re

_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)
_SEPARATORS = re.compile(r"[, \t\v\f\r]+")
_LINE_BREAK = re.compile(r"\r?\n")


def _number(token: str, where: str) -> float:
    if not _NUMBER.fullmatch(token):
        raise ValueError(f'{where}"{token}" is not a number.')
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f'{where}"{token}" is too large to use.')
    return value


def _lines(text: str):
    """(line number, tokens) for each line that holds data."""
    for index, line in enumerate(_LINE_BREAK.split(text)):
        tokens = [token for token in _SEPARATORS.split(line.split("#", 1)[0]) if token]
        if tokens:
            yield index + 1, tokens


def parse_edge_list(text: str) -> tuple[list[tuple[int, int, float]], int]:
    """Read an edge list. Returns (edges, n): the edges as (i, j, weight),
    and n, one more than the largest node number used."""
    edges: list[tuple[int, int, float]] = []
    n = 0
    for line, tokens in _lines(text):
        where = f"Line {line}: "
        if not 2 <= len(tokens) <= 3:
            raise ValueError(where + "an edge is two node numbers and an optional weight, for example 0 1 2.5.")
        values = [_number(token, where) for token in tokens]
        i, j = values[0], values[1]
        if i != math.floor(i) or j != math.floor(j) or i < 0 or j < 0:
            raise ValueError(where + "node numbers must be whole numbers, 0 or more.")
        i, j = int(i), int(j)
        if i == j:
            raise ValueError(where + f"an edge cannot join node {i} to itself.")
        edges.append((i, j, values[2] if len(values) == 3 else 1.0))
        n = max(n, i + 1, j + 1)
    if not edges:
        raise ValueError("No edges given. Enter one edge per line: two node numbers and an optional weight.")
    return edges, n


def parse_matrix(text: str) -> list[list[float]]:
    """Read a matrix, one row per line. Returns the rows, checked to be
    square. QUBO() then checks that the layout is symmetric or triangular."""
    rows: list[list[float]] = []
    line_of: list[int] = []
    for line, tokens in _lines(text):
        rows.append([_number(token, f"Line {line}: ") for token in tokens])
        line_of.append(line)
    if not rows:
        raise ValueError("No matrix given. Enter one row per line, numbers separated by spaces or commas.")
    for row, line in zip(rows, line_of):
        if len(row) != len(rows):
            numbers = "1 number" if len(row) == 1 else f"{len(row)} numbers"
            count = "1 row" if len(rows) == 1 else f"{len(rows)} rows"
            raise ValueError(
                f"Line {line} has {numbers}, but the matrix has {count}. A QUBO matrix is square: "
                "every row needs as many numbers as there are rows."
            )
    return rows


def parse_offset(text: str) -> float:
    """Read a constant offset: one number, or nothing for 0."""
    token = text.strip(" \t\v\f\r\n")
    return 0.0 if token == "" else _number(token, "Offset: ")
