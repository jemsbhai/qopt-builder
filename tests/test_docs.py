"""The learning pages (primer, glossary, user guide) against the code.

The pages quote numbers, messages and names. Every quoted number is
marked in the HTML with a data-fact attribute, and this file recomputes
each one from the package, so a page cannot state a figure the code does
not produce. The values that come from the browser's own code are the
ones in tests/pinned_record.py, which tests/test_web_parity.py checks
under Node.js. Links are followed, the glossary is checked for order,
the messages the guide quotes must be ones the page can produce, and
the project's wording rules are applied to everything a visitor reads.
"""

import html
import os
import re
from itertools import combinations

import numpy as np
import pytest

from page_text_cases import PAGE_LIMIT, TEXTS
from pinned_record import WEB_DEMO
from qopt_builder import PRESETS, build_qaoa, maxcut_qubo, recommend, scan_angles
from qopt_builder.presets import DEMO_EDGES
from qopt_builder.problems import TIE_TOLERANCE
from qopt_builder.recommend import EXACT_CLASSICAL_LIMIT
from qopt_builder.simulate import SCAN_SAMPLES, SIMULATION_LIMIT
from validation_cases import REJECTED

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB = os.path.join(ROOT, "web")
DOCS = ["learn.html", "glossary.html", "guide.html"]
PAGES = sorted(name for name in os.listdir(WEB) if name.endswith(".html"))


def read(name: str) -> str:
    with open(os.path.join(WEB, name), encoding="utf-8") as handle:
        return handle.read()


def visible_text(page: str) -> str:
    """The words a visitor reads: tags removed, entities resolved."""
    return html.unescape(re.sub(r"<[^>]+>", " ", page))


# ---------------------------------------------------------------- the facts

def compute_facts() -> dict:
    problem = maxcut_qubo(DEMO_EDGES, 6)
    costs = problem.all_costs()
    optimum, _ = problem.brute_force()
    optimal = int(np.count_nonzero(costs <= optimum + TIE_TOLERANCE))
    p = recommend(problem).params["p"]
    qc, _ = build_qaoa(problem, p)
    counts = dict(qc.count_ops())
    triangle, _ = build_qaoa(maxcut_qubo([(0, 1), (1, 2), (0, 2)], 3), 1)
    mantissa, exponent = f"{2 ** 100:.1e}".split("e+")
    return {
        # the demo graph and its record
        "demo-nodes": str(problem.n),
        "demo-edges": str(len(DEMO_EDGES)),
        "demo-weight": f"{sum(w for _, _, w in DEMO_EDGES):g}",
        "demo-pairs": str(problem.n * (problem.n - 1) // 2),
        "demo-density": f"{problem.density():.2f}",
        "demo-density-3": f"{problem.density():.3f}",
        "demo-assignments": str(len(costs)),
        "demo-optimum": f"{optimum:.3f}",
        "demo-cut": f"{-optimum:g}",
        "demo-cut-3": f"{-optimum:.3f}",
        "demo-optimal-count": str(optimal),
        "demo-mean": f"{costs.mean():.3f}",
        "chance-start": f"{100 * optimal / len(costs):.1f}%",
        "demo-p": str(p),
        "demo-depth": str(qc.depth()),
        "demo-gates": str(qc.size()),
        "demo-h": str(counts["h"]),
        "demo-rzz": str(counts["rzz"]),
        "demo-rx": str(counts["rx"]),
        "demo-params": str(qc.num_parameters),
        "python-scan": f"{scan_angles(problem, p).value:.3f}",
        # the triangle of the primer's circuit figure
        "tri-gates": str(triangle.size()),
        "tri-depth": str(triangle.depth()),
        # limits and rules
        "exact-limit": str(EXACT_CLASSICAL_LIMIT),
        "simulation-limit": str(SIMULATION_LIMIT),
        "page-limit": str(PAGE_LIMIT),
        "scan-samples": str(SCAN_SAMPLES),
        "band-low": "0.15",   # checked against recommend() below
        "band-high": "0.5",
        "vertex-cover-offset": f"{PRESETS['vertex-cover-6'].offset:g}",
        # arithmetic
        "pow-6": f"{2 ** 6:,}",
        "pow-14": f"{2 ** SIMULATION_LIMIT:,}",
        "pow-24": f"{2 ** EXACT_CLASSICAL_LIMIT:,}",
        "pow-50": f"{2 ** 50:,}",
        "pow-100-mantissa": mantissa,
        "pow-100-exponent": exponent,
        "statevector-50-petabytes": f"{2 ** 50 * 16 / 1e15:.0f}",
        # what the browser page shows, pinned and checked under Node.js
        **WEB_DEMO,
    }


FACTS = compute_facts()
QUOTED = [
    (page, name, value.strip())
    for page in DOCS
    for name, value in re.findall(r'data-fact="([^"]+)"[^>]*>([^<]*)<', read(page))
]


def test_the_pages_quote_facts():
    assert len(QUOTED) > 60
    assert {page for page, _, _ in QUOTED} == set(DOCS)


@pytest.mark.parametrize("page, name, value", QUOTED, ids=[f"{page}:{name}" for page, name, _ in QUOTED])
def test_every_quoted_number_is_what_the_code_produces(page, name, value):
    assert name in FACTS, f"{page} quotes a fact this test does not know: {name}"
    assert value == FACTS[name]


def test_every_fact_is_quoted_somewhere():
    """No entry of the table above is left over from an earlier wording."""
    assert {name for _, name, _ in QUOTED} == set(FACTS)


def test_the_density_bands_quoted_are_the_ones_recommend_applies():
    def layers(n, edges):
        pairs = list(combinations(range(n), 2))[:edges]
        return recommend(maxcut_qubo(pairs, n)).params["p"]

    low, high = float(FACTS["band-low"]), float(FACTS["band-high"])
    assert 18 / 120 == low and layers(16, 18) == 1 and layers(16, 19) == 2
    assert 3 / 6 == high and layers(4, 3) == 2 and layers(4, 4) == 3


def test_the_bars_are_drawn_to_the_numbers_beside_them():
    widths = re.findall(r'class="fill[^"]*" style="width: ([0-9.]+%)"', read("learn.html"))
    assert widths == [FACTS["chance-start"], FACTS["web-success"]]


def test_the_graph_figure_shows_the_certified_best_cut():
    """Every edge of the demo graph is in the picture with its weight,
    the nodes are coloured by the certified optimum, and the edges drawn
    as cut are the ones that optimum cuts."""
    figure = read("learn.html")
    figure = figure[figure.index('<figure id="fig-graph">'):]
    figure = figure[:figure.index("</figure>")]
    problem = maxcut_qubo(DEMO_EDGES, 6)
    optimum, x = problem.brute_force()
    drawn = {
        (int(i), int(j)): (float(w), cut == "1")
        for i, j, w, cut in re.findall(r'data-edge="(\d+)-(\d+)" data-weight="([^"]+)" data-cut="(\d)"', figure)
    }
    assert drawn == {(i, j): (w, bool(x[i] != x[j])) for i, j, w in DEMO_EDGES}
    sides = {int(i): int(side) for i, side in re.findall(r'data-node="(\d+)" data-side="(\d)"', figure)}
    assert sides == {i: int(x[i]) for i in range(6)}
    assert sum(w for w, cut in drawn.values() if cut) == -optimum


# ------------------------------------------------------------------- links

HREFS = [(page, target) for page in PAGES for target in re.findall(r'href="([^"]+)"', read(page))]
IDS = {page: set(re.findall(r'\sid="([^"]+)"', read(page))) for page in PAGES}


def test_ids_are_unique_within_a_page():
    for page in PAGES:
        found = re.findall(r'\sid="([^"]+)"', read(page))
        assert len(found) == len(set(found)), page


@pytest.mark.parametrize("page, target", sorted(set(HREFS)), ids=[f"{p}->{t}" for p, t in sorted(set(HREFS))])
def test_every_link_leads_somewhere(page, target):
    if re.match(r"^[a-z]+:", target):
        assert target.startswith("https://"), f"{page} links to {target}"
        return
    file, _, anchor = target.partition("#")
    file = file.split("?")[0] or page
    assert os.path.isfile(os.path.join(WEB, file)), f"{page} links to {file}, which is not in web/"
    if anchor:
        assert file in IDS and anchor in IDS[file], f"{page} links to #{anchor}, which {file} does not have"


def test_each_learning_page_links_to_the_others_and_to_the_builder():
    for page in DOCS:
        targets = {target for source, target in HREFS if source == page}
        assert {"index.html", "index.html#app", "demo.html", "learn.html", "glossary.html", "guide.html"} <= targets, page
        assert "<script" not in read(page), f"{page} is a text page and loads no script"


def test_the_builder_points_to_the_learning_pages():
    targets = {target for source, target in HREFS if source == "index.html"}
    assert {"learn.html", "glossary.html", "guide.html"} <= targets
    # each result card, and both input headings, carry a link to their section of the guide
    assert len(re.findall(r'<a class="help" href="guide\.html#[a-z-]+"', read("index.html"))) == 8


# ---------------------------------------------------------------- glossary

TERMS = re.findall(r'<dt id="([^"]+)">([^<]+)</dt>', read("glossary.html"))


def test_glossary_terms_are_in_alphabetical_order():
    names = [name for _, name in TERMS]
    assert len(names) >= 45
    assert names == sorted(names, key=str.lower)
    assert all(term_id.startswith("term-") for term_id, _ in TERMS)


def test_glossary_letters_match_the_terms_under_them():
    page = read("glossary.html")
    sections = re.split(r'<h2 id="letter-([a-z])">', page)[1:]
    letters = sections[0::2]
    for letter, body in zip(letters, sections[1::2]):
        names = re.findall(r'<dt id="[^"]+">([^<]+)</dt>', body)
        assert names and all(name[0].lower() == letter for name in names), letter
    assert letters == sorted(set(letters))
    assert re.findall(r'<li><a href="#letter-([a-z])">', page) == letters


def test_every_term_used_on_a_page_is_in_the_glossary():
    known = {term_id for term_id, _ in TERMS}
    for page in ("learn.html", "guide.html"):
        used = re.findall(r'<a class="term" href="glossary\.html#([^"]+)"', read(page))
        assert used, page
        assert set(used) <= known, page


# ------------------------------------------------- what the guide promises

def test_messages_quoted_in_the_guide_are_ones_the_page_produces():
    produced = {message for _, message in REJECTED.values()}
    produced |= {expected["error"] for _, expected in TEXTS.values() if "error" in expected}
    quoted = [html.unescape(text) for text in re.findall(r'<code class="msg">([^<]+)</code>', read("guide.html"))]
    assert len(quoted) >= 10
    for message in quoted:
        assert message in produced, message


def test_the_guide_lists_every_preset_with_its_title_and_layers():
    rows = re.findall(r'<tr data-preset="([^"]+)"><td>([^<]+)</td><td class="num">(\d+)</td>', read("guide.html"))
    assert [name for name, _, _ in rows] == list(PRESETS)
    for name, title, layers in rows:
        assert title == PRESETS[name].title
        assert int(layers) == recommend(PRESETS[name].build()).params["p"]


def test_the_guide_documents_the_options_the_guided_demo_reads():
    guide, demo = read("guide.html"), read("demo.js")
    for option in ("preset", "budget", "speed", "autoplay", "controls", "captions"):
        assert f'query.get("{option}")' in demo, option
        assert f"<code>{option}=" in guide, option
    for name in PRESETS:
        assert f"<code>{name}</code>" in guide, name


# ------------------------------------------------------------ wording rules

READ_BY_VISITORS = [os.path.join(WEB, page) for page in PAGES] + [os.path.join(ROOT, "README.md")]
NOT_USED = r"\b(robust|leverag|seamless|delv|harness|moreover|furthermore|utili[sz])"


@pytest.mark.parametrize("path", READ_BY_VISITORS, ids=[os.path.basename(path) for path in READ_BY_VISITORS])
def test_wording_rules(path):
    """Plain ASCII, straight quotes, no long dashes, and none of the
    words the project does not use."""
    with open(path, "rb") as handle:
        raw = handle.read()
    assert all(byte < 128 for byte in raw), "not plain ASCII"
    text = raw.decode("ascii")
    for entity in ("&mdash;", "&ndash;", "&ldquo;", "&rdquo;", "&lsquo;", "&rsquo;", "&#8211;", "&#8212;", "&#8216;",
                   "&#8217;", "&#8220;", "&#8221;"):
        assert entity not in text, entity
    found = re.findall(NOT_USED, visible_text(text), flags=re.IGNORECASE)
    assert not found, found
