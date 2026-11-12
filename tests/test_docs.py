"""The text pages (primer, glossary, user guide, command line,
hardware) against the code.

The pages quote numbers, messages and names. Every quoted number is
marked in the HTML with a data-fact attribute, and this file recomputes
each one from the package, so a page cannot state a figure the code does
not produce. The values that come from the browser's own code are the
ones in tests/pinned_record.py, which tests/test_web_parity.py checks
under Node.js. Links are followed, the glossary is checked for order,
the messages the guide quotes must be ones the page can produce, and
the project's wording rules are applied to everything a visitor reads.

The command line page shows commands with what they print. Each one is
run here and has to print exactly that. The two reports on the hardware
page come from runs that draw shots at random, so they are checked for
what cannot vary: the costs, the marks, and the sums.
"""

import contextlib
import html
import io
import os
import re
import shlex
from itertools import combinations

import numpy as np
import pytest

from page_text_cases import PAGE_LIMIT, TEXTS
from pinned_record import WEB_DEMO
from qiskit.quantum_info import Statevector

from qopt_builder import PRESETS, build_qaoa, maxcut_qubo, recommend, run_on_ibm, scan_angles
from qopt_builder.cli import main
from qopt_builder.package import README_TEMPLATE, REQUIREMENTS, RUN_SCRIPT
from qopt_builder.presets import DEMO_EDGES
from qopt_builder.problems import TIE_TOLERANCE
from qopt_builder.recommend import EXACT_CLASSICAL_LIMIT
from qopt_builder.simulate import SCAN_SAMPLES, SIMULATION_LIMIT
from validation_cases import REJECTED

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB = os.path.join(ROOT, "web")
DOCS = ["learn.html", "glossary.html", "guide.html", "cli.html", "hardware.html"]
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
    # the circuit the command line packages: the scan's best angles
    scan = scan_angles(problem, p)
    qc_meta = build_qaoa(problem, p)
    values = {qc_meta[1]["gammas"][k]: scan.gammas[k] for k in range(p)}
    values.update({qc_meta[1]["betas"][k]: scan.betas[k] for k in range(p)})
    probabilities = Statevector.from_instruction(qc_meta[0].assign_parameters(values)).probabilities()
    tested = re.search(r"with qiskit ([\d.]+), qiskit-ibm-runtime ([\d.]+) and qiskit-aer ([\d.]+)\. It",
                       " ".join(README_TEMPLATE.split()))
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
        "python-scan": f"{scan.value:.3f}",
        # the hardware package
        "python-ideal-share": f"{100 * probabilities[costs <= optimum + TIE_TOLERANCE].sum():.1f}%",
        "package-shots": str(run_on_ibm.SHOTS),
        "rehearsal-device": run_on_ibm.REHEARSAL_DEVICE,
        "tested-qiskit": tested.group(1),
        "tested-runtime": tested.group(2),
        "tested-aer": tested.group(3),
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
    assert len(QUOTED) > 90
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
        assert {"index.html", "index.html#app", "demo.html", "learn.html", "glossary.html", "guide.html", "cli.html",
                "hardware.html"} <= targets, page
        assert "<script" not in read(page), f"{page} is a text page and loads no script"


def test_the_builder_points_to_the_learning_pages():
    targets = {target for source, target in HREFS if source == "index.html"}
    assert {"learn.html", "glossary.html", "guide.html", "cli.html", "hardware.html"} <= targets
    # each result card, and both input headings, carry a link to their section of the guide
    assert len(re.findall(r'<a class="help" href="guide\.html#[a-z-]+"', read("index.html"))) == 9


# ---------------------------------------------------------------- glossary

TERMS = re.findall(r'<dt id="([^"]+)">([^<]+)</dt>', read("glossary.html"))


def test_glossary_terms_are_in_alphabetical_order():
    names = [name for _, name in TERMS]
    assert len(names) >= 55
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
    for page in ("learn.html", "guide.html", "cli.html", "hardware.html"):
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


# ------------------------------------------------ commands the pages show

def marked_blocks(page: str, attribute: str) -> list:
    """Every <pre> of a page that carries the attribute: (its attributes,
    its text as a visitor reads it)."""
    found = []
    for attributes, body in re.findall(r"<pre([^>]*)>(.*?)</pre>", read(page), flags=re.DOTALL):
        attributes = dict(re.findall(r'([\w-]+)="([^"]*)"', attributes))
        if attribute in attributes:
            attributes = {name: html.unescape(value) for name, value in attributes.items()}
            found.append((attributes, html.unescape(re.sub(r"<[^>]+>", "", body))))
    return found


COMMAND_PAGES = ["cli.html", "hardware.html"]
RUNS = [(page, attributes, body) for page in COMMAND_PAGES for attributes, body in marked_blocks(page, "data-run")]


def test_the_pages_show_commands():
    assert len(RUNS) >= 9
    shown = {attributes["data-run"].split()[0] for _, attributes, _ in RUNS}
    assert shown == {"--version", "run", "package", "presets"}  # every command there is


@pytest.mark.parametrize("page, attributes, body", RUNS, ids=[f"{page}: {a['data-run']}" for page, a, _ in RUNS])
def test_every_command_shown_prints_what_the_page_says(page, attributes, body, tmp_path, monkeypatch, capsys):
    """The command is run in an empty folder that holds the files the
    page shows, and has to print the block under it, to the letter, and
    end with the exit status the page states."""
    for file, content in marked_blocks(page, "data-file"):
        (tmp_path / file["data-file"]).write_text(content + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    lines = body.split("\n")
    assert lines[0] == "python -m qopt_builder " + attributes["data-run"]
    try:
        status = main(shlex.split(attributes["data-run"]))
    except SystemExit as stop:
        status = stop.code
    printed = capsys.readouterr()
    assert status == int(attributes.get("data-exit", 0))
    assert (printed.out + printed.err).rstrip("\n") == "\n".join(lines[1:])


def test_the_python_example_prints_what_the_page_says():
    code, = marked_blocks("cli.html", "data-python")
    output, = marked_blocks("cli.html", "data-output")
    assert code[0]["data-python"] == output[0]["data-output"]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code[1], "the example on cli.html", "exec"), {})
    assert printed.getvalue().rstrip("\n") == output[1]


def test_the_functions_the_command_line_page_lists_exist():
    import qopt_builder

    section = read("cli.html")[read("cli.html").index('id="python"'):]
    listed = re.findall(r"<code>((?:problem\.)?\w+)\(", section[section.index("<table>"):])
    assert len(listed) >= 9
    for name in listed:
        owner = qopt_builder.QUBO if name.startswith("problem.") else qopt_builder
        assert callable(getattr(owner, name.split(".")[-1])), name


def test_every_option_is_in_the_page_and_every_option_in_the_page_exists(capsys):
    """The option tables of the command line page against the program's
    own help, both ways."""
    page = read("cli.html")
    documented = set(re.findall(r"<code>(--[a-z-]+)", page))
    real = set()
    for command in ("run", "package"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        real |= set(re.findall(r"(--[a-z-]+)", capsys.readouterr().out))
    assert documented - {"--version", "--help"} == real - {"--help"}


def test_file_excerpts_are_from_the_package_the_page_makes(tmp_path, capsys):
    """The pieces of circuit.qasm and problem.json on the hardware page
    are pieces of the files that the command shown there writes."""
    assert main(["package", "--preset", "demo", "--out", str(tmp_path / "package")]) == 0
    capsys.readouterr()
    excerpts = marked_blocks("hardware.html", "data-excerpt")
    assert [attributes["data-excerpt"] for attributes, _ in excerpts] == ["circuit.qasm", "problem.json"]
    for attributes, body in excerpts:
        lines = (tmp_path / "package" / attributes["data-excerpt"]).read_text(encoding="utf-8").split("\n")
        shown = body.split("\n")
        if shown[-1] == "...":  # the start of the file
            assert shown[:-1] == lines[:len(shown) - 1]
        else:
            start = lines.index(shown[0])
            assert shown == lines[start:start + len(shown)]


EXAMPLES = dict((attributes["data-example"], body) for attributes, body in marked_blocks("hardware.html", "data-example"))


@pytest.mark.parametrize("name", ["simulate", "rehearse"])
def test_the_reports_on_the_hardware_page_add_up(name):
    """A report on the page is from one run, and its counts of shots
    would differ in another. What cannot differ is checked: the lines
    that describe the package, every cost against the problem, the
    marks on the best answers, and the sums."""
    lines = EXAMPLES[name].split("\n")
    assert lines[0].startswith(f"python run_on_ibm.py --{name} ")
    assert lines[1:3] == ["Package: MaxCut on 6 nodes, 8 edges", "Circuit: QAOA with 3 layers on 6 qubits"]
    problem = maxcut_qubo(DEMO_EDGES, 6)
    optimum = float(FACTS["demo-optimum"])
    ideal_share = float(FACTS["python-ideal-share"].rstrip("%"))

    if name == "simulate":
        assert lines[3] == "Target: an ideal simulator on this computer"
        assert not any(line.startswith("Circuit as sent") for line in lines)
    else:
        assert lines[3].startswith(f"Target: {FACTS['rehearsal-device']}, ")
        assert lines[3].endswith(": a simulated copy of an IBM quantum computer, on this computer")
        sent = re.fullmatch(r"Circuit as sent: depth (\d+), (\d+) two-qubit gates", lines[4])
        # "more than three times as many" two-qubit gates as the builder's circuit, as the page says
        assert int(sent.group(2)) > 3 * int(FACTS["demo-rzz"])

    text = "\n".join(lines)
    assert f"Shots: {FACTS['package-shots']}\n" in text
    assert (f"  An ideal run of this circuit averages {FACTS['python-scan']}. "
            f"Assignments picked at random average {FACTS['demo-mean']}.") in lines
    assert f"  An ideal run gives {FACTS['python-ideal-share']}. Assignments picked at random give {FACTS['chance-start']}." in lines

    rows = [re.fullmatch(r"  x = \[([01 ]+)\] +(\d+) shots  cost (-?\d+\.\d{3})(  optimal)?", line)
            for line in lines[lines.index("Most frequent results") + 1:] if line.startswith("  x = ")]
    assert len(rows) == 8 and all(rows)
    shots = [int(row.group(2)) for row in rows]
    assert shots == sorted(shots, reverse=True)
    at_optimum = 0
    for row in rows:
        x = [int(v) for v in row.group(1).split()]
        assert f"{problem.cost(x):.3f}" == row.group(3)
        assert bool(row.group(4)) == (float(row.group(3)) == optimum)
        at_optimum += int(row.group(2)) if row.group(4) else 0
    assert sum(1 for row in rows if row.group(4)) == int(FACTS["demo-optimal-count"])  # all six best cuts are listed

    counted = re.search(r"Shots at the certified optimum of -8\.000: (\d+) of (\d+) \((\d+\.\d)%\)", text)
    assert int(counted.group(1)) == at_optimum and counted.group(2) == FACTS["package-shots"]
    assert counted.group(3) == f"{100 * at_optimum / int(counted.group(2)):.1f}"
    best = re.search(r"Best assignment found: x = \[([01 ]+)\], cost -8\.000, in (\d+) shots", text)
    assert (best.group(1), int(best.group(2))) == (rows[0].group(1), shots[0])
    assert "  It is a certified optimum." in lines and "  As a cut: weight 8.000" in lines
    average = float(re.search(r"Average cost of the shots \(lower is better\): (-\d+\.\d{3})", text).group(1))
    assert optimum < average < float(FACTS["demo-mean"])
    if name == "rehearse":
        # "the share of shots that reach the optimum has dropped below what an ideal run gives"
        assert float(counted.group(3)) < ideal_share


def test_the_hardware_page_agrees_with_the_package():
    """The files, the commands and the options the page names are the
    ones a package holds and its script takes."""
    from qopt_builder.package import FILES

    page = read("hardware.html")
    assert re.findall(r'<td class="keep"><code>([\w.]+)</code></td>', page)[:5] == list(FILES)
    for flag in ("--simulate", "--rehearse", "--hardware", "--backend NAME", "--shots N", "--job ID", "--seed S",
                 "--out FILE"):
        assert f"<code>{flag}</code>" in page or f"python run_on_ibm.py {flag}" in page, flag
        assert flag.split()[0] in RUN_SCRIPT
    assert "run_on_ibm.run(backend, shots=1024)" in page and "def run(backend=None, shots: int = SHOTS" in RUN_SCRIPT
    assert "It has not been run on a real quantum computer." in page
    minimums = dict(line.split(">=") for line in REQUIREMENTS.split("\n") if line and not line.startswith("#"))
    for package, fact in (("qiskit", "tested-qiskit"), ("qiskit-ibm-runtime", "tested-runtime"), ("qiskit-aer", "tested-aer")):
        assert FACTS[fact].startswith(minimums[package] + ".")  # the package asks for the versions it was tested with


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


@pytest.mark.parametrize("text", [README_TEMPLATE, REQUIREMENTS, RUN_SCRIPT], ids=["README.txt", "requirements.txt",
                                                                                 "run_on_ibm.py"])
def test_wording_rules_in_the_hardware_package(text):
    """The files a package carries are read by its user, so the same
    rules apply to them."""
    text.encode("ascii")
    found = re.findall(NOT_USED, text, flags=re.IGNORECASE)
    assert not found, found
