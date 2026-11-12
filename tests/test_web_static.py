"""The site in web/ is plain static files: nothing to install, nothing to
build, no request to anywhere else. These checks keep it that way."""

import json
import os
import re

import pytest

from package_text_js import expected as expected_package_text
from pinned_record import PAGE_ANGLES_FROM
from qopt_builder.simulate import SCAN_SAMPLES, SCAN_SEED

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB = os.path.join(ROOT, "web")
PAGES = sorted(name for name in os.listdir(WEB) if name.endswith(".html"))
SCRIPTS = sorted(name for name in os.listdir(WEB) if name.endswith(".js"))


def read(name: str) -> str:
    with open(os.path.join(WEB, name), encoding="utf-8") as handle:
        return handle.read()


APP_PAGES = ["demo.html", "index.html"]  # the two pages that compute; the rest are text


def test_the_site_has_its_pages():
    assert PAGES == ["cli.html", "demo.html", "glossary.html", "guide.html", "hardware.html", "index.html",
                     "learn.html"]


@pytest.mark.parametrize("page", PAGES)
def test_pages_load_only_files_that_ship_with_the_site(page):
    """Every script and stylesheet a page loads is a file in web/."""
    text = read(page)
    loaded = re.findall(r"<script[^>]*\ssrc=\"([^\"]+)\"", text) + re.findall(r"<link[^>]*\shref=\"([^\"]+)\"", text)
    assert loaded, page
    for target in loaded:
        assert not re.match(r"^(https?:)?//", target), f"{page} loads {target} from another site"
        assert os.path.isfile(os.path.join(WEB, target)), f"{page} loads {target}, which is not in web/"
    assert "<iframe" not in text and "@import" not in text


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_fetch_nothing(script):
    """No network calls and no module loading: the pages work from a
    folder on disk as well as from a server."""
    text = read(script)
    for call in ("fetch", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts", "sendBeacon"):
        assert not re.search(r"\b" + call + r"\s*\(", text), f"{script} calls {call}"
    assert not re.search(r"^\s*import\s", text, flags=re.MULTILINE), script
    assert "require(" not in text, script


def test_stylesheets_fetch_nothing():
    for name in sorted(os.listdir(WEB)):
        if name.endswith(".css"):
            assert "url(" not in read(name) and "@import" not in read(name), name


def test_deployment_is_a_static_folder():
    """vercel.json serves web/ as it is: no build step, no framework, and
    no serverless functions in the repository."""
    with open(os.path.join(ROOT, "vercel.json"), encoding="utf-8") as handle:
        assert json.load(handle) == {"buildCommand": None, "framework": None, "outputDirectory": "web"}
    assert not os.path.exists(os.path.join(ROOT, "api"))
    assert not os.path.exists(os.path.join(ROOT, "package.json"))


def test_the_two_pages_link_to_each_other():
    """The main page offers the guided demo, and the guided demo offers
    the way back: the brand in its header, a link in its controls, and a
    link on its title and end cards, which cover the other two."""
    assert 'href="demo.html' in read("index.html")
    assert read("demo.html").count('href="index.html"') == 2
    demo = read("demo.js")
    assert 'const HOME = "index.html";' in demo
    assert demo.count('"Back to the main page", HOME') == 2
    assert 'const PRIMER = "learn.html";' in demo and demo.count('"Read the primer", PRIMER') == 2
    assert 'href="learn.html"' in read("index.html")


def test_pages_read_their_inputs_through_the_tested_parsers():
    """app.js does no parsing of its own: what is typed into the page goes
    through the functions that tests/test_web_parity.py pins, and so
    does the budget box of the guided demo."""
    app = read("app.js")
    for call in ("Q.parseEdgeList(", "Q.parseMatrix(", "Q.parseOffset(", "Q.parseBudget("):
        assert call in app, call
    for own in ("parseInt", "parseFloat", ".split(", "Number("):
        assert own not in app, own
    demo = read("demo.js")
    assert "Q.parseBudget(" in demo and "parseInt" not in demo


def test_guided_demo_uses_the_tested_modules():
    """demo.html loads the same four modules as the main page, the ones
    the parity test runs, and its own script computes no results."""
    for page in APP_PAGES:
        scripts = re.findall(r"<script src=\"([^\"]+)\"", read(page))
        assert scripts[:4] == ["qubo.js", "qaoa.js", "circuit.js", "presets.js"], page
    demo = read("demo.js")
    for own_maths in ("Math.cos", "Math.sin", "Math.exp"):
        assert demo.count(own_maths) <= 2  # layout of the graph only
    assert not re.search(r"density\s*[<>]", demo)  # the depth rule stays in qubo.js
    for call in ("Q.recommend(", "Q.exactSolver(", "Q.bruteForce(", "Q.circuitStats(", "Q.angleScan(", "Q.angleRefiner(",
                 "Q.successProbability(", "Q.topBitstrings("):
        assert call in demo, call


def test_the_package_texts_on_the_page_are_the_python_ones():
    """web/package-text.js is written by tests/package_text_js.py from
    the README, the requirements and the run script on the Python side.
    If this fails, one side was changed without the other: run
    python tests/package_text_js.py"""
    assert read("package-text.js") == expected_package_text()


def test_the_page_builds_its_package_with_the_tested_module():
    """The main page loads the package module after the ones it needs,
    and its own script only passes the run on screen to it."""
    scripts = re.findall(r"<script src=\"([^\"]+)\"", read("index.html"))
    assert scripts == ["qubo.js", "qaoa.js", "circuit.js", "presets.js", "package-text.js", "package.js", "app.js"]
    app = read("app.js")
    assert "Q.zipBytes(Q.packageFiles(packageInput))" in app
    assert f'anglesFrom: "{PAGE_ANGLES_FROM}"' in app
    # the note says what the page does: this many settings, this seed, then the refinement
    assert f"Q.scanAngles(ising, n, rec.params.p, {SCAN_SAMPLES}, {SCAN_SEED})" in app
    assert f"the best of {SCAN_SAMPLES} random settings (seed {SCAN_SEED}), then refined" in PAGE_ANGLES_FROM
    assert "gammas: refined.gammas, betas: refined.betas" in app
    for own in ("crc", "DataView", "OPENQASM"):  # no archive or circuit text of its own
        assert own not in app, own
    assert "package.js" not in read("demo.html")  # the guided demo offers no download
