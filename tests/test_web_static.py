"""The site in web/ is plain static files: nothing to install, nothing to
build, no request to anywhere else. These checks keep it that way."""

import json
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB = os.path.join(ROOT, "web")
PAGES = sorted(name for name in os.listdir(WEB) if name.endswith(".html"))
SCRIPTS = sorted(name for name in os.listdir(WEB) if name.endswith(".js"))


def read(name: str) -> str:
    with open(os.path.join(WEB, name), encoding="utf-8") as handle:
        return handle.read()


def test_the_site_has_its_two_pages():
    assert PAGES == ["demo.html", "index.html"]


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


def test_guided_demo_uses_the_tested_modules():
    """demo.html loads the same four modules as the main page, the ones
    the parity test runs, and its own script computes no results."""
    for page in PAGES:
        scripts = re.findall(r"<script src=\"([^\"]+)\"", read(page))
        assert scripts[:4] == ["qubo.js", "qaoa.js", "circuit.js", "presets.js"], page
    demo = read("demo.js")
    for own_maths in ("Math.cos", "Math.sin", "Math.exp"):
        assert demo.count(own_maths) <= 2  # layout of the graph only
    assert not re.search(r"density\s*[<>]", demo)  # the depth rule stays in qubo.js
    for call in ("Q.recommend(", "Q.exactSolver(", "Q.bruteForce(", "Q.circuitStats(", "Q.angleScan(", "Q.angleRefiner(",
                 "Q.successProbability(", "Q.topBitstrings("):
        assert call in demo, call
