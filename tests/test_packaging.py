"""pyproject.toml against the code.

pyproject.toml is what `pip install qopt-builder` goes by: the name, the
version, the packages that are installed with it, and the qopt-builder
command. Each of those is stated somewhere else as well: the version in
qopt_builder/__init__.py, the imports in the code, requirements.txt for
work from a clone, and the requirements file that every hardware package
carries. This file keeps them in step, so the installed package cannot
ask for something other than what the code uses.

Nothing here builds or installs anything. What the built files hold is
checked when a release is made.
"""

import ast
import importlib
import os
import re
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import qopt_builder
from qopt_builder import cli
from qopt_builder.package import REQUIREMENTS, RUN_SCRIPT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE = os.path.join(ROOT, "qopt_builder")

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
    PYPROJECT = tomllib.load(handle)
PROJECT = PYPROJECT["project"]
HATCH = PYPROJECT["tool"]["hatch"]
REQUIRED = PROJECT["dependencies"]
EXTRAS = PROJECT["optional-dependencies"]


def floors(specs) -> dict:
    """{"qiskit": "2.6"} from ["qiskit>=2.6"]."""
    return dict(spec.split(">=") for spec in specs)


def module_name(distribution: str) -> str:
    """The name a package is imported by: qiskit_aer for qiskit-aer."""
    return distribution.replace("-", "_")


# ------------------------------------------------- the name and the command

def test_the_name_and_the_command(capsys):
    """One command, qopt-builder, and it is the main() that
    `python -m qopt_builder` runs. --version prints the name the package
    has on PyPI."""
    assert PROJECT["name"] == "qopt-builder"
    assert PROJECT["scripts"] == {"qopt-builder": "qopt_builder.cli:main"}
    module, function = PROJECT["scripts"]["qopt-builder"].split(":")
    assert getattr(importlib.import_module(module), function) is cli.main
    with open(os.path.join(PACKAGE, "__main__.py"), encoding="utf-8") as handle:
        assert "from .cli import main" in handle.read()
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out == f"{PROJECT['name']} {qopt_builder.__version__}\n"


def test_the_version_is_written_in_one_place():
    """pyproject.toml gives no version of its own. The build reads the
    one in qopt_builder/__init__.py."""
    assert "version" not in PROJECT and PROJECT["dynamic"] == ["version"]
    assert HATCH["version"] == {"path": "qopt_builder/__init__.py"}
    with open(os.path.join(ROOT, HATCH["version"]["path"]), encoding="utf-8") as handle:
        written = re.findall(r'^__version__ = "([^"]+)"$', handle.read(), flags=re.MULTILINE)
    assert written == [qopt_builder.__version__]
    assert re.fullmatch(r"\d+\.\d+\.\d+", qopt_builder.__version__)


# ------------------------------------------------- what is installed with it

def imported(node, when_imported: bool) -> set:
    """The top-level modules a piece of code imports. With when_imported,
    only the imports that run when the module itself is imported: those
    inside a function are left out."""
    found = set()
    for child in ast.iter_child_nodes(node):
        if when_imported and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Import):
            found.update(alias.name.split(".")[0] for alias in child.names)
        elif isinstance(child, ast.ImportFrom) and child.level == 0:
            found.add(child.module.split(".")[0])
        found |= imported(child, when_imported)
    return found


def from_other_projects(when_imported: bool) -> set:
    """What the package imports that is neither its own nor Python's."""
    found = set()
    for name in sorted(os.listdir(PACKAGE)):
        if name.endswith(".py"):
            with open(os.path.join(PACKAGE, name), encoding="utf-8") as handle:
                found |= imported(ast.parse(handle.read()), when_imported)
    return found - set(sys.stdlib_module_names) - {"qopt_builder"}


def test_what_is_installed_is_what_the_code_imports():
    """Importing qopt_builder needs the required packages and nothing
    else. The packages of the `ibm` extra are imported only inside the
    functions that talk to IBM or to a simulated copy of an IBM device,
    and the script asks for each of them by name when it is missing."""
    assert sorted(floors(REQUIRED)) == ["numpy", "qiskit"]
    assert from_other_projects(when_imported=True) == {"numpy", "qiskit"}
    assert sorted(EXTRAS) == ["ibm", "test"]
    assert sorted(floors(EXTRAS["ibm"])) == ["qiskit-aer", "qiskit-ibm-runtime"]
    assert from_other_projects(when_imported=False) == {"numpy", "qiskit", "qiskit_ibm_runtime"}
    for distribution in floors(EXTRAS["ibm"]):
        assert f'need("{module_name(distribution)}", "{distribution}")' in RUN_SCRIPT
    assert [spec.split(";")[0].strip() for spec in EXTRAS["test"]] == ["pytest", "tomli"]


def test_requirements_txt_asks_for_the_same():
    """requirements.txt is for work from a clone. It lists the required
    packages exactly as pyproject.toml does."""
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    assert lines == REQUIRED


def test_the_floors_are_the_ones_a_hardware_package_asks_for():
    """A hardware package carries a requirements file of its own. The
    versions it asks for are the lowest ones pyproject.toml accepts, so
    the two never name different versions of the same package."""
    carried = floors(line for line in REQUIREMENTS.split("\n") if line and not line.startswith("#"))
    declared = {**floors(REQUIRED), **floors(EXTRAS["ibm"])}
    assert sorted(carried) == ["qiskit", "qiskit-aer", "qiskit-ibm-runtime"]
    assert carried == {name: declared[name] for name in carried}


def test_python_versions():
    """The Python versions named in the classifiers start at the lowest
    one accepted and leave no gap."""
    assert PROJECT["requires-python"] == ">=3.10"
    named = [entry.rsplit(" :: ", 1)[1] for entry in PROJECT["classifiers"]
             if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", entry)]
    assert named and named == [f"3.{10 + step}" for step in range(len(named))]


# --------------------------------------------------- the rest of the record

def test_who_and_under_what_licence():
    """The author by name only, so no address goes on PyPI. The MIT
    licence, given once: a licence classifier beside it is an error for
    the build. The README is the page PyPI shows."""
    assert PROJECT["authors"] == [{"name": "Muntaser Syed"}]
    assert PROJECT["license"] == "MIT" and PROJECT["license-files"] == ["LICENSE"]
    with open(os.path.join(ROOT, "LICENSE"), encoding="utf-8") as handle:
        licence = handle.read()
    assert licence.startswith("MIT License\n") and "Muntaser Syed" in licence
    assert not [entry for entry in PROJECT["classifiers"] if entry.startswith("License ::")]
    assert PROJECT["readme"] == "README.md" and os.path.isfile(os.path.join(ROOT, "README.md"))


def test_what_the_two_archives_hold():
    """The wheel is the package and nothing else. The source archive adds
    what the tests need: the tests, the site they compare the package
    with, the demo script, and the files at the top of the repository
    that tests read."""
    targets = HATCH["build"]["targets"]
    assert targets["wheel"] == {"packages": ["qopt_builder"]}
    listed = targets["sdist"]["include"]
    for entry in listed:
        assert entry.startswith("/") and os.path.exists(os.path.join(ROOT, entry[1:])), entry
    for needed in ("/qopt_builder", "/tests", "/web", "/examples", "/run_demo.py", "/requirements.txt",
                   "/vercel.json"):
        assert needed in listed, needed


def test_pyproject_gives_the_site_nothing_to_run():
    """The site is a static folder (tests/test_web_static.py). A
    [tool.vercel] table here would name a Python program for Vercel to
    run in its place."""
    assert "vercel" not in PYPROJECT["tool"]
