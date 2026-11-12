"""Shared test setup: make the repository root importable, and keep the
machine's logging setup out of the tests.

`python -m pytest tests` already puts the working directory on sys.path.
This keeps a bare `pytest` call, or a call from another directory,
working the same way.
"""

import logging
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True, scope="session")
def no_log_records_below_warning():
    """Switch off INFO and DEBUG log records for the test run.

    Qiskit logs the time of every transpiler pass at INFO level. Whether
    anything comes of that depends on how logging is set up on the
    machine, which is not this project's business. Where something has
    set the root logger to INFO before pytest starts (a pytest plugin
    that calls logging.basicConfig() when it is imported will do it),
    the records are printed after the test summary, and on Windows the
    log handler made before pytest took over the console fails on every
    record and writes a traceback to the error stream. A test that
    expects only this project's own message on that stream then fails,
    for a reason that has nothing to do with the code under test.

    No test here reads a log record, and qopt_builder writes none, so
    the suite runs with those records switched off.
    """
    before = logging.root.manager.disable
    logging.disable(logging.INFO)
    yield
    logging.disable(before)
