"""Writes web/package-text.js: the fixed texts of a hardware package.

The page and the Python package put the same README, requirements file
and run script into a package. The texts live on the Python side, in
qopt_builder/package.py and qopt_builder/run_on_ibm.py. This module
turns them into the JavaScript file the page loads:

    python tests/package_text_js.py

tests/test_web_static.py fails when the file on disk is not what this
module would write, and says to run the line above.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..", "web", "package-text.js")

HEADER = """\
/* The fixed texts of a hardware package: its README, its requirements
 * file and its run script.
 *
 * This file is generated. The texts live in qopt_builder/package.py and
 * qopt_builder/run_on_ibm.py, so that the page and the Python package
 * put the same files into a package. To change one, change the Python
 * side and run:  python tests/package_text_js.py
 *
 * Load before package.js. */
(function (global) {
  "use strict";
  const lines = (list) => list.join("\\n") + "\\n";
"""

FOOTER = """
  const api = { packageText: { README_TEMPLATE, REQUIREMENTS, RUN_SCRIPT } };
  global.QOPT = Object.assign(global.QOPT || {}, api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
"""


def block(name: str, text: str) -> str:
    """One text as a JavaScript constant, a source line per line of text."""
    if not text.endswith("\n") or "\r" in text:
        raise ValueError(f"{name} must end with a line break and use unix line ends.")
    rows = "".join(f"    {json.dumps(line)},\n" for line in text[:-1].split("\n"))
    return f"\n  const {name} = lines([\n{rows}  ]);\n"


def expected() -> str:
    """The text web/package-text.js should hold."""
    sys.path.insert(0, os.path.join(HERE, ".."))
    from qopt_builder.package import README_TEMPLATE, REQUIREMENTS, RUN_SCRIPT

    return (
        HEADER
        + block("README_TEMPLATE", README_TEMPLATE)
        + block("REQUIREMENTS", REQUIREMENTS)
        + block("RUN_SCRIPT", RUN_SCRIPT)
        + FOOTER
    )


if __name__ == "__main__":
    with open(TARGET, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(expected())
    print(f"wrote {os.path.normpath(TARGET)}")
