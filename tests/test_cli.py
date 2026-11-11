"""run_demo.py from the command line: the preset flags and the budget flag.

examples/presets_output.txt is what `python run_demo.py --all` prints.
It is compared line for line here, so the numbers in that file are the
numbers this machine produces.
"""

import os

import pytest

import run_demo
from qopt_builder import PRESETS, maxcut_qubo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(ROOT, "examples", "presets_output.txt"), encoding="utf-8") as handle:
    EXPECTED = handle.read()
BLOCKS = {block.splitlines()[0].split("]")[0].lstrip("["): block.splitlines()
          for block in EXPECTED.strip().split("\n\n")}


def output(capsys, *args) -> list[str]:
    run_demo.main(list(args))
    return capsys.readouterr().out.splitlines()


def test_examples_file_covers_every_preset():
    assert list(BLOCKS) == list(PRESETS)


def test_all_matches_the_examples_file(capsys):
    assert output(capsys, "--all") == EXPECTED.splitlines()


@pytest.mark.parametrize("name", ["ring-10", "chorded-ring-24", "vertex-cover-6"])
def test_one_preset_prints_its_block(capsys, name):
    assert output(capsys, "--preset", name) == BLOCKS[name]


def test_no_arguments_is_the_demo_record_without_a_header(capsys):
    assert output(capsys) == BLOCKS["demo"][1:]


def test_list_shows_every_preset(capsys):
    lines = output(capsys, "--list")
    for name, preset in PRESETS.items():
        assert any(line.startswith(name) and line.endswith(preset.title) for line in lines)


def test_budget_below_n_prints_the_refusal(capsys):
    assert output(capsys, "--budget", "4") == [
        "Recommendation: none {'n': 6}",
        " - n=6 <= 24: an exact classical solve is cheap and certifies the quantum result; "
        "recommending exact with QAOA built alongside for comparison.",
        " - n=6 exceeds the qubit budget of 4: no full-width circuit fits; the builder will "
        "refuse rather than silently truncate.",
        "No circuit built.",
    ]


def test_budget_that_fits_changes_nothing(capsys):
    assert output(capsys, "--budget", "6") == BLOCKS["demo"][1:]


def test_unknown_preset_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_demo.main(["--preset", "nope"])
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "invalid choice: 'nope'" in message
    assert all(name in message for name in PRESETS)


def test_preset_and_all_cannot_be_combined(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_demo.main(["--all", "--preset", "demo"])
    assert exit_info.value.code == 2
    capsys.readouterr()


def test_beyond_both_limits_nothing_is_solved_or_simulated(capsys):
    """n = 26: past the exact-solve limit and the simulation limit. The
    recommendation and the circuit are still produced, and the output
    says plainly what was not computed."""
    run_demo.run(maxcut_qubo([(i, i + 1, 1.0) for i in range(25)], 26))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Recommendation: qaoa {'p': 1}"
    assert lines[1] == " - n=26: exact enumeration is not feasible (2^26 states)."
    assert lines[3] == "Brute-force optimum: not computed (beyond the exact-solve limit)"
    assert lines[4] == "QAOA circuit: 26 qubits, depth 27, params 2"
    assert lines[5] == "Best scanned expectation: not simulated (n=26 is above the statevector limit of 14)"
