"""A JSON value typed without single quotes keeps its double quotes.

Typed into the wizard as ``--model-loader-extra-config {"read_mode":"bounce"}``,
it reached vLLM as ``{read_mode:bounce}`` and the launch died:
"Value {read_mode:bounce} cannot be converted to <function loads …>".
"""

from __future__ import annotations

import json

import pytest

from ainode.engine.serve_args import split_command_line
from ainode.planner.household import _args


def test_bare_json_survives():
    args = split_command_line('--model-loader-extra-config {"read_mode":"bounce"}')
    assert args == ["--model-loader-extra-config", '{"read_mode":"bounce"}']
    assert json.loads(args[1]) == {"read_mode": "bounce"}


def test_single_quoted_json_is_unquoted_as_before():
    assert split_command_line("""--compilation-config '{"a": {"b": true}}' --x""") == [
        "--compilation-config", '{"a": {"b": true}}', "--x"]


def test_ordinary_words_and_quotes():
    assert split_command_line('--max-num-seqs 8 --name "a b" drop:--load-format') == [
        "--max-num-seqs", "8", "--name", "a b", "drop:--load-format"]


def test_the_wizard_path_uses_it():
    assert _args('--x {"k":"v"}') == ["--x", '{"k":"v"}']


def test_an_unbalanced_quote_still_raises():
    with pytest.raises(ValueError):
        split_command_line("--x 'oops")
