"""Call the dashboard's pure functions (ainode/web/static/js/lib.js) from a
test, through node. Behaviour instead of a string search through app.js.

Skips when node is not installed; CI has it, and so does scripts/dev-setup.sh.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LIB = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static"
       / "js" / "lib.js")

_SCRIPT = ("const L = require(process.argv[1]);"
           "const out = L[process.argv[2]](...JSON.parse(process.argv[3]));"
           "process.stdout.write(JSON.stringify(out === undefined ? null : out));")


def call(name: str, *args):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    done = subprocess.run([node, "-e", _SCRIPT, str(LIB), name, json.dumps(args)],
                          capture_output=True, text=True, timeout=30)
    if done.returncode != 0:
        raise AssertionError(done.stderr)
    return json.loads(done.stdout)
