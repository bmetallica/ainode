#!/usr/bin/env bash
# One-time setup for working on AINode from a checkout: a virtualenv with the
# test tools, so `scripts/test.sh` runs the same checks CI does.
#
# W4 in upgrade-fixes.md: the hosts had no pytest, CLAUDE.md said
# `pip install -e ".[dev]"` and nothing had ever run it, and the suite was only
# ever exercised in CI or in an ad-hoc venv somewhere in /tmp.
#
# Usage: scripts/dev-setup.sh            (creates ./.venv)
#        VENV=/path/to/venv scripts/dev-setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

say() { printf '==> %s\n' "$*"; }

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "!! Python 3.10 or newer is needed ($("$PYTHON" --version 2>&1))" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import venv, ensurepip' 2>/dev/null; then
    echo "!! $PYTHON has no venv/ensurepip. On Debian/Ubuntu:" >&2
    echo "     sudo apt install python3-venv" >&2
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    say "creating $VENV"
    "$PYTHON" -m venv "$VENV"
fi
say "installing AINode (editable) with the dev tools"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -e "$ROOT[dev]"

if command -v node >/dev/null 2>&1; then
    major="$(node -p 'process.versions.node.split(".")[0]')"
    if [ "$major" -lt 18 ]; then
        echo "!! node $(node --version) is too old for the JS tests (node:test needs 18+)" >&2
    else
        say "node $(node --version) found — the dashboard's JS tests will run too"
    fi
else
    echo "   node is not installed: the JS tests (tests/js) will be skipped." >&2
    echo "   On Debian/Ubuntu: sudo apt install nodejs" >&2
fi

say "done. Run the checks with:  scripts/test.sh"
