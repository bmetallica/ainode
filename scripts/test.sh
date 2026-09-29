#!/usr/bin/env bash
# The checks CI runs, locally: ruff, pytest (which also runs the dashboard's
# JavaScript tests through node), and a syntax check of the shell scripts.
# Set up once with scripts/dev-setup.sh. Extra arguments go to pytest:
#   scripts/test.sh -k planner -x
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "!! no virtualenv at $VENV — run scripts/dev-setup.sh first" >&2
    exit 1
fi
cd "$ROOT"

echo "==> ruff"
"$VENV/bin/ruff" check ainode/ tests/

echo "==> shell scripts"
for script in scripts/*.sh; do
    bash -n "$script"
done

echo "==> pytest"
"$VENV/bin/python" -m pytest tests/ -q "$@"
