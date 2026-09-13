"""Print the project's dependencies, one per line, for `pip install -r`.

Exists so the image can install dependencies WITHOUT building the project.
The previous approach installed a stub package — same name, same version,
different contents — and both pip's wheel cache and Docker's layer cache key
on name and version. Either one could hand the stub back in place of the real
package, and one of them did: an image shipped an `ainode` module consisting
of an empty __init__.py, and every node crash-looped with

    ImportError: cannot import name '__version__' from 'ainode'

No stub, no collision. The dependency layer never sees our source, so editing
a Python file cannot invalidate it, which is the whole point.

Usage: python scripts/_deps_from_pyproject.py [pyproject.toml] [extras]
       extras is "[embeddings]" or "embeddings,training" or empty.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def dependencies(pyproject: Path, extras: str = "") -> list:
    project = tomllib.loads(pyproject.read_text())["project"]
    requirements = list(project.get("dependencies") or [])
    optional = project.get("optional-dependencies") or {}
    wanted = [name.strip() for name in extras.strip().strip("[]").split(",")]
    for name in wanted:
        if not name:
            continue
        if name not in optional:
            raise SystemExit(
                f"pyproject.toml has no optional-dependencies group {name!r}. "
                f"Known groups: {', '.join(sorted(optional)) or 'none'}."
            )
        requirements += list(optional[name])
    return requirements


def main(argv: list) -> int:
    pyproject = Path(argv[1]) if len(argv) > 1 else Path("pyproject.toml")
    extras = argv[2] if len(argv) > 2 else ""
    for requirement in dependencies(pyproject, extras):
        print(requirement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
