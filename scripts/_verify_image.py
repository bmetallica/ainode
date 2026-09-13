"""Check that the image contains a usable AINode, and fail the build if not.

An empty or stale package passes every test that runs against the repository —
they import from the source tree — and fails only on the node, at boot, in a
restart loop. This runs inside the image, against what was actually installed.
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    import ainode
    import ainode.api.server  # noqa: F401 - import side effects are the test
    import ainode.cli.main  # noqa: F401
    from ainode.models.registry import CURATED_CLUSTER_MODELS

    version = getattr(ainode, "__version__", "")
    if not version or version == "0.0.0+unknown":
        print(f"installed ainode reports version {version!r}", file=sys.stderr)
        return 1
    if not CURATED_CLUSTER_MODELS:
        print("the curated model catalog is empty", file=sys.stderr)
        return 1

    web = pathlib.Path(ainode.__file__).parent / "web"
    for asset in ("templates/index.html", "static/js/app.js", "static/css/style.css"):
        if not (web / asset).exists():
            print(f"web asset missing: {asset}", file=sys.stderr)
            return 1

    print(f"ainode {version}: {len(CURATED_CLUSTER_MODELS)} curated models, "
          f"web assets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
