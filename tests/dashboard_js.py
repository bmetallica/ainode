"""The dashboard's script, as the tests that read it as text expect it.

app.js was split by view into ainode/web/static/js/views/*.js (W2). Tests that
search the source for a string read app.js followed by every view, which is the
same text they read before the split — in the same order the page loads it.
New tests should prefer behaviour: tests/jslib.py and tests/js/.
"""

from __future__ import annotations

from pathlib import Path

JS = Path(__file__).resolve().parent.parent / "ainode" / "web" / "static" / "js"
VIEWS = ("launch", "chat", "downloads", "training", "config", "server")


def parts() -> str:
    """Every view file, concatenated, to append to app.js's text."""
    return "".join("\n" + (JS / "views" / f"{name}.js").read_text()
                   for name in VIEWS)
