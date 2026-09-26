"""Web UI file serving — serves the embedded dashboard.

Reported from the cluster, after an update that landed a dozen server-side
fixes and two UI ones:

    wenn ich werte in den feldern ändere wird der rest aber nach wie vor nicht
    live angepasst

Every change the operator could see was server-side — the plan JSON, the
figures in it. The two that lived in ``app.js`` had not arrived, because the
template asked for ``/static/js/app.js`` with no version on it. aiohttp's
static handler sends Last-Modified and answers a conditional request, but a
browser is not obliged to make one: with no ``Cache-Control`` it may apply
heuristic freshness — a tenth of the file's age — and simply not ask. For a
file a few days old that is hours of serving the previous UI against the new
API, which is the worst of the two possible wrongs, because the numbers look
new and the behaviour is old.

So the asset URLs carry the file's own modification time. It changes exactly
when the file changes, including on a source mount where no version number
moves, and it costs one stat per page load.
"""

from pathlib import Path

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

#: Assets the templates reference and that must never be served stale.
VERSIONED = ("/static/js/app.js", "/static/js/topology.js",
             "/static/css/style.css")


def asset_token(relative: str) -> str:
    """A short string that changes when ``relative`` changes.

    The file's mtime in seconds, base-36. Falls back to the package version
    when the file cannot be stat'ed — a wrong token is better than an
    exception on the one page that would explain why.
    """
    try:
        stamp = int((STATIC_DIR / relative.split("/static/", 1)[1]).stat().st_mtime)
    except (OSError, IndexError):
        try:
            from ainode import __version__

            return str(__version__).replace(".", "")
        except Exception:
            return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while stamp:
        stamp, remainder = divmod(stamp, 36)
        out = digits[remainder] + out
    return out or "0"


def _stamp(html: str) -> str:
    """Put a version on every asset URL the page asks for."""
    for asset in VERSIONED:
        html = html.replace(f'"{asset}"', f'"{asset}?v={asset_token(asset)}"')
    return html


def get_index_html() -> str:
    """Return the main dashboard HTML."""
    index = TEMPLATES_DIR / "index.html"
    return _stamp(index.read_text())


def get_onboarding_html() -> str:
    """Return the onboarding wizard HTML."""
    onboarding = TEMPLATES_DIR / "onboarding.html"
    return _stamp(onboarding.read_text())


def get_static_path() -> Path:
    """Return the path to static assets directory."""
    return STATIC_DIR
