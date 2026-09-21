"""Is there anything new on our own main?

The existing update path compares this node's VERSION against the latest
published image tag. That is the right question for a deployment that pulls
images, and the wrong one for this cluster: it builds on the head from a
checkout of a fork, where the code moves far more often than the version
number does. A node can be forty commits behind and report itself current.

So the question asked here is the other one: **which commit was this image
built from, and how far is the fork's branch ahead of it?** The commit is
baked in at build time (``AINODE_GIT_SHA``); the branch head comes from
GitHub, which needs no credentials for a public repo and no checkout on this
node.

Bound to the fork, not to upstream. Which fork is configuration, because a
fork of a fork has a different answer and hard-coding one would make this
useful to exactly one person.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["SourceState", "built_from", "check_for_updates", "CACHE_SECONDS"]

#: How long an answer stays good. The UI checks hourly; a manual check passes
#: force=True and skips this.
CACHE_SECONDS = 3600

#: GitHub's unauthenticated limit is 60 requests an hour per address. One an
#: hour per node leaves room for manual checks and for three nodes behind one
#: NAT, which is this cluster.
_TIMEOUT = 15

#: How many subjects to carry back. Enough to see what is coming, few enough
#: that the payload stays a payload.
_MAX_COMMITS = 20


@dataclass
class SourceState:
    repo: str = ""
    branch: str = "main"
    #: The commit this image was built from, or "" when it was not recorded —
    #: an image built before this existed, which is not an error and must not
    #: read as one.
    current: str = ""
    latest: str = ""
    behind: int = 0
    update_available: bool = False
    commits: List[dict] = field(default_factory=list)
    checked_at: float = 0.0
    #: Why the check could not be made, when it could not. Never a raised
    #: exception: a node that cannot reach GitHub is still a working node.
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def built_from() -> str:
    """The commit this image was built from, or ""."""
    value = (os.environ.get("AINODE_GIT_SHA") or "").strip()
    return "" if value in ("", "unknown") else value


def _get(url: str) -> Optional[object]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "ainode-update-check"})
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        # Optional: only raises the rate limit. A public fork needs none.
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
            TimeoutError, OSError) as exc:
        logger.debug("GitHub request failed: %s", exc)
        raise


def check_for_updates(repo: str, branch: str = "main",
                      current: Optional[str] = None) -> SourceState:
    """Compare the built commit against the fork's branch head."""
    state = SourceState(repo=repo, branch=branch,
                        current=current if current is not None else built_from(),
                        checked_at=time.time())
    if not repo or "/" not in repo:
        state.error = (f"{repo!r} is not an owner/name repository. Set it in "
                       f"Settings → Updates.")
        return state
    try:
        head = _get(f"https://api.github.com/repos/{repo}/commits/{branch}")
    except Exception as exc:
        state.error = f"could not reach GitHub: {type(exc).__name__}: {exc}"
        return state
    if not isinstance(head, dict) or not head.get("sha"):
        state.error = f"{repo}@{branch} did not answer with a commit"
        return state
    state.latest = str(head["sha"])

    if not state.current:
        # An image built before the commit was recorded. Saying "up to date"
        # would be a guess and saying "behind" would be a different one, so
        # say neither.
        state.error = ("this image does not record the commit it was built "
                       "from, so it cannot be compared — the next update "
                       "fixes that")
        return state
    if state.current == state.latest:
        return state

    try:
        comparison = _get(
            f"https://api.github.com/repos/{repo}/compare/"
            f"{state.current}...{state.latest}")
    except Exception as exc:
        # The head is known and differs; that alone is worth reporting.
        state.update_available = True
        state.error = f"could not list the commits: {exc}"
        return state
    if isinstance(comparison, dict):
        state.behind = int(comparison.get("ahead_by") or 0)
        state.update_available = state.behind > 0
        for commit in (comparison.get("commits") or [])[-_MAX_COMMITS:]:
            message = ((commit.get("commit") or {}).get("message") or "")
            state.commits.append({
                "sha": str(commit.get("sha") or "")[:8],
                # First line only: a commit body here would be several
                # screens per entry.
                "subject": message.splitlines()[0][:120] if message else "",
            })
        state.commits.reverse()
    return state
